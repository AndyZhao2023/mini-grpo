"""Run one full-parameter GRPO update on the first selected mixed-reward group."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import MathProblem, build_prompt_messages, load_pinned_train
from src.data.schema import Trajectory, read_jsonl, validate_group
from src.grpo.advantage import normalize_group_advantages
from src.grpo.trainer import StepMetrics, run_one_grpo_step
from src.reward.math_verifier import verify_answer


def _validate_smoke_config(config: dict, checkpoint_dir: Path) -> None:
    """Reject unsupported training settings before reading rollout artifacts."""
    model_config, algorithm, optimizer_config = (
        config["model"], config["algorithm"], config["optimizer"]
    )
    if model_config["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    if model_config["dtype"] != "float32" or model_config["training_method"] != "full_finetune":
        raise ValueError("M0 smoke requires full-parameter float32 training")
    if algorithm["beta"] != 0.0 or config["smoke"]["grpo_update_epochs"] != 1:
        raise ValueError("M0 smoke requires beta=0 and one update epoch")
    if optimizer_config["name"] != "AdamW":
        raise ValueError("M0 smoke requires AdamW")
    if checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint directory already exists: {checkpoint_dir}")


def _load_verified_group(
    config: dict, config_bytes: bytes, input_path: Path, scan_path: Path,
) -> tuple[list[Trajectory], MathProblem]:
    """Match a complete selected group to its scan record and pinned training row."""
    model_config, algorithm = config["model"], config["algorithm"]
    group = read_jsonl(input_path)
    validate_group(group, require_complete=True)
    if group[0].group_size != config["generation"]["group_size"]:
        raise ValueError("trajectory group size differs from config")
    if group[0].policy_version != "step-000000":
        raise ValueError("smoke group must come from the initial policy")
    if (
        group[0].model_revision != model_config["revision"]
        or group[0].tokenizer_revision != model_config["tokenizer_revision"]
    ):
        raise ValueError("trajectory model/tokenizer revision differs from config")
    split = load_pinned_train(config["data"])
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    if (
        scan["source_sha256"] != split.source_sha256
        or scan["selected_prompt_id"] != group[0].prompt_id
        or scan["model_revision"] != model_config["revision"]
        or scan.get("config_sha256") != hashlib.sha256(config_bytes).hexdigest()
    ):
        raise ValueError("scan record does not match the selected group and pinned sources")
    training_by_id = {problem.prompt_id: problem for problem in split.training}
    problem = training_by_id.get(group[0].prompt_id)
    if problem is None or problem.question != group[0].prompt_text:
        raise ValueError("selected group is not from the pinned training partition")
    for item in group:
        checked = verify_answer(item.response_text, problem.reference_answer)
        if (item.reward, item.reward_reason) != (checked.reward, checked.reason):
            raise ValueError("stored reward differs from the deterministic verifier")
    expected = normalize_group_advantages(
        [replace(item, advantage=None) for item in group], algorithm,
    ).group
    if any(abs(item.advantage - check.advantage) > 1e-6 for item, check in zip(group, expected)):
        raise ValueError("stored advantage differs from group normalization")
    if len({item.reward for item in group}) < 2:
        raise ValueError("one-step smoke requires mixed rewards")
    return group, problem


def _load_initial_policy(config: dict, problem: MathProblem, group: list[Trajectory]):
    """Load the pinned tokenizer/model and confirm the saved prompt tokens."""
    torch.manual_seed(config["seed"])
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["tokenizer_revision"],
        trust_remote_code=False,
    )
    expected_prompt_ids = tokenizer.apply_chat_template(
        build_prompt_messages(problem, config["prompt"]["system"]),
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )[0].tolist()
    if group[0].prompt_token_ids != expected_prompt_ids:
        raise ValueError("trajectory prompt tokens differ from the configured template")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"], revision=model_config["revision"],
        dtype=torch.float32, trust_remote_code=False,
    ).to(model_config["device"])
    return model, tokenizer, pad_token_id


def _write_report(
    config: dict, group: list[Trajectory], metrics: StepMetrics,
    started: float, checkpoint_dir: Path, output_path: Path, scan_path: Path,
) -> None:
    """Write the step metrics and the location of the saved model checkpoint."""
    model_config, optimizer_config = config["model"], config["optimizer"]
    record = {
        "status": "passed",
        "prompt_id": group[0].prompt_id,
        "policy_before": group[0].policy_version,
        "policy_after": "step-000001",
        "model_revision_before": model_config["revision"],
        "reward_vector": [item.reward for item in group],
        "advantage_vector": [item.advantage for item in group],
        "group_size": len(group),
        "update_epochs": 1,
        "data_age_updates": 0,
        "optimizer": optimizer_config,
        "metrics": asdict(metrics),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_dir": str(checkpoint_dir),
        "optimizer_state_saved": False,
        "scan_record": str(scan_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"GRPO step complete: prompt={record['prompt_id']}, "
        f"grad_norm={metrics.gradient_norm_before_clip:.6g}, "
        f"post_kl={metrics.post_approx_kl_old:.6g} -> {output_path}"
    )


def main(
    config_path: Path, input_path: Path, scan_path: Path,
    output_path: Path, checkpoint_dir: Path,
) -> None:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    _validate_smoke_config(config, checkpoint_dir)
    group, problem = _load_verified_group(config, config_bytes, input_path, scan_path)
    model, tokenizer, pad_token_id = _load_initial_policy(config, problem, group)
    model_config, algorithm, optimizer_config = (
        config["model"], config["algorithm"], config["optimizer"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]), eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )
    started = time.perf_counter()
    metrics = run_one_grpo_step(
        model, optimizer, group,
        pad_token_id=pad_token_id, device=model_config["device"],
        clip_epsilon=algorithm["clip_epsilon"],
        loss_reduction=algorithm["loss_reduction"],
        max_grad_norm=optimizer_config["max_grad_norm"],
    )
    checkpoint_dir.mkdir(parents=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True, max_shard_size="1GB")
    tokenizer.save_pretrained(checkpoint_dir)
    _write_report(config, group, metrics, started, checkpoint_dir, output_path, scan_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0.yaml"))
    parser.add_argument("--input", type=Path, default=Path("results/m0/train-smoke-advantaged.jsonl"))
    parser.add_argument("--scan", type=Path, default=Path("results/m0/train-smoke-scan.json"))
    parser.add_argument("--output", type=Path, default=Path("results/m0/train-smoke.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/m0/checkpoints/step-000001"))
    args = parser.parse_args()
    main(args.config, args.input, args.scan, args.output, args.checkpoint_dir)
