"""Run pinned data preparation, fresh GRPO updates, and fixed validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.find_trainable_group import order_training_problems, scan_for_mixed_group
from src.data.dataset import TrainValidationSplit, build_manifest, load_pinned_train
from src.data.schema import validate_group, write_jsonl
from src.evaluation import evaluate_model
from src.grpo.trainer import run_one_grpo_step
from src.rollout.hf_rollout import sample_group


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _validate_run(config: dict, split: TrainValidationSplit) -> None:
    run = config["run"]
    update_epochs = config["algorithm"].get("update_epochs", 1)
    if type(update_epochs) is not int or update_epochs not in (1, 2):
        raise ValueError("algorithm.update_epochs must be 1 or 2")
    for name in ("update_steps", "max_prompts_per_step"):
        if type(run[name]) is not int or run[name] < 1:
            raise ValueError(f"run.{name} must be a positive integer")
    if run["update_steps"] % update_epochs:
        raise ValueError("run.update_steps must be divisible by algorithm.update_epochs")
    if run["order"] not in ("original", "shortest"):
        raise ValueError("run.order must be original or shortest")
    bound = run["max_post_approx_kl_old"]
    if type(bound) not in (int, float) or not math.isfinite(bound) or bound <= 0:
        raise ValueError("run.max_post_approx_kl_old must be finite and positive")
    if not split.training or not split.validation:
        raise ValueError("both training and validation partitions must be nonempty")
    if {item.prompt_id for item in split.training} & {item.prompt_id for item in split.validation}:
        raise ValueError("training and validation prompt IDs must be disjoint")


def run_cycles(
    model, tokenizer, optimizer, split: TrainValidationSplit, config: dict,
    output_dir: Path, *, config_sha256: str,
    sample_fn=sample_group, scan_fn=scan_for_mixed_group,
    update_fn=run_one_grpo_step, evaluation_fn=evaluate_model,
) -> dict:
    """Run fresh policy versions in one process, preserving partial evidence."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    _validate_run(config, split)
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "dataset-split.json",
        build_manifest(split, config["data"], config["prompt"]["system"]),
    )
    run = config["run"]
    initial_rows, baseline = evaluation_fn(model, tokenizer, split.validation, config)
    _write_rows(output_dir / "validation-initial.jsonl", initial_rows)
    _write_json(output_dir / "validation-initial.json", baseline)

    ordered = order_training_problems(split.training, run["order"])
    update_epochs = config["algorithm"].get("update_epochs", 1)
    selected_ids: set[str] = set()
    steps: list[dict] = []
    status = "completed"
    selected = None
    rollout_version = None
    rollout_artifact_dir = None
    for index in range(run["update_steps"]):
        before_version = f"step-{index:06d}"
        after_version = f"step-{index + 1:06d}"
        step_dir = output_dir / "steps" / after_version
        update_epoch = index % update_epochs + 1
        if update_epoch == 1:
            candidates = tuple(item for item in ordered if item.prompt_id not in selected_ids)
            if not candidates:
                status = "incomplete_no_mixed_group"
                break
            rollout_version = before_version
            rollout_artifact_dir = step_dir
            scan = scan_fn(
                candidates,
                lambda problem: sample_fn(model, tokenizer, problem, config, rollout_version),
                config["algorithm"], min(run["max_prompts_per_step"], len(candidates)),
            )
            if not scan.attempts:
                raise RuntimeError("scan returned no attempts")
            reasons: Counter[str] = Counter()
            for attempt in scan.attempts:
                reasons.update(attempt["reason_counts"])
            selected = scan.selected_advantaged
            scan_record = {
                "policy_version": rollout_version,
                "source_sha256": split.source_sha256,
                "config_sha256": config_sha256,
                "order": run["order"],
                "max_prompts": min(run["max_prompts_per_step"], len(candidates)),
                "scanned_prompts": len(scan.attempts),
                "zero_std_group_fraction": sum(item["zero_variance"] for item in scan.attempts) / len(scan.attempts),
                "reason_counts": dict(reasons),
                "selected_prompt_id": selected[0].prompt_id if selected else None,
                "attempts": scan.attempts,
            }
            _write_json(step_dir / "scan.json", scan_record)
            if selected is None:
                status = "incomplete_no_mixed_group"
                break
            if scan.selected_rollout is None or scan.selected_scored is None:
                raise RuntimeError("selected scan has incomplete trajectory stages")
            if selected[0].prompt_id in selected_ids:
                raise ValueError("training prompt was selected twice")
            for name, group in (
                ("rollout", scan.selected_rollout),
                ("scored", scan.selected_scored),
                ("advantaged", selected),
            ):
                write_jsonl(step_dir / f"{name}.jsonl", group)
            selected_ids.add(selected[0].prompt_id)
        if selected is None or rollout_version is None or rollout_artifact_dir is None:
            raise RuntimeError("update epoch has no selected rollout")
        validate_group(selected, require_complete=True)
        if selected[0].policy_version != rollout_version:
            raise ValueError("selected group was not sampled from the recorded rollout policy")
        if len({item.reward for item in selected}) < 2:
            raise ValueError("selected group must have mixed rewards")

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        metrics = update_fn(
            model, optimizer, selected, pad_token_id=pad_token_id,
            device=config["model"]["device"],
            clip_epsilon=config["algorithm"]["clip_epsilon"],
            loss_reduction=config["algorithm"]["loss_reduction"],
            max_grad_norm=config["optimizer"]["max_grad_norm"],
            require_initial_ratio=update_epoch == 1,
        )
        if not math.isfinite(metrics.post_approx_kl_old):
            raise ValueError("post-update approx_kl_old must be finite")
        checkpoint = output_dir / "checkpoints" / after_version
        checkpoint.mkdir(parents=True)
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="1GB")
        tokenizer.save_pretrained(checkpoint)
        record = {
            "policy_before": before_version,
            "policy_after": after_version,
            "rollout_policy_version": rollout_version,
            "rollout_artifact_dir": str(rollout_artifact_dir),
            "update_epoch": update_epoch,
            "data_age_updates": update_epoch - 1,
            "prompt_id": selected[0].prompt_id,
            "reward_vector": [item.reward for item in selected],
            "advantage_vector": [item.advantage for item in selected],
            "metrics": asdict(metrics),
            "checkpoint_dir": str(checkpoint),
            "optimizer_state_saved": False,
        }
        _write_json(step_dir / "step.json", record)
        steps.append(record)
        print(f"{after_version}: prompt={record['prompt_id']} post_kl={metrics.post_approx_kl_old:.6g}", flush=True)
        if metrics.post_approx_kl_old > run["max_post_approx_kl_old"] and index + 1 < run["update_steps"]:
            status = "incomplete_post_step_drift"
            break

    final_rows, final = evaluation_fn(model, tokenizer, split.validation, config)
    _write_rows(output_dir / "validation-final.jsonl", final_rows)
    _write_json(output_dir / "validation-final.json", final)
    summary = {
        "status": status,
        "requested_steps": run["update_steps"],
        "completed_steps": len(steps),
        "update_epochs": update_epochs,
        "config_sha256": config_sha256,
        "dataset_revision": config["data"]["revision"],
        "source_sha256": split.source_sha256,
        "training_count": len(split.training),
        "validation_count": len(split.validation),
        "initial_model_revision": config["model"]["revision"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "generation_use_model_defaults": False,
        "baseline": baseline,
        "final": final,
        "steps": steps,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main(config_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    model_config = config["model"]
    if model_config["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    if model_config["dtype"] != "float32" or model_config["training_method"] != "full_finetune":
        raise ValueError("M0 runner requires full-parameter float32 training")
    if config["algorithm"]["beta"] != 0.0:
        raise ValueError("M0 runner requires beta=0")
    optimizer_config = config["optimizer"]
    if optimizer_config["name"] != "AdamW":
        raise ValueError("M0 runner requires AdamW")
    split = load_pinned_train(config["data"])
    _validate_run(config, split)
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["tokenizer_revision"], trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"], revision=model_config["revision"],
        dtype=torch.float32, trust_remote_code=False,
    ).to(model_config["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]), eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )
    return run_cycles(
        model, tokenizer, optimizer, split, config, output_dir,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0-run.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m0/end-to-end-v2"))
    args = parser.parse_args()
    result = main(args.config, args.output_dir)
    print(f"M0 run {result['status']}: {result['completed_steps']}/{result['requested_steps']} updates -> {args.output_dir}")
    if result["status"] != "completed":
        raise SystemExit(2)
