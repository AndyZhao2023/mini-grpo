"""Scan a bounded prefix of pinned training prompts for a mixed-reward group."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import MathProblem, load_pinned_train
from src.data.schema import Trajectory, read_jsonl, validate_group, write_jsonl
from src.grpo.advantage import normalize_group_advantages
from src.reward.math_verifier import verify_group
from src.rollout.hf_rollout import sample_group


@dataclass(frozen=True)
class ScanResult:
    attempts: list[dict]
    selected_rollout: list[Trajectory] | None
    selected_scored: list[Trajectory] | None
    selected_advantaged: list[Trajectory] | None


def scan_for_mixed_group(
    problems: Sequence[MathProblem],
    rollout_fn: Callable[[MathProblem], list[Trajectory]],
    algorithm_config: dict,
    max_prompts: int,
) -> ScanResult:
    if type(max_prompts) is not int or not 0 < max_prompts <= len(problems):
        raise ValueError("max_prompts must be positive and no greater than the training size")
    attempts = []
    for problem in problems[:max_prompts]:
        rollout = rollout_fn(problem)
        scored = verify_group(problem, rollout)
        rewards = [item.reward for item in scored]
        attempts.append({
            "prompt_id": problem.prompt_id,
            "rewards": rewards,
            "reason_counts": dict(Counter(item.reward_reason for item in scored)),
            "response_lengths": [sum(item.response_mask) for item in scored],
            "truncation_count": sum(item.finish_reason == "length" for item in scored),
            "zero_variance": len(set(rewards)) == 1,
        })
        if not attempts[-1]["zero_variance"]:
            advantaged = normalize_group_advantages(scored, algorithm_config).group
            return ScanResult(attempts, rollout, scored, advantaged)
    return ScanResult(attempts, None, None, None)


def order_training_problems(
    problems: Sequence[MathProblem], order: str,
) -> tuple[MathProblem, ...]:
    if order == "original":
        return tuple(problems)
    if order == "shortest":
        return tuple(sorted(problems, key=lambda problem: (
            len(problem.question.split()), problem.prompt_id,
        )))
    raise ValueError("order must be original or shortest")


def main(config_path: Path, output_dir: Path, max_prompts: int, order: str) -> None:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    model_config = config["model"]
    if model_config["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    if model_config["dtype"] != "float32":
        raise ValueError("M0 scan currently supports float32 only")
    split = load_pinned_train(config["data"])
    problems = order_training_problems(split.training, order)
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["tokenizer_revision"],
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"], revision=model_config["revision"],
        dtype=torch.float32, trust_remote_code=False,
    ).to(model_config["device"])
    model.eval()

    def rollout(problem: MathProblem) -> list[Trajectory]:
        group = sample_group(model, tokenizer, problem, config, "step-000000")
        print(f"Scanned {problem.prompt_id}", flush=True)
        return group

    result = scan_for_mixed_group(problems, rollout, config["algorithm"], max_prompts)
    reason_counts = Counter()
    for attempt in result.attempts:
        reason_counts.update(attempt["reason_counts"])
    record = {
        "dataset_revision": config["data"]["revision"],
        "source_sha256": split.source_sha256,
        "model_revision": model_config["revision"],
        "tokenizer_revision": model_config["tokenizer_revision"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "seed": config["seed"],
        "selection": f"first mixed-reward group in {order} training order",
        "order": order,
        "max_prompts": max_prompts,
        "scanned_prompts": len(result.attempts),
        "zero_std_group_fraction": sum(item["zero_variance"] for item in result.attempts) / len(result.attempts),
        "reason_counts": dict(reason_counts),
        "selected_prompt_id": result.attempts[-1]["prompt_id"] if result.selected_rollout else None,
        "attempts": result.attempts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train-smoke-scan.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    if result.selected_rollout is None:
        raise RuntimeError(f"no mixed-reward group in first {max_prompts} training prompts")
    for name, group in (
        ("train-smoke-rollout.jsonl", result.selected_rollout),
        ("train-smoke-scored.jsonl", result.selected_scored),
        ("train-smoke-advantaged.jsonl", result.selected_advantaged),
    ):
        path = output_dir / name
        write_jsonl(path, group)
        saved = read_jsonl(path)
        validate_group(saved, require_complete=name == "train-smoke-advantaged.jsonl")
        if saved != group:
            raise RuntimeError(f"{path} changed during JSONL round trip")
    print(
        f"Selected {record['selected_prompt_id']} after {record['scanned_prompts']} prompts; "
        f"zero_std_group_fraction={record['zero_std_group_fraction']:.3f} -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m0"))
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--order", choices=("original", "shortest"), default="original")
    args = parser.parse_args()
    main(args.config, args.output_dir, args.max_prompts, args.order)
