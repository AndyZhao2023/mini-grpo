"""Diagnose strict-answer failures without updating the M0 model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import MathProblem, load_pinned_train
from src.evaluation import evaluate_model


def select_problems(
    training: Sequence[MathProblem], excluded_ids: set[str], count: int,
    *, salt: str = "m0-failure-diagnostic-v1",
) -> tuple[MathProblem, ...]:
    """Choose a stable sample without ranking questions by difficulty."""
    candidates = [item for item in training if item.prompt_id not in excluded_ids]
    if type(count) is not int or count < 1 or count > len(candidates):
        raise ValueError("count must be positive and fit the unscanned training partition")
    return tuple(sorted(
        candidates,
        key=lambda item: (hashlib.sha256(f"{salt}:{item.prompt_id}".encode()).digest(), item.prompt_id),
    )[:count])


def summarize(rows: list[dict]) -> dict:
    reasons = Counter(row["reward_reason"] for row in rows)
    endings = Counter(row["finish_reason"] for row in rows)
    cross = Counter(f"{row['finish_reason']}:{row['reward_reason']}" for row in rows)
    return {
        "question_count": len(rows),
        "correct_count": reasons["correct_answer"],
        "accuracy": reasons["correct_answer"] / len(rows) if rows else None,
        "reason_counts": dict(reasons),
        "finish_reason_counts": dict(endings),
        "finish_by_reward_reason": dict(cross),
    }


def compare_extended(short_rows: list[dict], long_rows: list[dict]) -> dict:
    truncated = {row["prompt_id"]: row for row in short_rows if row["finish_reason"] == "length"}
    if len(truncated) != len(long_rows) or {row["prompt_id"] for row in long_rows} != set(truncated):
        raise ValueError("extended rows must match exactly the short-cap truncated prompts")
    return {
        "rerun_count": len(long_rows),
        "new_correct_count": sum(
            truncated[row["prompt_id"]]["reward"] == 0 and row["reward"] == 1
            for row in long_rows
        ),
        "still_truncated_count": sum(row["finish_reason"] == "length" for row in long_rows),
        "decoded_short_text_is_long_prefix_count": sum(
            row["response_text"].startswith(truncated[row["prompt_id"]]["response_text"])
            for row in long_rows
        ),
        "extended": summarize(long_rows),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _evaluate_to_file(
    model, tokenizer, problems: Sequence[MathProblem], config: dict, path: Path,
    *, label: str,
) -> list[dict]:
    rows = []
    with path.open("x", encoding="utf-8") as output:
        for index, problem in enumerate(problems, 1):
            result, _ = evaluate_model(model, tokenizer, (problem,), config)
            rows.extend(result)
            output.write(json.dumps(result[0], ensure_ascii=False, allow_nan=False) + "\n")
            if index % 10 == 0 or index == len(problems):
                output.flush()
                print(f"{label}: {index}/{len(problems)}", flush=True)
    return rows


def main(
    config_path: Path, run_dir: Path, checkpoint_dir: Path, output_dir: Path,
    *, count: int = 100, extended_max_new_tokens: int = 256,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["evaluation"]["do_sample"] is not False:
        raise ValueError("diagnostic requires greedy evaluation")
    short_cap = config["evaluation"]["max_new_tokens"]
    if type(extended_max_new_tokens) is not int or extended_max_new_tokens <= short_cap:
        raise ValueError("extended cap must exceed the configured evaluation cap")
    if config["model"]["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)

    split = load_pinned_train(config["data"])
    run = json.loads((run_dir / "summary.json").read_text())
    if (run["status"] != "completed" or run["source_sha256"] != split.source_sha256
            or run["config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
            or not run["steps"] or checkpoint_dir.name != run["steps"][-1]["policy_after"]
            or checkpoint_dir.resolve() != Path(run["steps"][-1]["checkpoint_dir"]).resolve()):
        raise ValueError("checkpoint, run summary, config, and pinned data must agree")
    scanned_ids: set[str] = set()
    for step in run["steps"]:
        scan = json.loads((run_dir / "steps" / step["policy_after"] / "scan.json").read_text())
        if scan["selected_prompt_id"] != step["prompt_id"]:
            raise ValueError("step and scan selected prompt IDs disagree")
        scanned_ids.update(attempt["prompt_id"] for attempt in scan["attempts"])
    problems = select_problems(split.training, scanned_ids, count)
    if {item.prompt_id for item in problems} & {item.prompt_id for item in split.validation}:
        raise ValueError("diagnostic sample overlaps validation")

    output_dir.mkdir(parents=True)
    _write_json(output_dir / "manifest.json", {
        "source_split": "train", "source_sha256": split.source_sha256,
        "dataset_revision": config["data"]["revision"],
        "config_sha256": run["config_sha256"],
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_policy_version": run["steps"][-1]["policy_after"],
        "selection": "lowest SHA256(m0-failure-diagnostic-v1:prompt_id) among unscanned training prompts",
        "excluded_scanned_prompt_count": len(scanned_ids),
        "prompt_ids": [item.prompt_id for item in problems],
        "short_max_new_tokens": short_cap,
        "extended_max_new_tokens": extended_max_new_tokens,
    })
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, dtype=torch.float32, trust_remote_code=False,
    ).to(config["model"]["device"])
    model.eval()
    short_rows = _evaluate_to_file(
        model, tokenizer, problems, config,
        output_dir / f"evaluation-{short_cap}.jsonl", label="short",
    )
    truncated_ids = {row["prompt_id"] for row in short_rows if row["finish_reason"] == "length"}
    extended_config = {
        **config, "evaluation": {**config["evaluation"], "max_new_tokens": extended_max_new_tokens},
    }
    long_rows = _evaluate_to_file(
        model, tokenizer, [item for item in problems if item.prompt_id in truncated_ids],
        extended_config, output_dir / f"evaluation-{extended_max_new_tokens}-truncated.jsonl",
        label="extended",
    )
    result = {
        "status": "completed", "model_updated": False,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "short": summarize(short_rows),
        "extended_on_short_truncations": compare_extended(short_rows, long_rows),
    }
    _write_json(output_dir / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0-run.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/m0/end-to-end-v2"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/m0/end-to-end-v2/checkpoints/step-000002"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m0/failure-diagnostic-v1"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--extended-max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    print(main(
        args.config, args.run_dir, args.checkpoint_dir, args.output_dir,
        count=args.count, extended_max_new_tokens=args.extended_max_new_tokens,
    ))
