"""Audit the strict-correct flips in the reserved M1 prompt comparison.

All 200 pairs receive their recorded verifier and finish classifications. Only
the 22 pairs whose strict correctness changed receive a manual answer-value
annotation; the other responses are not silently treated as math failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Sequence

import yaml

from scripts.audit_m0_answers import read_jsonl
from scripts.compare_prompt_quality import compare_paired, summarize
from src.data.dataset import MathProblem, load_pinned_train
from src.reward.math_verifier import parse_final_answer, verify_answer


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)")
_FAILURE_CLASSES = {"correct_value_wrong_format", "wrong_value", "truncated_unresolved"}


def analyze_pairs(
    baseline: Sequence[dict], variant: Sequence[dict], annotations: Sequence[dict],
    problems: dict[str, MathProblem],
) -> tuple[list[dict], dict]:
    """Validate original scores and classify every strict-correct flip."""
    if not baseline or [row["prompt_id"] for row in baseline] != [row["prompt_id"] for row in variant]:
        raise ValueError("both arms must contain the same ordered, nonempty questions")
    ids = [row["prompt_id"] for row in baseline]
    if len(set(ids)) != len(ids) or set(ids) - problems.keys():
        raise ValueError("paired IDs must be unique and present in the pinned training data")
    by_id = {item["prompt_id"]: item for item in annotations}
    if len(by_id) != len(annotations):
        raise ValueError("duplicate annotation prompt ID")
    flip_ids = {
        a["prompt_id"] for a, b in zip(baseline, variant)
        if a["reward"] != b["reward"]
    }
    if set(by_id) != flip_ids:
        raise ValueError("annotations must cover exactly the strict-correct flips")

    paired = []
    changes = Counter()
    failure_classes = {"newly_correct": Counter(), "lost_correct": Counter()}
    for a, b in zip(baseline, variant):
        prompt_id = a["prompt_id"]
        problem = problems[prompt_id]
        reference = parse_final_answer(problem.reference_answer)
        if reference is None:
            raise ValueError(f"pinned reference has no final answer: {prompt_id}")
        for row in (a, b):
            checked = verify_answer(row["response_text"], problem.reference_answer)
            if (row["reward"], row["reward_reason"]) != (checked.reward, checked.reason):
                raise ValueError(f"saved verifier result differs from pinned reference: {prompt_id}")
            if row["finish_reason"] not in {"eos", "length"}:
                raise ValueError(f"unknown finish reason: {prompt_id}")
        change = (
            "newly_correct" if a["reward"] == 0 and b["reward"] == 1 else
            "lost_correct" if a["reward"] == 1 and b["reward"] == 0 else
            "correct_in_both" if a["reward"] == 1 else "incorrect_in_both"
        )
        changes[change] += 1
        detail = None
        if prompt_id in flip_ids:
            detail = by_id[prompt_id]
            if set(detail) != {"prompt_id", "failed_arm", "failure_class", "candidate", "evidence"}:
                raise ValueError(f"invalid annotation fields: {prompt_id}")
            failed_arm = "baseline" if change == "newly_correct" else "two_step_example"
            failed = a if failed_arm == "baseline" else b
            if detail["failed_arm"] != failed_arm or detail["failure_class"] not in _FAILURE_CLASSES:
                raise ValueError(f"wrong failed arm or failure class: {prompt_id}")
            evidence = detail["evidence"]
            if not isinstance(evidence, str) or not evidence or evidence not in failed["response_text"]:
                raise ValueError(f"annotation evidence absent from failed response: {prompt_id}")
            if detail["failure_class"] == "truncated_unresolved":
                if (failed["finish_reason"] != "length" or detail["candidate"] is not None
                        or failed["reward_reason"] not in {"missing_final_answer", "invalid_final_answer"}):
                    raise ValueError(f"truncated answer must have no candidate: {prompt_id}")
            else:
                candidate = detail["candidate"]
                if (failed["finish_reason"] != "eos" or not isinstance(candidate, str)
                        or not re.fullmatch(r"-?\d+(?:\.\d+)?", candidate)
                        or Decimal(candidate) not in {
                            Decimal(token.replace(",", "")) for token in _NUMBER.findall(evidence)
                        }):
                    raise ValueError(f"candidate is not supported by a completed response: {prompt_id}")
                if (Decimal(candidate) == reference) != (detail["failure_class"] == "correct_value_wrong_format"):
                    raise ValueError(f"candidate and failure class disagree with reference: {prompt_id}")
                if detail["failure_class"] == "correct_value_wrong_format" and failed["reward_reason"] == "incorrect_answer":
                    raise ValueError(f"valid final answer cannot be a format-only failure: {prompt_id}")
            failure_classes[change][detail["failure_class"]] += 1
        paired.append({
            "prompt_id": prompt_id,
            "reference": str(reference),
            "change": change,
            "baseline": {"reward_reason": a["reward_reason"], "finish_reason": a["finish_reason"]},
            "two_step_example": {"reward_reason": b["reward_reason"], "finish_reason": b["finish_reason"]},
            "failed_response_annotation": detail,
        })
    return paired, {
        "question_count": len(paired),
        "strict_change_counts": dict(sorted(changes.items())),
        "flip_failure_class_counts": {
            arm: dict(sorted(counts.items())) for arm, counts in failure_classes.items()
        },
        "baseline": summarize(baseline),
        "two_step_example": summarize(variant),
        "paired": compare_paired(baseline, variant),
    }


def main(
    config_path: Path, manifest_path: Path, experiment_summary_path: Path,
    baseline_path: Path, variant_path: Path, annotations_path: Path, output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = load_pinned_train(config["data"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_summary = json.loads(experiment_summary_path.read_text(encoding="utf-8"))
    baseline, variant = read_jsonl(baseline_path), read_jsonl(variant_path)
    ids = [row["prompt_id"] for row in baseline]
    if (prior_summary["status"] != "completed" or manifest["source_split"] != "train"
            or manifest["source_sha256"] != split.source_sha256
            or ids != manifest["reserved_prompt_ids"]
            or set(ids) & {item.prompt_id for item in split.validation}):
        raise ValueError("confirmation rows do not match the completed pinned training manifest")
    paired, counts = analyze_pairs(
        baseline, variant, read_jsonl(annotations_path),
        {item.prompt_id: item for item in split.training},
    )
    if any(counts[key] != prior_summary[key] for key in ("baseline", "two_step_example", "paired")):
        raise ValueError("recomputed prompt comparison differs from the original summary")
    summary = {
        "status": "completed", "model_updated": False,
        "source_split": "train", "source_sha256": split.source_sha256,
        "scope": "all pairs: strict verifier and finish reason; flips: manual answer-value annotation",
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("manifest", manifest_path), ("experiment_summary", experiment_summary_path),
                ("baseline", baseline_path), ("two_step_example", variant_path),
                ("annotations", annotations_path),
            )
        },
        **counts,
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "paired.jsonl").open("x", encoding="utf-8") as output:
        for item in paired:
            output.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("results/m1/prompt-two-step-confirm-v1")
    parser.add_argument("--config", type=Path, default=Path("configs/m0-run.yaml"))
    parser.add_argument("--manifest", type=Path, default=root / "manifest.json")
    parser.add_argument("--experiment-summary", type=Path, default=root / "summary.json")
    parser.add_argument("--baseline", type=Path, default=root / "baseline.jsonl")
    parser.add_argument("--variant", type=Path, default=root / "two-step-example.jsonl")
    parser.add_argument("--annotations", type=Path, default=Path("results/m1/prompt-error-audit-v1/annotations.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m1/prompt-error-audit-v1/output"))
    args = parser.parse_args()
    print(main(
        args.config, args.manifest, args.experiment_summary, args.baseline,
        args.variant, args.annotations, args.output_dir,
    ))
