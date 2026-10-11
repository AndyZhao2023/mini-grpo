"""Compare reference-free answer annotations with pinned GSM8K answers.

This is a read-only diagnostic of existing M0 generations. It does not change
the strict verifier, prompt, checkpoint, or training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

import yaml

from src.data.dataset import MathProblem, load_pinned_train
from src.reward.math_verifier import parse_final_answer


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)")
_KINDS = {"terminal", "nonterminal", "unresolved"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def compare_answers(
    rows: Sequence[dict], annotations: Sequence[dict], problems: dict[str, MathProblem],
) -> list[dict]:
    """Require one auditable annotation per completed, missing-final-line response."""
    targets = [
        row for row in rows
        if row["finish_reason"] == "eos" and row["reward_reason"] == "missing_final_answer"
    ]
    target_ids = [row["prompt_id"] for row in targets]
    annotation_ids = [item["prompt_id"] for item in annotations]
    if len(set(target_ids)) != len(target_ids) or len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("duplicate prompt ID in responses or annotations")
    if set(annotation_ids) != set(target_ids):
        raise ValueError("annotations must cover exactly the completed missing-final-line responses")
    if any(row["prompt_id"] not in problems for row in rows):
        raise ValueError("response row is absent from the pinned training partition")

    by_id = {item["prompt_id"]: item for item in annotations}
    compared = []
    for row in targets:
        item = by_id[row["prompt_id"]]
        if set(item) != {"prompt_id", "kind", "candidate", "evidence"}:
            raise ValueError("annotation fields must be prompt_id, kind, candidate, evidence")
        kind, candidate, evidence = item["kind"], item["candidate"], item["evidence"]
        if kind not in _KINDS or not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"invalid annotation kind or evidence: {row['prompt_id']}")
        if evidence not in row["response_text"]:
            raise ValueError(f"annotation evidence is absent from response: {row['prompt_id']}")
        if kind == "unresolved":
            if candidate is not None:
                raise ValueError("unresolved annotation must have a null candidate")
            parsed_candidate = None
        else:
            if not isinstance(candidate, str) or not re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
                raise ValueError(f"candidate must be a canonical decimal string: {row['prompt_id']}")
            try:
                parsed_candidate = Decimal(candidate)
                seen = {Decimal(token.replace(",", "")) for token in _NUMBER.findall(evidence)}
            except InvalidOperation as exc:
                raise ValueError(f"invalid candidate evidence: {row['prompt_id']}") from exc
            if parsed_candidate not in seen:
                raise ValueError(f"candidate number is absent from evidence: {row['prompt_id']}")
        reference = parse_final_answer(problems[row["prompt_id"]].reference_answer)
        if reference is None:
            raise ValueError(f"pinned reference has no final number: {row['prompt_id']}")
        compared.append({
            "prompt_id": row["prompt_id"],
            "kind": kind,
            "candidate": candidate,
            "evidence": evidence,
            "reference": str(reference),
            "comparison": (
                "unresolved" if parsed_candidate is None else
                "matches_reference" if parsed_candidate == reference else
                "different_from_reference"
            ),
        })
    return compared


def summarize(rows: Sequence[dict], compared: Sequence[dict]) -> dict:
    comparisons = Counter(item["comparison"] for item in compared)
    kinds = Counter(item["kind"] for item in compared)
    return {
        "source_question_count": len(rows),
        "strict_correct_count": sum(row["reward_reason"] == "correct_answer" for row in rows),
        "completed_missing_final_count": len(compared),
        "truncated_missing_final_count": sum(
            row["finish_reason"] == "length" and row["reward_reason"] == "missing_final_answer"
            for row in rows
        ),
        "extraction_kind_counts": dict(sorted(kinds.items())),
        "comparison_counts": dict(sorted(comparisons.items())),
        "terminal_match_count": sum(
            item["kind"] == "terminal" and item["comparison"] == "matches_reference"
            for item in compared
        ),
    }


def main(config_path: Path, manifest_path: Path, rows_path: Path,
         annotations_path: Path, output_dir: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = load_pinned_train(config["data"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(rows_path)
    annotations = read_jsonl(annotations_path)
    row_ids = [row["prompt_id"] for row in rows]
    if (manifest["source_split"] != "train"
            or manifest["source_sha256"] != split.source_sha256
            or manifest["dataset_revision"] != config["data"]["revision"]
            or row_ids != manifest["prompt_ids"] or len(set(row_ids)) != len(row_ids)
            or set(row_ids) & {item.prompt_id for item in split.validation}):
        raise ValueError("diagnostic rows do not match the pinned training manifest")
    compared = compare_answers(
        rows, annotations, {item.prompt_id: item for item in split.training},
    )
    summary = {
        "status": "completed", "model_updated": False,
        "source_split": "train", "source_sha256": split.source_sha256,
        "input_sha256": {
            "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "responses": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            "annotations": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        },
        **summarize(rows, compared),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "audit.jsonl").open("x", encoding="utf-8") as output:
        for item in compared:
            output.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    with (output_dir / "summary.json").open("x", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write("\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0-run.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("results/m0/failure-diagnostic-v1/manifest.json"))
    parser.add_argument("--rows", type=Path, default=Path("results/m0/failure-diagnostic-v1/evaluation-128.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("results/m0/answer-audit-v1/annotations.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m0/answer-audit-v1"))
    args = parser.parse_args()
    print(main(args.config, args.manifest, args.rows, args.annotations, args.output_dir))
