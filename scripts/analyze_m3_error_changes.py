"""Audit saved M3 validation pairs without generating or updating a model."""

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
from scripts.compare_prompt_quality import compare_paired
from src.data.dataset import MathProblem, load_pinned_train
from src.reward.math_verifier import parse_final_answer, verify_answer


_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)")
_GAIN_MECHANISMS = {
    "same_value_and_derivation", "changed_answer_value", "completed_answer",
}
_ANNOTATION_FIELDS = {
    "prompt_id", "initial_kind", "initial_candidate", "initial_evidence",
    "gain_mechanism", "final_evidence",
}


def target_class(initial: dict, final: dict) -> str | None:
    if initial["reward"] == 0 and final["reward"] == 1:
        return "newly_correct"
    if (initial["reward_reason"] == "missing_final_answer"
            and final["reward_reason"] == "incorrect_answer"):
        return "new_wrong_final"
    return None


def _validate_annotation(
    annotation: dict, initial: dict, final: dict, reference: Decimal, target: str,
) -> dict:
    prompt_id = initial["prompt_id"]
    if set(annotation) != _ANNOTATION_FIELDS or annotation["prompt_id"] != prompt_id:
        raise ValueError(f"invalid annotation fields or prompt ID: {prompt_id}")
    kind, candidate = annotation["initial_kind"], annotation["initial_candidate"]
    evidence = annotation["initial_evidence"]
    if (kind not in {"terminal", "unresolved"}
            or not isinstance(evidence, str) or not evidence.strip()
            or evidence not in initial["response_text"]):
        raise ValueError(f"initial evidence is absent or invalid: {prompt_id}")
    if kind == "unresolved":
        if candidate is not None:
            raise ValueError(f"unresolved answer must have no candidate: {prompt_id}")
        value = None
    else:
        if (initial["finish_reason"] != "eos"
                or not isinstance(candidate, str)
                or not re.fullmatch(r"-?\d+(?:\.\d+)?", candidate)):
            raise ValueError(f"terminal candidate requires a completed answer: {prompt_id}")
        value = Decimal(candidate)
        if value not in {Decimal(token.replace(",", "")) for token in _NUMBER.findall(evidence)}:
            raise ValueError(f"candidate number is absent from evidence: {prompt_id}")

    mechanism, final_evidence = annotation["gain_mechanism"], annotation["final_evidence"]
    if target == "newly_correct":
        if (mechanism not in _GAIN_MECHANISMS
                or not isinstance(final_evidence, str) or not final_evidence.strip()
                or final_evidence not in final["response_text"]):
            raise ValueError(f"gain needs supported final evidence: {prompt_id}")
        if (mechanism == "same_value_and_derivation" and value != reference
                or mechanism == "changed_answer_value" and (value is None or value == reference)
                or mechanism == "completed_answer" and (value is not None or initial["finish_reason"] != "eos")):
            raise ValueError(f"gain mechanism disagrees with initial answer value: {prompt_id}")
        relation = mechanism
    else:
        if mechanism is not None or final_evidence is not None:
            raise ValueError(f"wrong-final case cannot have a gain mechanism: {prompt_id}")
        final_value = parse_final_answer(final["response_text"])
        if final_value is None or final_value == reference:
            raise ValueError(f"new wrong final answer is not valid and wrong: {prompt_id}")
        relation = (
            "unresolved_initial" if value is None else
            "same_wrong_value" if value == final_value else "different_wrong_value"
        )
    return {"initial_value_matches_reference": value == reference if value is not None else None,
            "value_relation": relation}


def analyze_pairs(
    initial: Sequence[dict], final: Sequence[dict], annotations: Sequence[dict],
    problems: dict[str, MathProblem],
) -> tuple[list[dict], dict]:
    """Recheck every score, then classify exactly the preselected 44 changes."""
    ids = [row["prompt_id"] for row in initial]
    if (not ids or ids != [row["prompt_id"] for row in final]
            or len(set(ids)) != len(ids) or set(ids) - problems.keys()):
        raise ValueError("validation pairs must have unique, aligned pinned train IDs")
    target_ids = {
        a["prompt_id"] for a, b in zip(initial, final) if target_class(a, b)
    }
    annotation_ids = [item["prompt_id"] for item in annotations]
    if len(annotation_ids) != len(set(annotation_ids)) or set(annotation_ids) != target_ids:
        raise ValueError("annotations must cover exactly the selected validation changes")
    by_id = {item["prompt_id"]: item for item in annotations}
    paired = []
    transitions = Counter()
    targets = Counter()
    gain_mechanisms = Counter()
    new_wrong_relations = Counter()
    for a, b in zip(initial, final):
        prompt_id = a["prompt_id"]
        reference = parse_final_answer(problems[prompt_id].reference_answer)
        if reference is None:
            raise ValueError(f"pinned reference has no final answer: {prompt_id}")
        for row in (a, b):
            checked = verify_answer(row["response_text"], problems[prompt_id].reference_answer)
            if (row["reward"], row["reward_reason"]) != (checked.reward, checked.reason):
                raise ValueError(f"saved verifier result differs from pinned answer: {prompt_id}")
            if row["finish_reason"] not in {"eos", "length"}:
                raise ValueError(f"invalid finish reason: {prompt_id}")
        target = target_class(a, b)
        annotation = by_id.get(prompt_id)
        relation = None
        if target:
            targets[target] += 1
            relation = _validate_annotation(annotation, a, b, reference, target)
            if target == "newly_correct":
                gain_mechanisms[relation["value_relation"]] += 1
            else:
                new_wrong_relations[relation["value_relation"]] += 1
        if a["reward_reason"] != b["reward_reason"]:
            transitions[f"{a['reward_reason']} -> {b['reward_reason']}"] += 1
        paired.append({
            "prompt_id": prompt_id,
            "reference": str(reference),
            "target_class": target,
            "initial": {key: a[key] for key in ("reward", "reward_reason", "finish_reason")},
            "final": {key: b[key] for key in ("reward", "reward_reason", "finish_reason")},
            "annotation": annotation,
            "value_relation": relation,
        })
    return paired, {
        "question_count": len(paired),
        "target_counts": dict(sorted(targets.items())),
        "gain_mechanism_counts": dict(sorted(gain_mechanisms.items())),
        "new_wrong_final_initial_relation_counts": dict(sorted(new_wrong_relations.items())),
        "reward_reason_transition_counts": dict(sorted(transitions.items())),
        "paired": compare_paired(initial, final),
    }


def _check_metrics(rows: Sequence[dict], metrics: dict) -> None:
    reasons = Counter(row["reward_reason"] for row in rows)
    if (metrics["question_count"] != len(rows)
            or metrics["correct_count"] != reasons["correct_answer"]
            or metrics["accuracy"] != reasons["correct_answer"] / len(rows)
            or metrics["reason_counts"] != dict(reasons)
            or metrics["truncation_count"] != sum(row["finish_reason"] == "length" for row in rows)
            or metrics["mean_response_tokens"] != sum(row["response_tokens"] for row in rows) / len(rows)):
        raise ValueError("saved validation metrics disagree with their rows")


def main(
    config_path: Path, run_dir: Path, annotations_path: Path, output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest_path, summary_path = run_dir / "dataset-split.json", run_dir / "summary.json"
    initial_path, final_path = run_dir / "validation-initial.jsonl", run_dir / "validation-final.jsonl"
    manifest, run = json.loads(manifest_path.read_text()), json.loads(summary_path.read_text())
    initial, final, annotations = (
        read_jsonl(initial_path), read_jsonl(final_path), read_jsonl(annotations_path)
    )
    source = load_pinned_train(config["data"])
    ids = [row["prompt_id"] for row in initial]
    if (run["status"] != "completed" or run["completed_steps"] != 10
            or run["config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
            or manifest["config_sha256"] != run["config_sha256"]
            or manifest["source_sha256"] != source.source_sha256
            or run["source_sha256"] != source.source_sha256
            or manifest["source_split"] != "train"
            or ids != manifest["validation_prompt_ids"]
            or set(ids) & set(manifest["training_prompt_ids"])
            or set(ids) & {item.prompt_id for item in source.validation}):
        raise ValueError("M3 validation rows do not match completed run and fixed split")
    _check_metrics(initial, run["baseline"])
    _check_metrics(final, run["final"])
    paired, counts = analyze_pairs(
        initial, final, annotations, {item.prompt_id: item for item in source.training},
    )
    summary = {
        "status": "completed", "model_updated": False, "new_generation": False,
        "source_split": "train", "source_sha256": source.source_sha256,
        "scope": "all 200 pairs rescored; 44 selected transitions annotated from saved text",
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("config", config_path), ("manifest", manifest_path),
                ("run_summary", summary_path), ("initial", initial_path),
                ("final", final_path), ("annotations", annotations_path),
            )
        },
        "baseline_metrics": run["baseline"], "final_metrics": run["final"],
        **counts,
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "paired.jsonl").open("x", encoding="utf-8") as output:
        for item in paired:
            output.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m3-warm-start-v1.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/m3/warm-start-v1"))
    parser.add_argument("--annotations", type=Path, default=Path("results/m3/error-audit-v1/annotations.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m3/error-audit-v1/output"))
    args = parser.parse_args()
    print(main(args.config, args.run_dir, args.annotations, args.output_dir))
