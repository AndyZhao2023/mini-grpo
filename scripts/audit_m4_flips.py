"""Validate manual answer-value annotations for M4's strict-correct flips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.analyze_m1_prompt_errors import analyze_pairs
from scripts.audit_m0_answers import read_jsonl
from scripts.evaluate_m4_paired import build_protocol


_FIELDS = {"prompt_id", "failed_arm", "failure_class", "candidate", "evidence"}
_CLASSES = {"correct_value_wrong_format", "wrong_value", "unresolved"}


def to_shared_annotations(annotations: list[dict]) -> list[dict]:
    """Use the existing tested flip validator with M4's policy names."""
    converted = []
    for item in annotations:
        if (set(item) != _FIELDS or item["failed_arm"] not in {"m0", "m3"}
                or item["failure_class"] not in _CLASSES):
            raise ValueError("invalid M4 flip annotation")
        converted.append({
            **item,
            "failed_arm": "baseline" if item["failed_arm"] == "m0" else "two_step_example",
            "failure_class": (
                "truncated_unresolved" if item["failure_class"] == "unresolved"
                else item["failure_class"]
            ),
        })
    return converted


def main(spec_path: Path, run_dir: Path, annotations_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    questions, expected_manifest, _, _ = build_protocol(spec_path)
    manifest_path, run_summary_path = run_dir / "manifest.json", run_dir / "summary.json"
    m0_path, m3_path = run_dir / "m0.jsonl", run_dir / "m3.jsonl"
    paired_path = run_dir / "paired.jsonl"
    manifest, run_summary = json.loads(manifest_path.read_text()), json.loads(run_summary_path.read_text())
    if (manifest != expected_manifest or run_summary["status"] != "completed"
            or any(run_summary["input_sha256"][name] != hashlib.sha256(path.read_bytes()).hexdigest()
                   for name, path in (
                       ("manifest", manifest_path), ("m0_rows", m0_path),
                       ("m3_rows", m3_path), ("paired", paired_path),
                   ))):
        raise ValueError("saved M4 run or input hashes disagree with frozen protocol")
    m0_rows, m3_rows = read_jsonl(m0_path), read_jsonl(m3_path)
    original = read_jsonl(annotations_path)
    paired, counts = analyze_pairs(
        m0_rows, m3_rows, to_shared_annotations(original),
        {item.prompt_id: item for item in questions},
    )
    if (counts["strict_change_counts"] != run_summary["strict_change_counts"]
            or counts["baseline"] != run_summary["m0"]
            or counts["two_step_example"] != run_summary["m3"]
            or counts["paired"] != run_summary["paired"]):
        raise ValueError("manual audit disagrees with M4 paired summary")
    flip_by_id = {item["prompt_id"]: item for item in original}
    flips = [{
        "prompt_id": row["prompt_id"], "reference": row["reference"],
        "change": row["change"], "m0": row["baseline"],
        "m3": row["two_step_example"],
        "failed_response_annotation": flip_by_id[row["prompt_id"]],
    } for row in paired if row["change"] in {"newly_correct", "lost_correct"}]
    summary = {
        "status": "completed", "model_updated": False,
        "question_count": len(questions), "strict_change_counts": counts["strict_change_counts"],
        "flip_failure_class_counts": {
            name: {("unresolved" if key == "truncated_unresolved" else key): value
                   for key, value in counts["flip_failure_class_counts"][name].items()}
            for name in ("newly_correct", "lost_correct")
        },
        "scope": "all 800 saved answers rechecked; only 24 strict flips manually annotated",
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("manifest", manifest_path), ("run_summary", run_summary_path),
                ("m0_rows", m0_path), ("m3_rows", m3_path),
                ("annotations", annotations_path),
            )
        },
    }
    output_dir.mkdir(parents=True)
    with (output_dir / "flips.jsonl").open("x", encoding="utf-8") as file:
        for item in flips:
            file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/m4-paired-confirm-v1.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/m4/paired-confirm-v1"))
    parser.add_argument("--annotations", type=Path, default=Path("results/m4/paired-confirm-v1/flip-audit/annotations.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m4/paired-confirm-v1/flip-audit/output"))
    args = parser.parse_args()
    print(main(args.spec, args.run_dir, args.annotations, args.output_dir))
