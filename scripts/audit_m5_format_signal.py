"""Check candidate annotations and advantage direction for frozen M5 groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

import yaml

from src.data.dataset import load_pinned_train
from src.reward.math_verifier import parse_final_answer, verify_answer


NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)")
KINDS = {"terminal", "nonterminal", "unresolved"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(item: dict) -> tuple[str, str, int]:
    return item["arm"], item["prompt_id"], item["sample_index"]


def validate_annotations(
    targets: list[dict], annotations: list[dict], responses: dict[tuple, dict],
) -> dict[tuple, dict]:
    """Require exactly one response-grounded label for every frozen review target."""
    expected = [key(item) for item in targets]
    observed = [key(item) for item in annotations]
    if len(set(expected)) != len(expected) or observed != expected:
        raise ValueError("annotations must match the ordered, unique frozen targets")
    for item in annotations:
        if set(item) != {
            "arm", "prompt_id", "sample_index", "kind", "candidate", "evidence",
        }:
            raise ValueError("unexpected annotation fields")
        candidate, evidence, kind = item["candidate"], item["evidence"], item["kind"]
        response = responses[key(item)]
        if (response["finish_reason"] != "eos"
                or parse_final_answer(response["response_text"]) is not None):
            raise ValueError("annotation target must be an invalid-format completed response")
        if kind not in KINDS or not isinstance(evidence, str) or not evidence:
            raise ValueError("invalid annotation kind or evidence")
        if evidence not in response["response_text"]:
            raise ValueError("annotation evidence does not occur in response")
        if kind == "unresolved":
            if candidate is not None:
                raise ValueError("unresolved candidate must be null")
        else:
            if not isinstance(candidate, str) or not re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
                raise ValueError("candidate must be a canonical decimal string")
            values = {Decimal(token.replace(",", "")) for token in NUMBER.findall(evidence)}
            if Decimal(candidate) not in values:
                raise ValueError("candidate number does not occur in evidence")
    return {key(item): item for item in annotations}


def summarize(rows: list[dict], group_count: int) -> dict:
    counts = Counter()
    by_arm = {"shortest": Counter(), "hash_random": Counter()}
    groups: dict[tuple[str, str], list[dict]] = {}
    for item in rows:
        counts[f"comparison_{item['comparison']}"] += 1
        counts[f"kind_{item['candidate_kind']}"] += 1
        by_arm[item["arm"]][f"comparison_{item['comparison']}"] += 1
        if item["comparison"] == "different_from_reference" and item["naive_advantage"] > 0:
            counts["positive_advantage_known_wrong"] += 1
            by_arm[item["arm"]]["positive_advantage_known_wrong"] += 1
        if item["comparison"] == "matches_reference" and item["naive_advantage"] < 0:
            counts["negative_advantage_known_correct"] += 1
            by_arm[item["arm"]]["negative_advantage_known_correct"] += 1
        if item["comparison"] == "unresolved" and item["naive_advantage"] < 0:
            counts["negative_advantage_unknown"] += 1
        groups.setdefault((item["arm"], item["prompt_id"]), []).append(item)
    if len(groups) != group_count or any(len(group) != 4 for group in groups.values()):
        raise ValueError("audit must contain every four-response group")
    counts["groups_with_known_correct"] = sum(any(
        item["comparison"] == "matches_reference" for item in group
    ) for group in groups.values())
    counts["groups_all_known_wrong"] = sum(all(
        item["comparison"] == "different_from_reference" for item in group
    ) for group in groups.values())
    counts["groups_with_unknown"] = sum(any(
        item["comparison"] == "unresolved" for item in group
    ) for group in groups.values())
    counts["groups_with_rank_inversion"] = sum(
        any(item["comparison"] == "different_from_reference" and item["naive_advantage"] > 0
            for item in group)
        and any(item["comparison"] == "matches_reference" and item["naive_advantage"] < 0
                for item in group)
        for group in groups.values()
    )
    return {
        "group_count": len(groups), "response_count": len(rows),
        "counts": dict(sorted(counts.items())),
        "by_arm": {arm: dict(sorted(values.items())) for arm, values in by_arm.items()},
    }


def main(manifest_path: Path, annotations_path: Path, config_path: Path,
         output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = json.loads(manifest_path.read_text())
    if manifest["experiment"] != "m5-format-signal-audit-v1":
        raise ValueError("wrong audit manifest")
    for name, expected_hash in manifest["source_file_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"source file changed: {name}")
    config = yaml.safe_load(config_path.read_text())
    split = load_pinned_train(config["data"])
    problems = {problem.prompt_id: problem for problem in split.training}
    m5_path = Path("results/m5/offline-reward-simulation-v1/groups.jsonl")
    m5_rows = read_jsonl(m5_path)
    selected = [row for row in m5_rows if (
        not row["scenarios"]["strict"]["trainable"]
        and row["scenarios"]["strict_format_0.1"]["trainable"]
    )]
    selected_ids = [{"arm": row["arm"], "prompt_id": row["prompt_id"]} for row in selected]
    if selected_ids != manifest["groups"]:
        raise ValueError("frozen group selection changed")
    m2: dict[tuple[str, str], dict] = {}
    for arm in ("shortest", "hash_random"):
        for row in read_jsonl(Path(f"results/m2/sampling-order-v1/{arm}.jsonl")):
            m2[(arm, row["prompt_id"])] = row
    responses = {
        (arm, prompt_id, sample["sample_index"]): sample
        for (arm, prompt_id), row in m2.items() for sample in row["samples"]
    }
    annotations = read_jsonl(annotations_path)
    labels = validate_annotations(manifest["annotation_targets"], annotations, responses)
    output_rows = []
    for row in selected:
        arm, prompt_id = row["arm"], row["prompt_id"]
        reference = problems[prompt_id].reference_answer
        expected = parse_final_answer(reference)
        for label in row["labels"]:
            sample_index = label["sample_index"]
            response = responses[(arm, prompt_id, sample_index)]
            strict = verify_answer(response["response_text"], reference)
            if (strict.reward != response["reward"]
                    or strict.reason != response["reward_reason"]
                    or label["format_valid"] != (
                        parse_final_answer(response["response_text"]) is not None
                    )):
                raise ValueError(f"strict score or format differs: {prompt_id}")
            annotation = labels.get((arm, prompt_id, sample_index))
            if label["format_valid"]:
                candidate = parse_final_answer(response["response_text"])
                kind, evidence = "valid_final", response["response_text"].strip().splitlines()[-1]
            elif response["finish_reason"] == "length":
                candidate, kind, evidence = None, "length_abstain", None
            elif annotation is not None:
                candidate = (
                    None if annotation["candidate"] is None else Decimal(annotation["candidate"])
                )
                kind, evidence = annotation["kind"], annotation["evidence"]
            else:
                raise ValueError(f"missing annotation: {prompt_id}/{sample_index}")
            comparison = (
                "unresolved" if candidate is None else
                "matches_reference" if candidate == expected else "different_from_reference"
            )
            output_rows.append({
                "arm": arm, "prompt_id": prompt_id, "sample_index": sample_index,
                "finish_reason": response["finish_reason"],
                "format_valid": label["format_valid"],
                "candidate": None if candidate is None else str(candidate),
                "candidate_kind": kind, "candidate_evidence": evidence,
                "comparison": comparison,
                "naive_reward": row["scenarios"]["strict_format_0.1"]["rewards"][sample_index],
                "naive_advantage": (
                    row["scenarios"]["strict_format_0.1"]["advantages"][sample_index]
                ),
            })
    summary = {
        "experiment": manifest["experiment"], "model_updated": False,
        "source_split": "train", "source_sha256": split.source_sha256,
        "input_sha256": {
            "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "annotations": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        },
        **summarize(output_rows, len(selected)),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "rows.jsonl").write_text("".join(
        json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in output_rows
    ))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("results/m5/format-signal-audit-v1/manifest.json"),
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=Path("results/m5/format-signal-audit-v1/annotations.jsonl"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/m0-run.yaml"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/m5/format-signal-audit-v1/output"),
    )
    args = parser.parse_args()
    print(json.dumps(main(args.manifest, args.annotations, args.config, args.output_dir), indent=2))
