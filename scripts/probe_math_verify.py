"""Compare pinned Math-Verify extraction modes with the frozen M5 answer audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from sympy import Basic, Rational, simplify

from src.reward.math_verifier import parse_final_answer


MODES = ("default_numeric_eos", "anchor_numeric_eos")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def numeric_candidate(parsed: list) -> Basic | None:
    """Reject strings, symbolic expressions, non-real values, and conflicting numbers."""
    values = [
        item for item in parsed
        if isinstance(item, Basic)
        and item.is_number is True and item.is_real is True and item.is_finite is True
    ]
    if not values:
        return None
    first = values[0]
    if any(simplify(item - first) != 0 for item in values[1:]):
        return None
    return first


def compare_to_manual(candidate: Basic | None, manual: dict, verify_fn) -> str:
    expected = manual["candidate"]
    if expected is None:
        return "unknown_extracted" if candidate is not None else "unknown_abstained"
    if candidate is None:
        return "missed"
    return "agreed" if verify_fn([Rational(expected)], [candidate]) else "disagreed"


def summarize(rows: list[dict]) -> dict:
    counts = {mode: Counter() for mode in MODES}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["prompt_id"])].append(row)
        for mode in MODES:
            outcome = row[mode]["comparison"]
            counts[mode][outcome] += 1
            if row["candidate_kind"] == "length_abstain":
                counts[mode]["length_rows"] += 1
            if row["candidate_kind"] == "unresolved":
                counts[mode]["human_unresolved_eos"] += 1
                if outcome == "unknown_extracted":
                    counts[mode]["extracted_on_human_unresolved_eos"] += 1
            if row[mode]["source"] == "math_verify":
                counts[mode]["math_verify_accepted"] += 1
            if row[mode]["source"] == "strict_final":
                counts[mode]["strict_final_accepted"] += 1
    if len(groups) != 38 or any(len(group) != 4 for group in groups.values()):
        raise ValueError("the probe must cover all 38 complete M5 groups")
    output = {}
    for mode in MODES:
        counts[mode]["complete_candidate_groups"] = sum(all(
            row[mode]["candidate"] is not None for row in group
        ) for group in groups.values())
        counts[mode]["fully_agreed_mixed_groups"] = sum(
            all(row[mode]["comparison"] == "agreed" for row in group)
            and {row["comparison"] for row in group} == {
                "matches_reference", "different_from_reference",
            }
            for group in groups.values()
        )
        output[mode] = dict(sorted(counts[mode].items()))
    raw = Counter()
    for row in rows:
        if row["raw_default"]["parsed"]:
            raw["nonempty_parse"] += 1
        if row["raw_default"]["numeric_candidate"] is not None:
            raw["numeric_candidate"] += 1
        if (row["candidate_kind"] == "length_abstain"
                and row["raw_default"]["numeric_candidate"] is not None):
            raw["numeric_on_truncated"] += 1
    return {
        "group_count": len(groups), "response_count": len(rows),
        "raw_default": dict(sorted(raw.items())),
        "modes": output,
        "acceptance_gate_by_mode": {
            mode: (
                counts[mode]["disagreed"] == 0
                and counts[mode]["extracted_on_human_unresolved_eos"] == 0
            )
            for mode in MODES
        },
    }


def main(manifest_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    from importlib.metadata import version

    from math_verify import parse, verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    manifest = json.loads(manifest_path.read_text())
    if manifest["experiment"] != "m6-math-verify-probe-v1":
        raise ValueError("wrong probe manifest")
    for name, expected in manifest["package"].items():
        if version(name) != expected:
            raise ValueError(f"package version differs: {name}")
    for name, expected_hash in manifest["source_file_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"source file changed: {name}")
    m5 = read_jsonl(Path("results/m5/format-signal-audit-v1/output/rows.jsonl"))
    if len(m5) != manifest["response_count"]:
        raise ValueError("M5 audit response count changed")
    responses = {}
    for arm in ("shortest", "hash_random"):
        for group in read_jsonl(Path(f"results/m2/sampling-order-v1/{arm}.jsonl")):
            for sample in group["samples"]:
                responses[(arm, group["prompt_id"], sample["sample_index"])] = sample

    anchored = [
        LatexExtractionConfig(try_extract_without_anchor=False),
        ExprExtractionConfig(try_extract_without_anchor=False),
    ]
    output_rows = []
    for index, manual in enumerate(m5, 1):
        key = manual["arm"], manual["prompt_id"], manual["sample_index"]
        sample = responses[key]
        response_text = sample["response_text"]
        if sample["finish_reason"] != manual["finish_reason"]:
            raise ValueError(f"finish reason differs: {key}")
        raw_parsed = parse(response_text)
        raw_numeric = numeric_candidate(raw_parsed)
        strict = parse_final_answer(response_text)
        if (strict is not None) != manual["format_valid"]:
            raise ValueError(f"strict format differs: {key}")
        if sample["finish_reason"] == "length":
            default_value = anchor_value = None
            default_source = anchor_source = "length_abstain"
        elif strict is not None:
            default_value = anchor_value = Rational(str(strict))
            default_source = anchor_source = "strict_final"
        else:
            default_value = raw_numeric
            default_source = "math_verify" if default_value is not None else "abstain"
            anchor_parsed = parse(
                response_text, extraction_config=anchored,
                fallback_mode="no_fallback", extraction_mode="first_match",
            )
            anchor_value = numeric_candidate(anchor_parsed)
            anchor_source = "math_verify" if anchor_value is not None else "abstain"
        row = {
            "arm": manual["arm"], "prompt_id": manual["prompt_id"],
            "sample_index": manual["sample_index"],
            "finish_reason": manual["finish_reason"],
            "candidate_kind": manual["candidate_kind"],
            "manual_candidate": manual["candidate"],
            "comparison": manual["comparison"],
            "raw_default": {
                "parsed": [str(item) for item in raw_parsed],
                "numeric_candidate": None if raw_numeric is None else str(raw_numeric),
            },
        }
        for name, value, source in (
            ("default_numeric_eos", default_value, default_source),
            ("anchor_numeric_eos", anchor_value, anchor_source),
        ):
            row[name] = {
                "candidate": None if value is None else str(value),
                "source": source,
                "comparison": compare_to_manual(value, manual, verify),
            }
        output_rows.append(row)
        if index % 32 == 0:
            print(f"Math-Verify: {index}/{len(m5)} responses", flush=True)
    summary = {
        "experiment": manifest["experiment"], "model_updated": False,
        "package": manifest["package"],
        "input_sha256": {
            "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            **manifest["source_file_sha256"],
        },
        **summarize(output_rows),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "rows.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in output_rows
    ))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("results/m6/math-verify-probe-v1/manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/m6/math-verify-probe-v1/output"),
    )
    args = parser.parse_args()
    print(json.dumps(main(args.manifest, args.output_dir), indent=2))
