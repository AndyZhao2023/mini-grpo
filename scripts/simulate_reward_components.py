"""Offline reward counterfactual on the saved M2 rollout groups; no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

import yaml

from src.data.dataset import load_pinned_train
from src.data.schema import Trajectory
from src.grpo.advantage import normalize_group_advantages
from src.reward.math_verifier import parse_final_answer, verify_answer


ARMS = ("shortest", "hash_random")
ALLOWED_TRAILERS = frozenset({
    "Use the same final-line format for the user's question.",
    "End of solution.",
    "Is there anything else you would like to see solved like the first problem?",
    "Please let me know if you need anymore assistance.",
})


def answer_candidate(response_text: str, finish_reason: str) -> tuple[Decimal | None, str]:
    """Abstain unless the answer line is final or has only a fixed harmless trailer."""
    final = parse_final_answer(response_text)
    if final is not None:
        return final, "valid_final"
    if finish_reason != "eos" or response_text.count("####") != 1:
        return None, "unknown"
    lines = response_text.strip().splitlines()
    matches = [(index, parse_final_answer(line)) for index, line in enumerate(lines)]
    matches = [(index, value) for index, value in matches if value is not None]
    if len(matches) != 1:
        return None, "unknown"
    index, value = matches[0]
    trailer = "\n".join(lines[index + 1:]).strip()
    if trailer in ALLOWED_TRAILERS:
        return value, "explicit_earlier_final"
    return None, "unknown"


def reward_vectors(labels: list[dict]) -> dict[str, list[float]]:
    strict = [float(label["strict_correct"]) for label in labels]
    format_valid = [float(label["format_valid"]) for label in labels]
    evidence = [float(label["answer_value_correct"] is True) for label in labels]
    return {
        "strict": strict,
        "strict_format_0.1": [a + 0.1 * f for a, f in zip(strict, format_valid)],
        "strict_format_0.01": [a + 0.01 * f for a, f in zip(strict, format_valid)],
        # An abstention is zero in this counterfactual, not a claim of an incorrect answer.
        "evidence_gated_format_0.1": [a * (1 + 0.1 * f) for a, f in zip(evidence, format_valid)],
    }


def group_advantages(rewards: list[float], algorithm: dict) -> tuple[list[float], bool]:
    """Use the production normalization on minimal, valid synthetic trajectories."""
    group = [Trajectory(
        schema_version=1, prompt_id="offline", group_id="offline",
        sample_index=index, group_size=len(rewards), prompt_text="offline",
        prompt_token_ids=[1], response_text="", response_token_ids=[1],
        response_mask=[1], finish_reason="eos", policy_version="offline",
        model_revision="offline", tokenizer_revision="offline", old_log_probs=[0.0],
        reward=reward, reward_reason="offline_counterfactual",
    ) for index, reward in enumerate(rewards)]
    result = normalize_group_advantages(group, algorithm)
    return [float(item.advantage) for item in result.group], result.zero_variance


def simulate(m2_dir: Path, config_path: Path) -> tuple[list[dict], dict]:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    split = load_pinned_train(config["data"])
    problems = {item.prompt_id: item for item in split.training}
    manifest_bytes = (m2_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest["source_sha256"] != split.source_sha256
            or manifest["run_config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
            or manifest["groups_per_arm"] != 64):
        raise ValueError("M2 manifest does not match pinned training data and run config")

    outputs = []
    source_hashes = {"manifest.json": hashlib.sha256(manifest_bytes).hexdigest()}
    counts = Counter()
    by_arm = {arm: Counter() for arm in ARMS}
    format_only_advantages = {"strict_format_0.1": [], "strict_format_0.01": []}
    for arm in ARMS:
        path = m2_dir / f"{arm}.jsonl"
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        expected = manifest["question_ids"][arm]
        if [row["prompt_id"] for row in rows] != expected or len(rows) != 64:
            raise ValueError(f"{arm} rows do not match the frozen M2 question order")
        for row in rows:
            problem = problems[row["prompt_id"]]
            samples = row["samples"]
            if len(samples) != config["generation"]["group_size"] or {
                sample["sample_index"] for sample in samples
            } != set(range(len(samples))):
                raise ValueError(f"incomplete group: {row['prompt_id']}")
            labels = []
            expected_answer = parse_final_answer(problem.reference_answer)
            for sample in sorted(samples, key=lambda item: item["sample_index"]):
                result = verify_answer(sample["response_text"], problem.reference_answer)
                if result.reward != sample["reward"] or result.reason != sample["reward_reason"]:
                    raise ValueError(f"saved strict reward differs: {row['prompt_id']}")
                candidate, source = answer_candidate(
                    sample["response_text"], sample["finish_reason"]
                )
                label = {
                    "sample_index": sample["sample_index"],
                    "finish_reason": sample["finish_reason"],
                    "format_valid": parse_final_answer(sample["response_text"]) is not None,
                    "strict_correct": result.reward == 1.0,
                    "answer_value_correct": (
                        None if candidate is None else candidate == expected_answer
                    ),
                    "answer_evidence": source,
                }
                labels.append(label)
                counts[f"evidence_{source}"] += 1
                by_arm[arm][f"evidence_{source}"] += 1
                if label["answer_value_correct"] is True:
                    counts["known_value_correct"] += 1
                    by_arm[arm]["known_value_correct"] += 1
                elif label["answer_value_correct"] is False:
                    counts["known_value_incorrect"] += 1
                    by_arm[arm]["known_value_incorrect"] += 1
                if label["format_valid"]:
                    counts["format_valid"] += 1
                    by_arm[arm]["format_valid"] += 1
            vectors = reward_vectors(labels)
            scenarios = {}
            for name, rewards in vectors.items():
                advantages, zero_variance = group_advantages(rewards, config["algorithm"])
                scenarios[name] = {
                    "rewards": rewards, "advantages": advantages,
                    "trainable": not zero_variance,
                }
                if not zero_variance:
                    counts[f"trainable_{name}"] += 1
                    by_arm[arm][f"trainable_{name}"] += 1
            if not scenarios["strict"]["trainable"] and scenarios["strict_format_0.1"]["trainable"]:
                counts["new_format_only_groups"] += 1
                by_arm[arm]["new_format_only_groups"] += 1
                for name, values in format_only_advantages.items():
                    values.extend(abs(value) for value in scenarios[name]["advantages"])
            if (not scenarios["strict"]["trainable"]
                    and scenarios["evidence_gated_format_0.1"]["trainable"]):
                counts["new_evidence_groups"] += 1
                by_arm[arm]["new_evidence_groups"] += 1
            outputs.append({
                "arm": arm, "prompt_id": row["prompt_id"],
                "labels": labels, "scenarios": scenarios,
            })
    summary = {
        "experiment": "m5-offline-reward-simulation-v1",
        "source": "M2 saved sampled training rollouts; no generation or optimizer update",
        "groups": len(outputs), "responses": sum(len(row["labels"]) for row in outputs),
        "source_sha256": split.source_sha256,
        "source_file_sha256": source_hashes,
        "counts": dict(sorted(counts.items())),
        "by_arm": {arm: dict(sorted(value.items())) for arm, value in by_arm.items()},
        "format_only_advantage_magnitude": {
            name: {"mean_absolute": statistics.fmean(values), "max_absolute": max(values)}
            for name, values in format_only_advantages.items()
        },
        "unknown_policy": "abstain on answer value; zero only for exploratory counterfactual",
    }
    return outputs, summary


def main(m2_dir: Path, config_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows, summary = simulate(m2_dir, config_path)
    output_dir.mkdir(parents=True)
    (output_dir / "groups.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    ))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-dir", type=Path, default=Path("results/m2/sampling-order-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/m0-run.yaml"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/m5/offline-reward-simulation-v1"),
    )
    args = parser.parse_args()
    print(json.dumps(main(args.m2_dir, args.config, args.output_dir), indent=2))
