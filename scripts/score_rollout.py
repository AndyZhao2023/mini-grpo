"""Score a saved GSM8K rollout group using the pinned train references."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from src.data.dataset import load_pinned_train
from src.data.schema import read_jsonl, validate_group, write_jsonl
from src.reward.math_verifier import verify_group


def main(config_path: Path, input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    group = read_jsonl(input_path)
    validate_group(group)
    if group[0].group_size != config["generation"]["group_size"]:
        raise ValueError("rollout group size differs from config")
    if (
        group[0].model_revision != config["model"]["revision"]
        or group[0].tokenizer_revision != config["model"]["tokenizer_revision"]
    ):
        raise ValueError("rollout model/tokenizer revision differs from config")
    split = load_pinned_train(config["data"])
    training_by_id = {problem.prompt_id: problem for problem in split.training}
    problem = training_by_id.get(group[0].prompt_id)
    if problem is None:
        raise ValueError("rollout prompt is not in the pinned training partition")
    scored = verify_group(problem, group)
    write_jsonl(output_path, scored)
    if read_jsonl(output_path) != scored:
        raise RuntimeError("scored trajectory group changed during JSONL round trip")
    print(
        f"Scored {len(scored)} trajectories for {problem.prompt_id}: "
        f"{dict(Counter(item.reward_reason for item in scored))} -> {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0.yaml"))
    parser.add_argument("--input", type=Path, default=Path("results/m0/rollout-sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/m0/rewarded-sample.jsonl"))
    args = parser.parse_args()
    main(args.config, args.input, args.output)
