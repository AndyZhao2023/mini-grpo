"""Normalize a scored GSM8K rollout group and save training-ready trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.schema import read_jsonl, validate_group, write_jsonl
from src.grpo.advantage import normalize_group_advantages


def main(config_path: Path, input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    group = read_jsonl(input_path)
    validate_group(group)
    if group[0].group_size != config["generation"]["group_size"]:
        raise ValueError("scored group size differs from config")
    if (
        group[0].model_revision != config["model"]["revision"]
        or group[0].tokenizer_revision != config["model"]["tokenizer_revision"]
    ):
        raise ValueError("scored model/tokenizer revision differs from config")
    result = normalize_group_advantages(group, config["algorithm"])
    write_jsonl(output_path, result.group)
    if read_jsonl(output_path) != result.group:
        raise RuntimeError("normalized trajectory group changed during JSONL round trip")
    print(
        f"Normalized {len(result.group)} trajectories for {group[0].prompt_id}: "
        f"mean={result.reward_mean:.6g}, population_std={result.reward_std:.6g}, "
        f"zero_variance={str(result.zero_variance).lower()} -> {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0.yaml"))
    parser.add_argument("--input", type=Path, default=Path("results/m0/rewarded-sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/m0/advantaged-sample.jsonl"))
    args = parser.parse_args()
    main(args.config, args.input, args.output)
