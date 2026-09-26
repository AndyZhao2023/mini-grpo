"""Fetch pinned GSM8K train and save the deterministic split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.data.dataset import build_manifest, load_pinned_train


def main(config_path: Path, output_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = load_pinned_train(config["data"])
    manifest = build_manifest(split, config["data"], config["prompt"]["system"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"GSM8K train: {manifest['training_count']} training, "
        f"{manifest['validation_count']} validation → {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/m0/dataset-split.json"))
    args = parser.parse_args()
    main(args.config, args.output)
