"""Save one real GSM8K training prompt's four-response rollout as JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import load_pinned_train
from src.data.schema import read_jsonl, validate_group, write_jsonl
from src.rollout.hf_rollout import sample_group


def main(config_path: Path, output_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config["model"]
    if model_config["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    if model_config["dtype"] != "float32":
        raise ValueError("M0 rollout currently supports float32 only")
    split = load_pinned_train(config["data"])
    problem = split.training[0]
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["tokenizer_revision"],
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"], revision=model_config["revision"],
        dtype=torch.float32, trust_remote_code=False,
    ).to(model_config["device"])
    group = sample_group(model, tokenizer, problem, config, "step-000000")
    write_jsonl(output_path, group)
    saved = read_jsonl(output_path)
    validate_group(saved)
    if saved != group:
        raise RuntimeError("written trajectory group changed during JSONL round trip")
    print(
        f"Saved {len(saved)} trajectories for {problem.prompt_id}; "
        f"response token lengths: {[sum(row.response_mask) for row in saved]} → {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/m0/rollout-sample.jsonl"))
    args = parser.parse_args()
    main(args.config, args.output)
