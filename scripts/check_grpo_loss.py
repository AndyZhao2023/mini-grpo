"""Recompute current log probabilities and inspect GRPO loss without an update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.schema import read_jsonl, validate_group
from src.grpo.loss import clipped_grpo_loss, pack_group
from src.rollout.hf_rollout import gather_response_log_probs


def main(config_path: Path, input_path: Path, output_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config, algorithm = config["model"], config["algorithm"]
    if model_config["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    if model_config["dtype"] != "float32":
        raise ValueError("M0 loss check currently supports float32 only")
    if algorithm["beta"] != 0.0:
        raise ValueError("M0 loss check requires beta=0")
    group = read_jsonl(input_path)
    validate_group(group, require_complete=True)
    if group[0].group_size != config["generation"]["group_size"]:
        raise ValueError("trajectory group size differs from config")
    if (
        group[0].model_revision != model_config["revision"]
        or group[0].tokenizer_revision != model_config["tokenizer_revision"]
    ):
        raise ValueError("trajectory model/tokenizer revision differs from config")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["tokenizer_revision"],
        trust_remote_code=False,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    packed = pack_group(group, pad_token_id, model_config["device"])
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"], revision=model_config["revision"],
        dtype=torch.float32, trust_remote_code=False,
    ).to(model_config["device"])
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=packed.input_ids,
            attention_mask=packed.attention_mask,
        ).logits
        new_log_probs = gather_response_log_probs(
            logits, packed.response_ids, packed.prompt_length,
        ).unsqueeze(0)
        result = clipped_grpo_loss(
            new_log_probs, packed.old_log_probs, packed.advantages,
            packed.response_mask, algorithm["clip_epsilon"], algorithm["loss_reduction"],
        )
    record = {
        "prompt_id": group[0].prompt_id,
        "policy_version": group[0].policy_version,
        "model_revision": model_config["revision"],
        "group_size": len(group),
        "valid_token_count": result.valid_token_count,
        "policy_loss": result.loss.item(),
        "mean_ratio": result.mean_ratio.item(),
        "max_ratio_deviation": result.max_ratio_deviation.item(),
        "clip_fraction": result.clip_fraction.item(),
        "approx_kl_old": result.approx_kl_old.item(),
        "pre_update_ratio_tolerance": 1e-3,
        "no_optimizer_step": True,
    }
    if record["max_ratio_deviation"] > record["pre_update_ratio_tolerance"]:
        raise RuntimeError("pre-update ratio differs from 1 beyond tolerance")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        f"Loss check: loss={record['policy_loss']:.6g}, "
        f"mean_ratio={record['mean_ratio']:.6g}, "
        f"max_ratio_deviation={record['max_ratio_deviation']:.6g}, "
        f"clip_fraction={record['clip_fraction']:.6g} -> {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0.yaml"))
    parser.add_argument("--input", type=Path, default=Path("results/m0/advantaged-sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/m0/loss-smoke.json"))
    args = parser.parse_args()
    main(args.config, args.input, args.output)
