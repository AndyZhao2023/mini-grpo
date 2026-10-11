"""Evaluate the preselected two-prompt comparison once on reserved questions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
import yaml
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.compare_prompt_quality import (
    _evaluate_to_file, _write_json, collect_prior_ids, compare_paired,
    select_development_and_reserve, summarize,
)
from scripts.evaluate_official_test import verify_checkpoint
from src.data.dataset import MathProblem, TrainValidationSplit, load_pinned_train


def validate_reserved_selection(
    manifest: dict, source_spec: dict, split: TrainValidationSplit, excluded: set[str],
) -> tuple[MathProblem, ...]:
    """Use exactly the IDs reserved before the development outputs were seen."""
    development, reserved = select_development_and_reserve(
        split.training, excluded, source_spec["selection"]["salt"],
        source_spec["selection"]["development_count"],
        source_spec["selection"]["reserved_count"],
    )
    if (manifest["source_split"] != "train"
            or manifest["source_sha256"] != split.source_sha256
            or manifest["development_prompt_ids"] != [item.prompt_id for item in development]
            or manifest["reserved_prompt_ids"] != [item.prompt_id for item in reserved]
            or manifest["excluded_prior_prompt_count"] != len(excluded)
            or {item.prompt_id for item in reserved} & {item.prompt_id for item in split.validation}):
        raise ValueError("reserved questions differ from the pre-evaluation manifest")
    return reserved


def main(confirm_spec_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    confirm_bytes = confirm_spec_path.read_bytes()
    confirmation = yaml.safe_load(confirm_bytes)
    source_spec_bytes = Path(confirmation["source_spec"]).read_bytes()
    source_spec = yaml.safe_load(source_spec_bytes)
    development_dir = Path(confirmation["development_dir"])
    development_manifest_bytes = (development_dir / "manifest.json").read_bytes()
    development_manifest = json.loads(development_manifest_bytes)
    development_summary_bytes = (development_dir / "summary.json").read_bytes()
    development_summary = json.loads(development_summary_bytes)
    if (development_summary["status"] != "completed"
            or development_manifest["spec_sha256"] != hashlib.sha256(source_spec_bytes).hexdigest()):
        raise ValueError("development experiment and source spec must agree")
    config = yaml.safe_load(Path(source_spec["run_config"]).read_text(encoding="utf-8"))
    checkpoint_dir = Path(source_spec["checkpoint_dir"])
    if config["model"]["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    checkpoint_hash = verify_checkpoint(Path(source_spec["checkpoint_sha256_file"]), checkpoint_dir)
    if (development_manifest["checkpoint_sha256_manifest_sha256"] != checkpoint_hash
            or development_manifest["baseline_system"] != config["prompt"]["system"]
            or development_manifest["variant_system"] != source_spec["variant"]["system"]):
        raise ValueError("checkpoint or prompts differ from the development experiment")
    split = load_pinned_train(config["data"])
    excluded, exclusion_hashes = collect_prior_ids(source_spec, split.source_sha256)
    if (development_manifest["prior_evidence_sha256"] != exclusion_hashes
            or not excluded <= {item.prompt_id for item in split.training}):
        raise ValueError("prior question exclusions have changed")
    reserved = validate_reserved_selection(development_manifest, source_spec, split, excluded)
    tokenizer_config = try_to_load_from_cache(
        config["model"]["id"], "tokenizer_config.json",
        revision=config["model"]["tokenizer_revision"],
    )
    if not isinstance(tokenizer_config, str):
        raise FileNotFoundError("pinned original tokenizer snapshot is not cached")
    tokenizer = AutoTokenizer.from_pretrained(Path(tokenizer_config).parent, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, dtype=torch.float32, trust_remote_code=False,
    ).to(config["model"]["device"])
    model.eval()
    torch.manual_seed(config["seed"])
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "manifest.json", {
        "experiment": confirmation["experiment"], "source_split": "train",
        "purpose": "one confirmation on reserved questions; no model update",
        "confirmation_spec_sha256": hashlib.sha256(confirm_bytes).hexdigest(),
        "development_manifest_sha256": hashlib.sha256(development_manifest_bytes).hexdigest(),
        "development_summary_sha256": hashlib.sha256(development_summary_bytes).hexdigest(),
        "source_sha256": split.source_sha256,
        "checkpoint_sha256_manifest_sha256": checkpoint_hash,
        "reserved_prompt_ids": [item.prompt_id for item in reserved],
        "selection_reason": confirmation["selection_reason"],
        "interpretation_gate": confirmation["interpretation_gate"],
        "baseline_system": config["prompt"]["system"],
        "variant_system": source_spec["variant"]["system"],
        "evaluation": config["evaluation"],
    })
    baseline = _evaluate_to_file(
        model, tokenizer, reserved, config, output_dir / "baseline.jsonl", "reserved-baseline",
    )
    variant_config = {
        **config, "prompt": {**config["prompt"], "system": source_spec["variant"]["system"]},
    }
    variant = _evaluate_to_file(
        model, tokenizer, reserved, variant_config,
        output_dir / "two-step-example.jsonl", "reserved-two-step-example",
    )
    summary = {
        "status": "completed", "model_updated": False,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "baseline": summarize(baseline), "two_step_example": summarize(variant),
        "paired": compare_paired(baseline, variant),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/m1-prompt-two-step-confirm.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m1/prompt-two-step-confirm-v1"))
    args = parser.parse_args()
    print(main(args.spec, args.output_dir))
