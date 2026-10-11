"""Train fresh GRPO rollouts from the retained M0 weights with a new optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.compare_prompt_quality import collect_prior_ids
from scripts.evaluate_official_test import verify_checkpoint
from scripts.find_trainable_group import order_training_problems
from scripts.run_m0 import run_cycles
from src.data.dataset import TrainValidationSplit, load_pinned_train


def _read_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def select_fresh_split(
    split: TrainValidationSplit, excluded: set[str], *,
    training_pool_size: int, validation_size: int, validation_salt: str,
) -> TrainValidationSplit:
    """Reserve unseen validation IDs after fixing the shortest training pool."""
    if (type(training_pool_size) is not int or training_pool_size < 1
            or type(validation_size) is not int or validation_size < 1
            or not isinstance(validation_salt, str) or not validation_salt):
        raise ValueError("training pool, validation size, and salt must be valid")
    training_ids = {item.prompt_id for item in split.training}
    if len(training_ids) != len(split.training) or not excluded <= training_ids:
        raise ValueError("excluded IDs must be unique training-partition questions")
    available = [item for item in split.training if item.prompt_id not in excluded]
    if training_pool_size + validation_size > len(available):
        raise ValueError("not enough unused training questions")
    pool = order_training_problems(available, "shortest")[:training_pool_size]
    pool_ids = {item.prompt_id for item in pool}
    remaining = (item for item in available if item.prompt_id not in pool_ids)
    held_out = tuple(sorted(remaining, key=lambda item: (
        hashlib.sha256(f"{validation_salt}:{item.prompt_id}".encode()).digest(),
        item.prompt_id,
    ))[:validation_size])
    return TrainValidationSplit(pool, held_out, split.source_sha256)


def prepare_run(config: dict, config_sha256: str) -> tuple[TrainValidationSplit, dict, Path]:
    """Verify M0 lineage and prior question usage before loading model weights."""
    warm = config["warm_start"]
    source_config_path = Path(warm["source_run_config"])
    source_bytes = source_config_path.read_bytes()
    source_config = yaml.safe_load(source_bytes)
    source_run, source_run_hash = _read_json(Path(warm["source_run_summary"]))
    checkpoint = Path(config["model"]["revision"])
    if (source_run["status"] != "completed" or source_run["completed_steps"] != 2
            or source_run["config_sha256"] != hashlib.sha256(source_bytes).hexdigest()
            or Path(source_run["steps"][-1]["checkpoint_dir"]).resolve() != checkpoint.resolve()
            or source_config["model"]["id"] != config["model"]["id"]
            or source_config["model"]["tokenizer_revision"] != config["model"]["tokenizer_revision"]
            or any(source_config[key] != config[key] for key in
                   ("data", "prompt", "generation", "algorithm", "optimizer"))
            or config["algorithm"]["update_epochs"] != 1
            or config["evaluation"]["do_sample"] is not False
            or config["evaluation"]["run_official_test_once"] is not False
            or config["evaluation"]["max_new_tokens"] != source_config["evaluation"]["max_new_tokens"]
            or config["run"]["order"] != "shortest"
            or config["run"].get("skip_scanned_prompts") is not True):
        raise ValueError("warm-start config does not match the completed M0 source and fixed training contract")
    checkpoint_hash = verify_checkpoint(Path(warm["checkpoint_sha256_file"]), checkpoint)
    original = load_pinned_train(config["data"])
    if source_run["source_sha256"] != original.source_sha256:
        raise ValueError("source M0 run used different GSM8K data")

    prior_spec_path = Path(warm["prior_prompt_spec"])
    prior_bytes = prior_spec_path.read_bytes()
    excluded, prior_hashes = collect_prior_ids(yaml.safe_load(prior_bytes), original.source_sha256)
    m1_path, m2_path = Path(warm["prior_m1_manifest"]), Path(warm["prior_m2_manifest"])
    m1, m1_hash = _read_json(m1_path)
    m2, m2_hash = _read_json(m2_path)
    if (m1["source_sha256"] != original.source_sha256
            or m2["source_sha256"] != original.source_sha256
            or m1["spec_sha256"] != hashlib.sha256(prior_bytes).hexdigest()
            or m1["checkpoint_sha256_manifest_sha256"] != checkpoint_hash
            or m2["checkpoint_sha256_manifest_sha256"] != checkpoint_hash
            or set(m2["question_ids"]) != {"shortest", "hash_random"}):
        raise ValueError("prior M1/M2 evidence does not match M0 data and checkpoint")
    m1_ids = m1["development_prompt_ids"] + m1["reserved_prompt_ids"]
    m2_ids = m2["question_ids"]["shortest"] + m2["question_ids"]["hash_random"]
    if len(set(m1_ids)) != len(m1_ids) or len(set(m2_ids)) != len(m2_ids):
        raise ValueError("prior M1/M2 question IDs must not repeat")
    excluded.update(m1_ids + m2_ids)
    selected = select_fresh_split(
        original, excluded,
        training_pool_size=warm["training_pool_size"],
        validation_size=warm["validation_size"],
        validation_salt=warm["validation_salt"],
    )
    manifest = {
        "experiment": config["experiment"],
        "source_split": "train",
        "source_count": len(original.training) + len(original.validation),
        "source_sha256": original.source_sha256,
        "dataset_revision": config["data"]["revision"],
        "config_sha256": config_sha256,
        "source_run_summary_sha256": source_run_hash,
        "source_run_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "starting_checkpoint": str(checkpoint),
        "checkpoint_sha256_manifest_sha256": checkpoint_hash,
        "optimizer_state_restored": False,
        "prior_evidence_sha256": {
            **prior_hashes,
            str(prior_spec_path): hashlib.sha256(prior_bytes).hexdigest(),
            str(m1_path): m1_hash,
            str(m2_path): m2_hash,
        },
        "excluded_prior_prompt_count": len(excluded),
        "training_selection": "shortest unused training-partition questions",
        "training_prompt_ids": [item.prompt_id for item in selected.training],
        "validation_selection": "SHA-256 rank of salt:prompt_id among remaining unused questions",
        "validation_salt": warm["validation_salt"],
        "validation_prompt_ids": [item.prompt_id for item in selected.validation],
        "original_m0_validation_count": len(original.validation),
        "official_test_used": False,
    }
    return selected, manifest, checkpoint


def main(config_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["model"]["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    if config["model"]["dtype"] != "float32" or config["model"]["training_method"] != "full_finetune":
        raise ValueError("warm-start runner requires float32 full-parameter training")
    if config["algorithm"]["beta"] != 0.0 or config["optimizer"]["name"] != "AdamW":
        raise ValueError("warm-start runner requires beta=0 and AdamW")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    split, manifest, checkpoint = prepare_run(config, config_hash)
    cached_tokenizer = try_to_load_from_cache(
        config["model"]["id"], "tokenizer_config.json",
        revision=config["model"]["tokenizer_revision"],
    )
    if not isinstance(cached_tokenizer, str):
        raise FileNotFoundError("pinned original tokenizer snapshot is not cached")
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(Path(cached_tokenizer).parent, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, trust_remote_code=False,
    ).to(config["model"]["device"])
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]), eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )
    return run_cycles(
        model, tokenizer, optimizer, split, config, output_dir,
        config_sha256=config_hash, dataset_manifest=manifest,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m3-warm-start-v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m3/warm-start-v1"))
    args = parser.parse_args()
    result = main(args.config, args.output_dir)
    print(f"M3 run {result['status']}: {result['completed_steps']}/{result['requested_steps']} updates -> {args.output_dir}")
    if result["status"] != "completed":
        raise SystemExit(2)
