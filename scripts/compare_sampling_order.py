"""Compare fixed-budget rollout reward density for two training-question orders.

This read-only experiment uses the M0 final checkpoint and original prompt.
It samples every selected group, including zero-variance groups, and never
constructs an optimizer or updates model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import torch
import transformers
import yaml
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.compare_prompt_quality import collect_prior_ids
from scripts.evaluate_official_test import verify_checkpoint
from scripts.find_trainable_group import order_training_problems
from src.data.dataset import MathProblem, load_pinned_train
from src.reward.math_verifier import verify_group
from src.rollout.hf_rollout import sample_group


def select_arms(
    training: Sequence[MathProblem], excluded: set[str], count: int, salt: str,
) -> dict[str, tuple[MathProblem, ...]]:
    """Compare M0's shortest-first rule with a disjoint SHA-ranked sample."""
    if type(count) is not int or count < 1 or not isinstance(salt, str) or not salt:
        raise ValueError("group count and selection salt must be positive and nonempty")
    available = [item for item in training if item.prompt_id not in excluded]
    if len({item.prompt_id for item in training}) != len(training):
        raise ValueError("training prompt IDs must be unique")
    if 2 * count > len(available):
        raise ValueError("not enough unused training questions for two disjoint arms")
    shortest = order_training_problems(available, "shortest")[:count]
    shortest_ids = {item.prompt_id for item in shortest}
    remaining = (item for item in available if item.prompt_id not in shortest_ids)
    ranked = sorted(remaining, key=lambda item: (
        hashlib.sha256(f"{salt}:{item.prompt_id}".encode()).digest(), item.prompt_id,
    ))
    return {"shortest": shortest, "hash_random": tuple(ranked[:count])}


def question_seed(base_seed: int, prompt_id: str) -> int:
    """Make each question's sampling independent of arm iteration order."""
    if type(base_seed) is not int or not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("base seed and prompt ID must be valid")
    digest = hashlib.sha256(f"{base_seed}:{prompt_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def summarize_attempts(attempts: Sequence[dict], expected_count: int, group_size: int) -> dict:
    if len(attempts) != expected_count or len({item["prompt_id"] for item in attempts}) != expected_count:
        raise ValueError("attempts must cover every selected question exactly once")
    if any(len(item["samples"]) != group_size for item in attempts):
        raise ValueError("every attempt must have one complete response group")
    reasons = Counter(sample["reward_reason"] for item in attempts for sample in item["samples"])
    mixed = sum(item["mixed_reward"] for item in attempts)
    if any(item["mixed_reward"] != (0 < sum(sample["reward"] for sample in item["samples"]) < group_size)
           for item in attempts):
        raise ValueError("mixed-reward flag does not match saved sample rewards")
    return {
        "group_count": expected_count,
        "sample_count": expected_count * group_size,
        "mixed_group_count": mixed,
        "mixed_group_fraction": mixed / expected_count,
        "zero_variance_group_count": expected_count - mixed,
        "correct_response_count": reasons["correct_answer"],
        "reward_reason_counts": dict(sorted(reasons.items())),
        "generated_response_tokens": sum(
            sample["response_tokens"] for item in attempts for sample in item["samples"]
        ),
        "generation_and_scoring_wall_seconds": sum(item["wall_seconds"] for item in attempts),
        "mean_question_words": sum(item["question_words"] for item in attempts) / expected_count,
    }


def run_arm(
    model, tokenizer, problems: Sequence[MathProblem], config: dict, output_path: Path,
    *, rollout_fn: Callable = sample_group,
) -> dict:
    """Write one auditable group per line, even when all rewards are zero."""
    attempts = []
    policy_version = "step-000002"
    with output_path.open("x", encoding="utf-8") as output:
        for index, problem in enumerate(problems, 1):
            seed = question_seed(config["seed"], problem.prompt_id)
            torch.manual_seed(seed)
            start = time.perf_counter()
            rollout = rollout_fn(model, tokenizer, problem, config, policy_version)
            scored = verify_group(problem, rollout)
            elapsed = time.perf_counter() - start
            samples = [{
                "sample_index": item.sample_index,
                "reward": item.reward,
                "reward_reason": item.reward_reason,
                "finish_reason": item.finish_reason,
                "response_tokens": sum(item.response_mask),
                "response_text": item.response_text,
            } for item in scored]
            record = {
                "prompt_id": problem.prompt_id,
                "question_words": len(problem.question.split()),
                "question_seed": seed,
                "mixed_reward": len({item["reward"] for item in samples}) > 1,
                "wall_seconds": elapsed,
                "samples": samples,
            }
            attempts.append(record)
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if index % 8 == 0 or index == len(problems):
                output.flush()
                print(f"{output_path.stem}: {index}/{len(problems)} groups", flush=True)
    return summarize_attempts(attempts, len(problems), config["generation"]["group_size"])


def main(spec_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    spec_bytes = spec_path.read_bytes()
    spec = yaml.safe_load(spec_bytes)
    if spec["selection"]["arms"] != ["shortest", "hash_random"]:
        raise ValueError("selection arms must be shortest and hash_random")
    config_bytes = Path(spec["run_config"]).read_bytes()
    config = yaml.safe_load(config_bytes)
    run_bytes = (Path(spec["run_dir"]) / "summary.json").read_bytes()
    run = json.loads(run_bytes)
    checkpoint = Path(spec["checkpoint_dir"])
    if (run["status"] != "completed" or not run["steps"]
            or run["config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
            or Path(run["steps"][-1]["checkpoint_dir"]).resolve() != checkpoint.resolve()
            or run["steps"][-1]["policy_after"] != checkpoint.name
            or config["run"]["order"] != "shortest"):
        raise ValueError("M0 run, original shortest-first order, and checkpoint must agree")
    if config["model"]["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    checkpoint_manifest_sha256 = verify_checkpoint(Path(spec["checkpoint_sha256_file"]), checkpoint)
    split = load_pinned_train(config["data"])
    if split.source_sha256 != run["source_sha256"]:
        raise ValueError("source run and pinned training data differ")
    prior_spec_bytes = Path(spec["prior_prompt_spec"]).read_bytes()
    prior_spec = yaml.safe_load(prior_spec_bytes)
    excluded, prior_hashes = collect_prior_ids(prior_spec, split.source_sha256)
    development_bytes = Path(spec["development_manifest"]).read_bytes()
    confirmation_bytes = Path(spec["confirmation_manifest"]).read_bytes()
    development = json.loads(development_bytes)
    confirmation = json.loads(confirmation_bytes)
    if (development["source_sha256"] != split.source_sha256
            or confirmation["source_sha256"] != split.source_sha256
            or development["spec_sha256"] != hashlib.sha256(prior_spec_bytes).hexdigest()
            or confirmation["reserved_prompt_ids"] != development["reserved_prompt_ids"]):
        raise ValueError("prior M1 development and confirmation manifests disagree")
    excluded.update(development["development_prompt_ids"] + development["reserved_prompt_ids"])
    if not excluded <= {item.prompt_id for item in split.training}:
        raise ValueError("excluded question is outside the training partition")
    arms = select_arms(
        split.training, excluded, spec["selection"]["groups_per_arm"], spec["selection"]["salt"],
    )
    if {item.prompt_id for values in arms.values() for item in values} & {item.prompt_id for item in split.validation}:
        raise ValueError("selection overlaps M0's fixed validation partition")
    tokenizer_config = try_to_load_from_cache(
        config["model"]["id"], "tokenizer_config.json",
        revision=config["model"]["tokenizer_revision"],
    )
    if not isinstance(tokenizer_config, str):
        raise FileNotFoundError("pinned original tokenizer snapshot is not cached")
    tokenizer = AutoTokenizer.from_pretrained(Path(tokenizer_config).parent, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, trust_remote_code=False,
    ).to(config["model"]["device"])
    model.eval()
    output_dir.mkdir(parents=True)
    manifest = {
        "experiment": spec["experiment"], "source_split": "train",
        "purpose": "fixed-budget comparison of training-question order; no model update",
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "run_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_run_summary_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "source_sha256": split.source_sha256,
        "checkpoint_sha256_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_dir": str(checkpoint),
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "excluded_prompt_count": len(excluded),
        "prior_evidence_sha256": {
            **prior_hashes,
            str(spec["prior_prompt_spec"]): hashlib.sha256(prior_spec_bytes).hexdigest(),
            str(spec["development_manifest"]): hashlib.sha256(development_bytes).hexdigest(),
            str(spec["confirmation_manifest"]): hashlib.sha256(confirmation_bytes).hexdigest(),
        },
        "selection_salt": spec["selection"]["salt"],
        "groups_per_arm": spec["selection"]["groups_per_arm"],
        "question_ids": {name: [item.prompt_id for item in values] for name, values in arms.items()},
        "question_seed_rule": "first eight SHA-256 bytes of '<base seed>:<prompt ID>', big-endian",
        "base_seed": config["seed"],
        "generation": config["generation"],
        "system_prompt": config["prompt"]["system"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    arm_summaries = {}
    for name, problems in arms.items():
        arm_summaries[name] = run_arm(model, tokenizer, problems, config, output_dir / f"{name}.jsonl")
    summary = {
        "status": "completed", "model_updated": False,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "shortest": arm_summaries["shortest"],
        "hash_random": arm_summaries["hash_random"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/m2-sampling-order-v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m2/sampling-order-v1"))
    args = parser.parse_args()
    print(main(args.spec, args.output_dir))
