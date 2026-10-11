"""Compare two prompts on fresh training-partition development questions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
import transformers
import yaml
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evaluate_official_test import verify_checkpoint
from src.data.dataset import MathProblem, load_pinned_train
from src.evaluation import evaluate_model


def select_development_and_reserve(
    training: Sequence[MathProblem], excluded_ids: set[str], salt: str,
    development_count: int, reserved_count: int,
) -> tuple[tuple[MathProblem, ...], tuple[MathProblem, ...]]:
    """Hash-rank unseen training IDs before evaluating either prompt."""
    if not isinstance(salt, str) or not salt:
        raise ValueError("selection salt must be nonempty")
    if (type(development_count) is not int or development_count < 1
            or type(reserved_count) is not int or reserved_count < 1):
        raise ValueError("development and reserved counts must be positive integers")
    available = [item for item in training if item.prompt_id not in excluded_ids]
    if len({item.prompt_id for item in available}) != len(available):
        raise ValueError("training prompt IDs must be unique")
    if development_count + reserved_count > len(available):
        raise ValueError("selection counts exceed unseen training questions")
    ranked = sorted(available, key=lambda item: (
        hashlib.sha256(f"{salt}:{item.prompt_id}".encode()).digest(), item.prompt_id,
    ))
    return (
        tuple(ranked[:development_count]),
        tuple(ranked[development_count:development_count + reserved_count]),
    )


def collect_prior_ids(spec: dict, source_sha256: str) -> tuple[set[str], dict[str, str]]:
    """Bind excluded prompts to explicit, versioned input files."""
    files = [Path(spec["prior_diagnostic_manifest"])] + [
        Path(value) for value in spec["prior_scan_files"]
    ]
    if len(set(files)) != len(files):
        raise ValueError("prior evidence file paths must be unique")
    excluded: set[str] = set()
    hashes = {}
    for path in files:
        raw = path.read_bytes()
        record = json.loads(raw)
        if record["source_sha256"] != source_sha256:
            raise ValueError(f"prior evidence has a different dataset hash: {path}")
        if path == files[0]:
            if record["source_split"] != "train":
                raise ValueError("prior diagnostic must use the train partition")
            ids = record["prompt_ids"]
        else:
            ids = [item["prompt_id"] for item in record["attempts"]]
        if not ids or any(not isinstance(value, str) for value in ids):
            raise ValueError(f"prior evidence has no valid prompt IDs: {path}")
        excluded.update(ids)
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return excluded, hashes


def summarize(rows: Sequence[dict]) -> dict:
    reasons = Counter(row["reward_reason"] for row in rows)
    return {
        "question_count": len(rows),
        "strict_correct_count": reasons["correct_answer"],
        "strict_pass_at_1": reasons["correct_answer"] / len(rows),
        "reward_reason_counts": dict(sorted(reasons.items())),
        "truncated_count": sum(row["finish_reason"] == "length" for row in rows),
        "mean_response_tokens": sum(row["response_tokens"] for row in rows) / len(rows),
    }


def compare_paired(baseline: Sequence[dict], variant: Sequence[dict]) -> dict:
    if not baseline or [row["prompt_id"] for row in baseline] != [row["prompt_id"] for row in variant]:
        raise ValueError("both prompt arms must contain the same ordered questions")
    return {
        "newly_strict_correct": sum(a["reward"] == 0 and b["reward"] == 1 for a, b in zip(baseline, variant)),
        "lost_strict_correct": sum(a["reward"] == 1 and b["reward"] == 0 for a, b in zip(baseline, variant)),
        "correct_in_both": sum(a["reward"] == 1 and b["reward"] == 1 for a, b in zip(baseline, variant)),
        "identical_response_text": sum(a["response_text"] == b["response_text"] for a, b in zip(baseline, variant)),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _evaluate_to_file(
    model, tokenizer, problems: Sequence[MathProblem], config: dict, path: Path, label: str,
) -> list[dict]:
    rows = []
    with path.open("x", encoding="utf-8") as output:
        for index, problem in enumerate(problems, 1):
            result, _ = evaluate_model(model, tokenizer, (problem,), config)
            rows.append(result[0])
            output.write(json.dumps(result[0], ensure_ascii=False, allow_nan=False) + "\n")
            if index % 10 == 0 or index == len(problems):
                output.flush()
                print(f"{label}: {index}/{len(problems)}", flush=True)
    return rows


def main(spec_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    spec_bytes = spec_path.read_bytes()
    spec = yaml.safe_load(spec_bytes)
    config_bytes = Path(spec["run_config"]).read_bytes()
    config = yaml.safe_load(config_bytes)
    run_bytes = (Path(spec["run_dir"]) / "summary.json").read_bytes()
    run = json.loads(run_bytes)
    checkpoint_dir = Path(spec["checkpoint_dir"])
    if (run["status"] != "completed" or not run["steps"]
            or run["config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
            or run["steps"][-1]["policy_after"] != checkpoint_dir.name
            or Path(run["steps"][-1]["checkpoint_dir"]).resolve() != checkpoint_dir.resolve()
            or config["evaluation"]["do_sample"] is not False):
        raise ValueError("completed M0 run, checkpoint, config, and greedy evaluation must agree")
    variant_system = spec["variant"]["system"]
    if not isinstance(variant_system, str) or not variant_system.strip() or variant_system == config["prompt"]["system"]:
        raise ValueError("variant prompt must differ from the baseline")
    if config["model"]["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    checkpoint_hash = verify_checkpoint(Path(spec["checkpoint_sha256_file"]), checkpoint_dir)
    split = load_pinned_train(config["data"])
    if run["source_sha256"] != split.source_sha256:
        raise ValueError("source run and pinned training data differ")
    excluded, exclusion_hashes = collect_prior_ids(spec, split.source_sha256)
    train_ids = {item.prompt_id for item in split.training}
    if not excluded <= train_ids:
        raise ValueError("prior exclusion includes a non-training question")
    development, reserved = select_development_and_reserve(
        split.training, excluded, spec["selection"]["salt"],
        spec["selection"]["development_count"], spec["selection"]["reserved_count"],
    )
    if {item.prompt_id for item in development + reserved} & {item.prompt_id for item in split.validation}:
        raise ValueError("new selection overlaps the fixed M0 validation partition")
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

    manifest = {
        "experiment": spec["experiment"], "source_split": "train",
        "purpose": "development prompt comparison; reserved IDs are not evaluated",
        "dataset_revision": config["data"]["revision"],
        "source_sha256": split.source_sha256,
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "run_config_sha256": run["config_sha256"],
        "source_run_summary_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256_manifest_sha256": checkpoint_hash,
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "excluded_prior_prompt_count": len(excluded),
        "prior_evidence_sha256": exclusion_hashes,
        "selection_salt": spec["selection"]["salt"],
        "development_prompt_ids": [item.prompt_id for item in development],
        "reserved_prompt_ids": [item.prompt_id for item in reserved],
        "baseline_system": config["prompt"]["system"],
        "variant_system": variant_system,
        "evaluation": config["evaluation"],
    }
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "manifest.json", manifest)
    baseline = _evaluate_to_file(
        model, tokenizer, development, config, output_dir / "baseline.jsonl", "baseline",
    )
    variant_config = {**config, "prompt": {**config["prompt"], "system": variant_system}}
    variant = _evaluate_to_file(
        model, tokenizer, development, variant_config, output_dir / "two-step-example.jsonl", "two-step-example",
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
    parser.add_argument("--spec", type=Path, default=Path("configs/m1-prompt-two-step.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m1/prompt-two-step-v1"))
    args = parser.parse_args()
    print(main(args.spec, args.output_dir))
