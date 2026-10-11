"""Run the one planned GSM8K official-test evaluation of the selected M0 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import MathProblem, load_pinned_test
from src.evaluation import evaluate_model


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_candidate(
    run: dict, config: dict, config_sha256: str, checkpoint_dir: Path,
) -> None:
    """Bind the official test to one completed, previously selected run."""
    if run["status"] != "completed" or not run["steps"]:
        raise ValueError("source run must be completed and have a trained checkpoint")
    if run["config_sha256"] != config_sha256:
        raise ValueError("source run config SHA-256 does not match")
    if (run["dataset_revision"] != config["data"]["revision"]
            or run["initial_model_revision"] != config["model"]["revision"]):
        raise ValueError("source run model or dataset revision does not match config")
    if (run["steps"][-1]["policy_after"] != checkpoint_dir.name
            or Path(run["steps"][-1]["checkpoint_dir"]).resolve() != checkpoint_dir.resolve()):
        raise ValueError("checkpoint does not match the source run's final policy")
    if config["evaluation"]["do_sample"] is not False or config["evaluation"]["run_official_test_once"] is not True:
        raise ValueError("source config must specify one greedy official-test evaluation")


def verify_checkpoint(checksum_file: Path, checkpoint_dir: Path) -> str:
    """Check every file named by the published final-checkpoint manifest."""
    manifest = checksum_file.read_bytes()
    checked = 0
    for line in manifest.decode("utf-8").splitlines():
        expected, relative_path = line.split("  ", 1)
        path = Path(relative_path)
        if path.parent.resolve() != checkpoint_dir.resolve():
            raise ValueError("checkpoint checksum path is outside the selected checkpoint")
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"checkpoint SHA-256 differs: {path.name}")
        checked += 1
    if checked == 0:
        raise ValueError("checkpoint checksum manifest is empty")
    return hashlib.sha256(manifest).hexdigest()


def evaluate_to_artifacts(
    model, tokenizer, problems: tuple[MathProblem, ...] | list[MathProblem],
    config: dict, output_dir: Path, manifest: dict,
    *, evaluation_fn=evaluate_model,
) -> dict:
    """Evaluate all test questions once and save inspectable rows and counts."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not problems or manifest["source_split"] != "test":
        raise ValueError("official test requires a nonempty test partition")
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "manifest.json", manifest)
    rows, metrics = evaluation_fn(model, tokenizer, problems, config)
    expected_ids = [problem.prompt_id for problem in problems]
    if ([row["prompt_id"] for row in rows] != expected_ids
            or metrics["question_count"] != len(problems)
            or metrics["correct_count"] != sum(row["reward"] == 1 for row in rows)
            or sum(metrics["reason_counts"].values()) != len(problems)):
        raise ValueError("evaluation rows do not match the official test partition")
    with (output_dir / "test-rows.jsonl").open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {"status": "completed", "source_split": "test", "metrics": metrics}
    _write_json(output_dir / "summary.json", summary)
    return summary


def main(spec_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    config_bytes = Path(spec["run_config"]).read_bytes()
    config = yaml.safe_load(config_bytes)
    run_bytes = (Path(spec["run_dir"]) / "summary.json").read_bytes()
    run = json.loads(run_bytes)
    checkpoint_dir = Path(spec["checkpoint_dir"])
    validate_candidate(run, config, hashlib.sha256(config_bytes).hexdigest(), checkpoint_dir)
    if config["model"]["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("configured MPS device is unavailable")
    checkpoint_manifest_sha256 = verify_checkpoint(Path(spec["checkpoint_sha256_file"]), checkpoint_dir)
    problems = load_pinned_test(
        config["data"], expected_size=spec["expected_test_size"],
        expected_parquet_sha256=spec["expected_test_parquet_sha256"],
    )
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
    torch.manual_seed(config["seed"])
    manifest = {
        "source_split": "test",
        "dataset_id": config["data"]["id"],
        "dataset_revision": config["data"]["revision"],
        "test_parquet_sha256": spec["expected_test_parquet_sha256"],
        "question_count": len(problems),
        "run_config_sha256": run["config_sha256"],
        "source_run_summary_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "initial_model_revision": config["model"]["revision"],
        "tokenizer_revision": config["model"]["tokenizer_revision"],
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_policy_version": run["steps"][-1]["policy_after"],
        "checkpoint_sha256_manifest_sha256": checkpoint_manifest_sha256,
        "selection_reason": spec["selection_reason"],
        "evaluation": config["evaluation"],
    }
    print(f"Evaluating {len(problems)} official test questions with {checkpoint_dir}", flush=True)
    return evaluate_to_artifacts(model, tokenizer, problems, config, output_dir, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/m0-final-test.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m0/official-test-v1"))
    args = parser.parse_args()
    result = main(args.spec, args.output_dir)
    print(f"Official test: {result['metrics']['correct_count']}/{result['metrics']['question_count']}")
