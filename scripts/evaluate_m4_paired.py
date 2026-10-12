"""Freeze new GSM8K questions, then evaluate retained M0 and M3 policies in pairs."""

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

from scripts.analyze_m1_prompt_errors import value_proxy
from scripts.compare_prompt_quality import collect_prior_ids, compare_paired, summarize
from scripts.evaluate_official_test import verify_checkpoint
from src.data.dataset import MathProblem, load_pinned_train
from src.evaluation import evaluate_model
from src.reward.math_verifier import parse_final_answer, verify_answer


def _read_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def select_new_questions(
    training: Sequence[MathProblem], excluded: set[str], salt: str, count: int,
) -> tuple[MathProblem, ...]:
    """Hash-rank never-used training-partition IDs independently of answer text."""
    ids = {item.prompt_id for item in training}
    if (len(ids) != len(training) or not excluded <= ids
            or not isinstance(salt, str) or not salt
            or type(count) is not int or count < 1):
        raise ValueError("invalid M4 question-selection inputs")
    available = [item for item in training if item.prompt_id not in excluded]
    if len(available) < count:
        raise ValueError("not enough unused questions for M4")
    return tuple(sorted(available, key=lambda item: (
        hashlib.sha256(f"{salt}:{item.prompt_id}".encode()).digest(), item.prompt_id,
    ))[:count])


def validate_lineage(
    m0_config: dict, m3_config: dict, m0_run: dict, m3_run: dict,
    m3_manifest: dict, m0_config_sha: str, m3_config_sha: str,
    m0_summary_sha: str, source_sha: str,
) -> tuple[Path, Path]:
    """Require completed, connected policies and identical inference contracts."""
    m0_checkpoint = Path(m3_config["model"]["revision"])
    m3_checkpoint = Path(m3_run["steps"][-1]["checkpoint_dir"])
    if (m0_run["status"] != "completed" or m0_run["completed_steps"] != 2
            or m3_run["status"] != "completed" or m3_run["completed_steps"] != 10
            or m0_run["config_sha256"] != m0_config_sha
            or m3_run["config_sha256"] != m3_config_sha
            or m3_manifest["config_sha256"] != m3_config_sha
            or any(item["source_sha256"] != source_sha
                   for item in (m0_run, m3_run, m3_manifest))
            or m3_manifest["source_run_summary_sha256"] != m0_summary_sha
            or Path(m3_manifest["starting_checkpoint"]).resolve() != m0_checkpoint.resolve()
            or Path(m3_run["initial_model_revision"]).resolve() != m0_checkpoint.resolve()
            or Path(m0_run["steps"][-1]["checkpoint_dir"]).resolve() != m0_checkpoint.resolve()
            or m3_checkpoint.resolve() == m0_checkpoint.resolve()
            or any(m0_config[key] != m3_config[key] for key in ("data", "prompt"))
            or any(m0_config["evaluation"][key] != m3_config["evaluation"][key]
                   for key in ("do_sample", "max_new_tokens"))
            or m3_config["evaluation"].get("run_official_test_once") is not False
            or any(m0_config["model"][key] != m3_config["model"][key]
                   for key in ("id", "tokenizer_revision", "device", "dtype"))
            or m3_config["evaluation"]["do_sample"] is not False
            or m3_config["evaluation"]["max_new_tokens"] != 128):
        raise ValueError("M0 and M3 lineage or fixed evaluation contract differs")
    return m0_checkpoint, m3_checkpoint


def build_protocol(spec_path: Path) -> tuple[tuple[MathProblem, ...], dict, dict, tuple[Path, Path]]:
    spec_bytes = spec_path.read_bytes()
    spec = yaml.safe_load(spec_bytes)
    if (spec["experiment"] != "m4-paired-confirm-v1"
            or spec["selection"] != {"salt": "m4-paired-confirm-v1", "question_count": 400}
            or spec["evaluation"]["primary_metric"] != "strict_correct_count"):
        raise ValueError("M4 fixed protocol has changed")
    m3_config_path, m3_dir = Path(spec["m3_config"]), Path(spec["m3_run_dir"])
    m3_config_bytes = m3_config_path.read_bytes()
    m3_config = yaml.safe_load(m3_config_bytes)
    m3_config_sha = hashlib.sha256(m3_config_bytes).hexdigest()
    m0_config_path = Path(m3_config["warm_start"]["source_run_config"])
    m0_config_bytes = m0_config_path.read_bytes()
    m0_config = yaml.safe_load(m0_config_bytes)
    m0_config_sha = hashlib.sha256(m0_config_bytes).hexdigest()
    m0_summary_path = Path(m3_config["warm_start"]["source_run_summary"])
    m0_run, m0_summary_sha = _read_json(m0_summary_path)
    m3_run, m3_summary_sha = _read_json(m3_dir / "summary.json")
    m3_manifest, m3_manifest_sha = _read_json(m3_dir / "dataset-split.json")
    split = load_pinned_train(m3_config["data"])
    checkpoints = validate_lineage(
        m0_config, m3_config, m0_run, m3_run, m3_manifest,
        m0_config_sha, m3_config_sha, m0_summary_sha, split.source_sha256,
    )
    if (m3_run["training_count"] != len(m3_manifest["training_prompt_ids"])
            or m3_run["validation_count"] != len(m3_manifest["validation_prompt_ids"])):
        raise ValueError("M3 manifest question counts disagree with its run")

    warm = m3_config["warm_start"]
    m1_spec_path = Path(warm["prior_prompt_spec"])
    m1_spec_bytes = m1_spec_path.read_bytes()
    prior, prior_hashes = collect_prior_ids(yaml.safe_load(m1_spec_bytes), split.source_sha256)
    prior_hashes[str(m1_spec_path)] = hashlib.sha256(m1_spec_bytes).hexdigest()
    m1_path, m2_path = Path(warm["prior_m1_manifest"]), Path(warm["prior_m2_manifest"])
    m1, prior_hashes[str(m1_path)] = _read_json(m1_path)
    m2, prior_hashes[str(m2_path)] = _read_json(m2_path)
    if (prior_hashes != m3_manifest["prior_evidence_sha256"]
            or m1["source_sha256"] != split.source_sha256
            or m2["source_sha256"] != split.source_sha256
            or m1["spec_sha256"] != prior_hashes[str(m1_spec_path)]
            or set(m2["question_ids"]) != {"shortest", "hash_random"}):
        raise ValueError("M3 prior-question evidence has changed")
    prior.update(m1["development_prompt_ids"] + m1["reserved_prompt_ids"])
    prior.update(m2["question_ids"]["shortest"] + m2["question_ids"]["hash_random"])
    if len(prior) != m3_manifest["excluded_prior_prompt_count"]:
        raise ValueError("M3 prior-question exclusion count has changed")
    m3_pool = m3_manifest["training_prompt_ids"]
    m3_validation = m3_manifest["validation_prompt_ids"]
    training_ids = {item.prompt_id for item in split.training}
    if (len(m3_pool) != len(set(m3_pool)) or len(m3_validation) != len(set(m3_validation))
            or set(m3_pool) & set(m3_validation)
            or not set(m3_pool + m3_validation + list(prior)) <= training_ids
            or set(m3_pool + m3_validation) & prior):
        raise ValueError("M3 or earlier question IDs overlap or leave the train partition")
    excluded = prior | set(m3_pool) | set(m3_validation)
    selection = spec["selection"]
    questions = select_new_questions(
        split.training, excluded, selection["salt"], selection["question_count"],
    )
    m0_checkpoint_hash = verify_checkpoint(
        Path(warm["checkpoint_sha256_file"]), checkpoints[0],
    )
    m3_checkpoint_hash = verify_checkpoint(
        Path(spec["m3_checkpoint_sha256_file"]), checkpoints[1],
    )
    manifest = {
        "experiment": spec["experiment"], "source_split": "train",
        "source_sha256": split.source_sha256,
        "dataset_revision": m3_config["data"]["revision"],
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "m0_config_sha256": m0_config_sha,
        "m3_config_sha256": m3_config_sha,
        "m0_summary_sha256": m0_summary_sha,
        "m3_summary_sha256": m3_summary_sha,
        "m3_dataset_manifest_sha256": m3_manifest_sha,
        "m0_checkpoint": str(checkpoints[0]),
        "m3_checkpoint": str(checkpoints[1]),
        "m0_checkpoint_sha256_manifest_sha256": m0_checkpoint_hash,
        "m3_checkpoint_sha256_manifest_sha256": m3_checkpoint_hash,
        "prior_evidence_sha256": prior_hashes,
        "excluded_prompt_count": len(excluded),
        "excluded_prompt_ids": sorted(excluded),
        "selection_salt": selection["salt"],
        "question_count": len(questions),
        "question_prompt_ids": [item.prompt_id for item in questions],
        "model_id": m3_config["model"]["id"],
        "tokenizer_revision": m3_config["model"]["tokenizer_revision"],
        "prompt_system": m3_config["prompt"]["system"],
        "evaluation": m3_config["evaluation"],
        "model_updated": False, "official_test_used": False,
    }
    return questions, manifest, m3_config, checkpoints


def load_verified_prefix(path: Path, questions: Sequence[MathProblem]) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= len(questions):
                raise ValueError(f"too many saved rows: {path}")
            row = json.loads(line)
            checked = verify_answer(row["response_text"], questions[index].reference_answer)
            if (row["prompt_id"] != questions[index].prompt_id
                    or (row["reward"], row["reward_reason"]) != (checked.reward, checked.reason)
                    or row["finish_reason"] not in {"eos", "length"}
                    or type(row["response_tokens"]) is not int or row["response_tokens"] < 1):
                raise ValueError(f"saved evaluation prefix is invalid at row {index + 1}: {path}")
            rows.append(row)
    return rows


def summarize_paired(
    m0_rows: Sequence[dict], m3_rows: Sequence[dict], questions: Sequence[MathProblem],
) -> tuple[list[dict], dict]:
    ids = [item.prompt_id for item in questions]
    if (not ids or len(set(ids)) != len(ids)
            or [row["prompt_id"] for row in m0_rows] != ids
            or [row["prompt_id"] for row in m3_rows] != ids):
        raise ValueError("both evaluation arms must fully match the fixed question order")
    changes = Counter()
    transitions = Counter()
    proxies = {"m0": Counter(), "m3": Counter()}
    paired = []
    for problem, a, b in zip(questions, m0_rows, m3_rows):
        for row in (a, b):
            checked = verify_answer(row["response_text"], problem.reference_answer)
            if (row["reward"], row["reward_reason"]) != (checked.reward, checked.reason):
                raise ValueError(f"saved reward disagrees with the pinned answer: {problem.prompt_id}")
        reference = parse_final_answer(problem.reference_answer)
        if reference is None:
            raise ValueError(f"pinned answer has no final value: {problem.prompt_id}")
        change = (
            "correct_in_both" if a["reward"] == b["reward"] == 1 else
            "newly_correct" if a["reward"] == 0 and b["reward"] == 1 else
            "lost_correct" if a["reward"] == 1 and b["reward"] == 0 else
            "incorrect_in_both"
        )
        changes[change] += 1
        transitions[f"{a['reward_reason']} -> {b['reward_reason']}"] += 1
        a_proxy, b_proxy = value_proxy(a, reference), value_proxy(b, reference)
        proxies["m0"][a_proxy["class"]] += 1
        proxies["m3"][b_proxy["class"]] += 1
        paired.append({
            "prompt_id": problem.prompt_id, "reference": str(reference), "change": change,
            "m0": {"reward_reason": a["reward_reason"], "finish_reason": a["finish_reason"]},
            "m3": {"reward_reason": b["reward_reason"], "finish_reason": b["finish_reason"]},
            "m0_last_number_proxy": a_proxy, "m3_last_number_proxy": b_proxy,
        })
    for name in ("correct_in_both", "newly_correct", "lost_correct", "incorrect_in_both"):
        changes.setdefault(name, 0)
    return paired, {
        "question_count": len(questions),
        "strict_change_counts": dict(sorted(changes.items())),
        "reward_reason_transition_counts": dict(sorted(transitions.items())),
        "last_number_proxy_counts": {
            arm: dict(sorted(counts.items())) for arm, counts in proxies.items()
        },
        "m0": summarize(m0_rows), "m3": summarize(m3_rows),
        "paired": compare_paired(m0_rows, m3_rows),
    }


def prepare(spec_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    _, manifest, _, _ = build_protocol(spec_path)
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _evaluate_arm(
    checkpoint: Path, tokenizer, questions: Sequence[MathProblem], config: dict,
    path: Path, label: str,
) -> list[dict]:
    rows = load_verified_prefix(path, questions)
    if len(rows) == len(questions):
        print(f"{label}: verified {len(rows)}/{len(questions)} saved rows", flush=True)
        return rows
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, trust_remote_code=False,
    ).to(config["model"]["device"])
    model.eval()
    with path.open("a", encoding="utf-8") as output:
        for index in range(len(rows), len(questions)):
            result, _ = evaluate_model(model, tokenizer, (questions[index],), config)
            if len(result) != 1 or result[0]["prompt_id"] != questions[index].prompt_id:
                raise ValueError("evaluation returned a mismatched question")
            rows.append(result[0])
            output.write(json.dumps(result[0], ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            if (index + 1) % 10 == 0 or index + 1 == len(questions):
                print(f"{label}: {index + 1}/{len(questions)}", flush=True)
    del model
    torch.mps.empty_cache()
    return load_verified_prefix(path, questions)


def run(spec_path: Path, output_dir: Path) -> dict:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file() or (output_dir / "summary.json").exists():
        raise ValueError("M4 requires a frozen manifest and no completed summary")
    questions, expected_manifest, config, checkpoints = build_protocol(spec_path)
    saved_manifest = json.loads(manifest_path.read_text())
    if saved_manifest != expected_manifest:
        raise ValueError("frozen M4 manifest disagrees with current inputs")
    if config["model"]["device"] != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("M4 needs the configured MPS device")
    cached = try_to_load_from_cache(
        config["model"]["id"], "tokenizer_config.json",
        revision=config["model"]["tokenizer_revision"],
    )
    if not isinstance(cached, str):
        raise FileNotFoundError("pinned original tokenizer is not cached")
    tokenizer = AutoTokenizer.from_pretrained(Path(cached).parent, trust_remote_code=False)
    torch.manual_seed(config["seed"])
    m0_rows = _evaluate_arm(checkpoints[0], tokenizer, questions, config, output_dir / "m0.jsonl", "M0")
    m3_rows = _evaluate_arm(checkpoints[1], tokenizer, questions, config, output_dir / "m3.jsonl", "M3")
    paired, counts = summarize_paired(m0_rows, m3_rows, questions)
    paired_path = output_dir / "paired.jsonl"
    with paired_path.open("w", encoding="utf-8") as file:
        for row in paired:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "status": "completed", "model_updated": False, "official_test_used": False,
        "source_split": "train", "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("manifest", manifest_path), ("m0_rows", output_dir / "m0.jsonl"),
                ("m3_rows", output_dir / "m3.jsonl"), ("paired", paired_path),
            )
        },
        **counts,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--spec", type=Path, default=Path("configs/m4-paired-confirm-v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m4/paired-confirm-v1"))
    args = parser.parse_args()
    result = prepare(args.spec, args.output_dir) if args.mode == "prepare" else run(args.spec, args.output_dir)
    print(f"M4 {args.mode}: {result.get('question_count', len(result.get('question_prompt_ids', [])))} questions")
