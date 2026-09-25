# Mini GRPO Day 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the reproducible M0 experiment contract, trajectory persistence, and a two-prompt model resource smoke test.

**Architecture:** Keep persistent trajectories as validated JSON native records in `src/data/schema.py`; model tensors exist only inside the smoke script. Pin external revisions and sampling settings in one YAML configuration, then document each producer, consumer, shape, and storage boundary.

**Tech Stack:** Python 3.12, standard library for schema tests, PyYAML, PyTorch, Transformers, pytest.

**Spec:** `phase1-mini-grpo.docx`, sections 3–5 and 10.

## Global Constraints

- M0 is single-turn GSM8K RLVR; agentic tool actions are outside this milestone.
- Dataset train/internal validation split uses seed 42 and 200 validation examples; official test is reserved for the final report.
- Training generation defaults to G=4, temperature=1, top_p=1, top_k=0, max_new_tokens=128, without other logits processors.
- Old log probabilities come from raw policy logits before any optimizer step; dropout is disabled.
- GRPO uses two update epochs, epsilon=0.2, beta=0; no trainer is part of Day 1.

## Review Focus

- A JSONL record with a mismatched response mask or log-probability length must fail validation in Task 1.
- A group with mixed policy versions or duplicate sample indices must fail validation in Task 1.
- NaN or infinite reward, advantage, or log probability must fail validation in Task 1.
- A truncated JSONL final line must fail with line context in Task 1.
- A missing model dependency or unavailable model revision must produce an explicit smoke failure in Task 2.

---

### Task 1: Configuration and trajectory contract

**Files:** Create `configs/m0.yaml`, `src/data/schema.py`, `tests/test_schema.py`, `docs/grpo-dataflow.md`, `README.md`.

**Interfaces:** `Trajectory.from_dict(dict) -> Trajectory`, `validate_group(list[Trajectory]) -> None`, `write_jsonl(path, Iterable[Trajectory]) -> None`, `read_jsonl(path) -> list[Trajectory]`.

- [x] Write tests for a two-row JSONL round trip, mask and log-probability mismatch, non-finite values, mixed policy versions, duplicate sample indices, and truncated JSONL.
- [x] Run `python3 -m unittest discover -s tests -v`; verify schema tests fail before implementation.
- [x] Implement the smallest dataclass and JSONL functions that satisfy those tests, including exact type checks for token IDs and masks.
- [x] Run the same command and inspect the result.
- [x] Document the `[B,G,P,T]` data flow, CPU/disk versus training-device ownership, and the M0 evaluation boundary.

### Task 2: Resource smoke test

**Files:** Create `scripts/smoke_model.py`, `results/m0/day1-smoke.json`; update `README.md` and `pyproject.toml`.

**Interfaces:** `python scripts/smoke_model.py --config configs/m0.yaml --output results/m0/day1-smoke.json` generates two prompts with G=4, takes one backward step on generated-token likelihood, and records pinned revisions, device, memory, timings, and success/failure.

- [x] Implement config loading and explicit dependency/load errors with a nonzero exit code.
- [x] Load the pinned model and tokenizer, sample two arithmetic prompts, compute a generated-token negative log-likelihood for one backward step, and write a JSON result.
- [x] Run the smoke command on the local Mac; if the model or dependency cannot be fetched, record the exact blocking stage without claiming the model smoke passed.
- [x] Run `python3 -m unittest discover -s tests -v` and `python3 -m compileall -q src scripts tests` again after the script is added.
