# GSM8K Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one complete four-response GSM8K training group with pre-update raw-logit token log probabilities and persist it as validated JSONL.

**Architecture:** Keep extraction and log-probability alignment in `src/rollout/hf_rollout.py`, and keep model/dataset loading in `scripts/sample_rollout.py`. A generated trajectory stores unpadded response IDs, so every stored response mask entry is one; EOS is included before decoding and padding is excluded. The sampling policy stays in eval mode and does not update weights.

**Tech Stack:** Python 3.12, locked PyTorch 2.14.0, Transformers 4.57.6, existing GSM8K loader and `Trajectory` schema.

**Spec:** `phase1-mini-grpo.docx`, sections 3.1–3.3, 4, Day 2 and the first-commit boundary.

## Global Constraints

- Use the pinned model, tokenizer, GSM8K train revision, and split in `configs/m0.yaml`.
- Generate G=4 responses per prompt with sampling, temperature=1, top_p=1, top_k=0, max_new_tokens=128, tokenizer EOS, and no other logits processors; override the model's inherited repetition_penalty=1.1 with 1.0.
- Compute old log probabilities with a no-grad raw model forward before any optimizer step; do not use processed `generate()` scores.
- Disable dropout and persist matching policy versions across the four responses.
- This slice leaves reward and advantage null; a deterministic verifier follows later.

## Review Focus

- Logits at `P-1` must score the first generated token, and logits at `P+T-2` the last one; Task 1 tests this with hand-set logits.
- EOS is counted once as a generated token and later padding is excluded; Task 1 tests EOS and length termination.
- Empty responses or invalid EOS IDs must fail before persistence; Task 1 tests them.
- All four records must share prompt IDs, prompt token IDs and policy version; Task 2 validates the group.
- The model's inherited repetition penalty must not affect sampling; Task 2 asserts the explicit generation setting and reruns the Day 1 smoke.

---

### Task 1: Token extraction and raw policy probabilities

**Files:** Create `src/rollout/hf_rollout.py`, `src/rollout/__init__.py`, `tests/test_rollout.py`.

**Interfaces:** `trim_generated_response(tokens: list[int], eos_token_id: int) -> tuple[list[int], str]`; `gather_response_log_probs(logits: torch.Tensor, response_ids: torch.Tensor, prompt_length: int) -> torch.Tensor`; `compute_old_log_probs(model, prompt_ids: list[int], response_ids: list[int], device: str) -> list[float]`.

- [x] Write failing tests for EOS/pad trimming, length termination, empty response rejection, a two-token hand-computed log-probability alignment, and no-grad model forward.
- [x] Run `.venv/bin/python -m unittest tests.test_rollout -v`; confirm failures identify missing interfaces.
- [x] Implement those three functions in `src/rollout/hf_rollout.py` with a raw forward, `log_softmax`, and token gather.
- [x] Run `.venv/bin/python -m unittest tests.test_rollout -v`; confirm all cases pass.

### Task 2: Full group and local smoke artifact

**Files:** Update `src/rollout/hf_rollout.py`, `scripts/smoke_model.py`, `README.md`, `docs/grpo-dataflow.md`; create `scripts/sample_rollout.py` and `results/m0/rollout-sample.jsonl`.

**Interfaces:** `sample_group(model, tokenizer, problem: MathProblem, config: dict, policy_version: str) -> list[Trajectory]`; `python -m scripts.sample_rollout --config configs/m0.yaml --output results/m0/rollout-sample.jsonl`.

- [x] Write a fake-model test that produces four sequences, asserts the exact generation arguments including `repetition_penalty=1.0`, then checks a validated four-record group and JSONL round trip.
- [x] Implement `sample_group` and a CLI that loads the first pinned training problem and pinned model/tokenizer, samples one group, writes JSONL and reads it back.
- [x] Add `repetition_penalty=1.0` to the Day 1 smoke call and rerun it to update its evidence log.
- [x] Run the real rollout CLI on MPS with the pinned model; inspect four records, log-probability lengths, EOS/length labels, and policy version.
- [x] Run all tests, compile checks, and a locked dependency check; document the exact result and commit this slice.
