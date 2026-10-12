# M4 Paired Confirmation Implementation Plan

> **For agentic workers:** Native implementation in this task; the user already authorized the next experiment. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate M0 and M3 on one new 400-question fixed sample and publish a paired result with provenance.

**Architecture:** A preparation step validates run lineage and freezes question IDs before inference. A run step verifies both checkpoints, reuses `src.evaluation.evaluate_model`, saves each arm's responses, and derives paired counts. A read-only audit then reviews strict flips and labels any last-number metric as a proxy.

**Tech Stack:** Python 3.12, PyTorch MPS, pinned Transformers and GSM8K cache, unittest.

**Spec:** `docs/m4-paired-confirm-v1.md` and `configs/m4-paired-confirm-v1.yaml`.

## Global Constraints

- No optimizer update or official-test evaluation.
- Both arms use identical prompt, tokenizer, greedy generation, cap, and strict verifier.
- Exclude all prior training-candidate, diagnostic, and validation IDs before freezing the 400-question manifest.
- Preserve raw per-question output and refuse to summarize incomplete or misaligned arms.

## Review Focus

- Changed M0/M3 lineage or config: reject before generation.
- Missing or altered checkpoint files: reject before generation.
- Tampered or overlapping question manifest: reject before generation.
- Interrupted run: preserve completed rows and resume only a verified prefix.
- Saved reward mismatch or unequal arm IDs: reject paired summary.

---

### Task 1: Freeze the paired protocol

**Files:** Create `scripts/evaluate_m4_paired.py`, `tests/test_evaluate_m4_paired.py`, `results/m4/paired-confirm-v1/manifest.json`.

- [ ] Test deterministic hash-ranked selection and all exclusions.
- [ ] Test run/config/checkpoint lineage rejection.
- [ ] Implement preparation; save a manifest with source and input hashes before any generation.
- [ ] Run targeted tests and commit the frozen protocol and manifest.

### Task 2: Evaluate and compare

**Files:** Modify `scripts/evaluate_m4_paired.py`, `tests/test_evaluate_m4_paired.py`; create M0/M3 JSONL rows and `summary.json`.

- [ ] Test verified-prefix resume, changed reward rejection, and four-way paired counts.
- [ ] Evaluate each checkpoint with `evaluate_model`; write and flush rows incrementally.
- [ ] Independently recheck all saved rewards and counts, then write summary only when both arms complete.
- [ ] Run full unit tests and compile check.

### Task 3: Explain evidence

**Files:** Modify `docs/m4-paired-confirm-v1.md`, `README.md`; create any flip annotations if needed.

- [ ] Review all strict flips against raw response text and pinned references.
- [ ] Separate strict score, last-number proxy, manual conclusions, and unknowns.
- [ ] Add a 1–10 causal handoff packet and five comprehension questions.
- [ ] Verify docs, commit, and sync the public repository.
