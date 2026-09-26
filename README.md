# Mini GRPO Lab

This repository builds a small, inspectable GRPO training loop for **single-turn GSM8K math questions**. The first milestone (M0) is a reproducible RLVR experiment: sampling, deterministic answer verification, group advantages, clipped loss, updates, and an independent held-out evaluation. It does not demonstrate multi-step agent behavior.

## Current state

Day 1 establishes the pinned experiment contract, a validated JSONL trajectory format, a data-flow map, and a two-prompt resource smoke test. The smoke backward pass uses generated-token negative log likelihood **only to check the model gradient path**; it is not a GRPO update or a training result.

The chosen baseline is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on Apple Silicon MPS in float32. The model, tokenizer, and [GSM8K](https://huggingface.co/datasets/openai/gsm8k) revisions are pinned in [`configs/m0.yaml`](configs/m0.yaml). Local hardware checked on 2026-09-25: Apple M5 Pro, 64 GB unified memory. Full-parameter training is the initial resource assumption; the Day 1 smoke checks only generation and backward feasibility at a short sequence length.

## Run Day 1 checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
uv sync
uv run python scripts/smoke_model.py --config configs/m0.yaml --output results/m0/day1-smoke.json
```

The model smoke fetches the pinned ~1 GB model on first use. Its local test overrides only the generation length to 16 tokens; the M0 experiment default remains 128. It generates four samples for each of two fixed arithmetic prompts, then computes one backward pass on one response from each prompt. A nonzero exit and a JSON record with `status: failed` mean this check did not pass. The project can run the schema tests without any third-party dependency.

## Prepare the actual GSM8K train/validation data

```bash
uv sync
uv run python -m scripts.prepare_data --config configs/m0.yaml --output results/m0/dataset-split.json
```

This downloads only the pinned GSM8K `main/train` parquet file. The library's whole-repository split check is disabled because it also expects `test`; the loader instead verifies 7,473 source rows and the canonical row-content SHA-256 in the config. It then selects 200 original row positions by SHA-256 rank of `seed:index` with seed 42. The remaining 7,273 rows are the training partition. Original row positions become stable prompt IDs; the two partitions retain source order. [`results/m0/dataset-split.json`](results/m0/dataset-split.json) records the selected validation IDs, source hash, counts, and two prompt previews. The preview contains the question and system instruction, while the reference answer stays available only for later verification. The official GSM8K test split is not loaded by this command.

The next implementation step is to generate `Trajectory` groups from these real prompts and recompute pre-update old-policy token log probabilities. Dataset preparation alone does not create rollouts or train the model.

## M0 acceptance contract

- Persist complete groups of four trajectories with prompt and response token IDs, response masks, pre-update raw-logit `old_log_probs`, outcome rewards, advantages, and policy versions.
- Verify that a newly computed policy has ratio near one before its first optimizer step; test both clipping sides, all-equal rewards, and prompt/padding masks.
- Run multiple fresh rollout/update cycles, with one update for the initial smoke and two updates for the measured run. Record zero-variance groups, parser failures, response lengths, loss, clip fraction, and `approx_kl_old`.
- Evaluate the initial and final models with the same greedy decoding on a fixed internal validation split. Reserve the official test split for one final report run. Report raw correct counts, pass@1, response lengths, and failures, including a negative result if training does not improve held-out accuracy.

The source plan is [`phase1-mini-grpo.docx`](phase1-mini-grpo.docx). [`docs/grpo-dataflow.md`](docs/grpo-dataflow.md) describes the representation and ownership of each artifact. No performance or accuracy claim is made until the training and evaluation artifacts exist.
