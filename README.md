# Mini GRPO Lab

This repository builds a small, inspectable GRPO training loop for **single-turn GSM8K math questions**. The first milestone (M0) is a reproducible RLVR experiment: sampling, deterministic answer verification, group advantages, clipped loss, updates, and an independent held-out evaluation. It does not demonstrate multi-step agent behavior.

## Current state

Day 1 establishes the pinned experiment contract, a validated JSONL trajectory format, a data-flow map, and a two-prompt resource smoke test. The smoke backward pass uses generated-token negative log likelihood **only to check the model gradient path**; it is not a GRPO update or a training result.

The chosen baseline is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on Apple Silicon MPS in float32. The model, tokenizer, and [GSM8K](https://huggingface.co/datasets/openai/gsm8k) revisions are pinned in [`configs/m0.yaml`](configs/m0.yaml). Local hardware checked on 2026-09-25: Apple M5 Pro, 64 GB unified memory. Full-parameter training is the initial resource assumption; the Day 1 smoke checks only generation and backward feasibility at a short sequence length.

## Run Day 1 checks

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q src scripts tests
uv run python scripts/smoke_model.py --config configs/m0.yaml --output results/m0/day1-smoke.json
```

The model smoke fetches the pinned ~1 GB model on first use. Its local test overrides only the generation length to 16 tokens; the M0 experiment default remains 128. It generates four samples for each of two fixed arithmetic prompts, then computes one backward pass on one response from each prompt. A nonzero exit and a JSON record with `status: failed` mean this check did not pass. The schema-only tests can run without third-party dependencies with `python3 -m unittest tests.test_schema -v`.

## Prepare the actual GSM8K train/validation data

```bash
uv sync
uv run python -m scripts.prepare_data --config configs/m0.yaml --output results/m0/dataset-split.json
```

This loads only the pinned GSM8K `main/train` parquet file, reusing the exact revision's local Hugging Face cache when available. The library's whole-repository split check is disabled on the download path because it also expects `test`; both paths verify 7,473 source rows and the canonical row-content SHA-256 in the config. The loader then selects 200 original row positions by SHA-256 rank of `seed:index` with seed 42. The remaining 7,273 rows are the training partition. Original row positions become stable prompt IDs; the two partitions retain source order. [`results/m0/dataset-split.json`](results/m0/dataset-split.json) records the selected validation IDs, source hash, counts, and two prompt previews. The preview contains the question and system instruction, while the reference answer stays available only for verification. The official GSM8K test split is not loaded by this command.

## Sample a real trajectory group

```bash
uv run python -m scripts.sample_rollout --config configs/m0.yaml --output results/m0/rollout-sample.jsonl
```

The command uses the first fixed training question, samples G=4 responses with at most 128 new tokens each, then runs the unchanged model on each prompt and response to obtain `old_log_probs` from raw logits. It persists four `Trajectory` lines and reads them back for group validation. [`results/m0/rollout-sample.jsonl`](results/m0/rollout-sample.jsonl) is the local evidence. EOS is kept as a valid response token, while padding after EOS is discarded; the stored mask therefore contains one entry per stored response token. The model's own generation settings include repetition_penalty=1.1, so this rollout explicitly uses 1.0 to match the M0 contract. The Day 1 smoke log was rerun with the same override.

The rollout command leaves reward and advantage null. It does not update model weights or measure answer accuracy.

## Score the saved trajectories

```bash
uv run python -m scripts.score_rollout --config configs/m0.yaml --input results/m0/rollout-sample.jsonl --output results/m0/rewarded-sample.jsonl
```

The verifier compares only the last nonblank `#### <number>` line with the pinned reference answer. It accepts signed integers, decimal numbers, and properly grouped thousands separators, comparing numeric values exactly. Missing or malformed final lines receive reward 0 with a reason code; an invalid reference answer stops scoring. The original rollout file is preserved, and [`results/m0/rewarded-sample.jsonl`](results/m0/rewarded-sample.jsonl) records the four scored trajectories. This sample has **four `missing_final_answer` results and four zero rewards**: two responses calculate 72 in prose without the required final line, one is incorrect/incomplete, and one reaches the 128-token limit. This is a single-group diagnostic, not an accuracy estimate.

## Normalize rewards within the group

```bash
uv run python -m scripts.normalize_advantages --config configs/m0.yaml --input results/m0/rewarded-sample.jsonl --output results/m0/advantaged-sample.jsonl
```

For rewards `r_1...r_G`, this uses the population standard deviation and sets `A_i = (r_i - mean(r)) / (std(r) + 1e-8)`. An all-equal group receives exact zero advantages. The command writes a separate [`results/m0/advantaged-sample.jsonl`](results/m0/advantaged-sample.jsonl) and validates that all required trajectory fields are present. The current real group has mean 0, standard deviation 0, and four advantages of 0; it contains no learning signal.

## Check the clipped GRPO loss

```bash
uv run python -m scripts.check_grpo_loss --config configs/m0.yaml --input results/m0/advantaged-sample.jsonl --output results/m0/loss-smoke.json
```

The check loads the pinned model in evaluation mode, recomputes raw current-policy log probabilities for the saved response tokens, and calculates `ratio = exp(new_log_prob - old_log_prob)`. The loss takes the smaller of `ratio * advantage` and its clipped counterpart on valid response tokens, averages tokens **within each response**, then averages responses within the group. Prompt and right-padding tokens are excluded; generated EOS is included. M0 uses `clip_epsilon=0.2` and `beta=0`. The code is differentiable, while this check uses no gradients or optimizer step.

[`results/m0/loss-smoke.json`](results/m0/loss-smoke.json) records 193 valid response tokens, mean pre-update ratio 1.0, maximum ratio deviation 0, clip fraction 0, approximate KL 0, and policy loss 0. The zero loss follows from this group's four zero advantages. Unit tests use nonzero advantages to verify both clipping directions and gradients. The next step is an actual optimizer smoke test with groups that contain reward variation.

## M0 acceptance contract

- Persist complete groups of four trajectories with prompt and response token IDs, response masks, pre-update raw-logit `old_log_probs`, outcome rewards, advantages, and policy versions.
- Verify that a newly computed policy has ratio near one before its first optimizer step; test both clipping sides, all-equal rewards, and prompt/padding masks.
- Run multiple fresh rollout/update cycles, with one update for the initial smoke and two updates for the measured run. Record zero-variance groups, parser failures, response lengths, loss, clip fraction, and `approx_kl_old`.
- Evaluate the initial and final models with the same greedy decoding on a fixed internal validation split. Reserve the official test split for one final report run. Report raw correct counts, pass@1, response lengths, and failures, including a negative result if training does not improve held-out accuracy.

The source plan is [`phase1-mini-grpo.docx`](phase1-mini-grpo.docx). [`docs/grpo-dataflow.md`](docs/grpo-dataflow.md) describes the representation and ownership of each artifact. No performance or accuracy claim is made until the training and evaluation artifacts exist.
