# Mini GRPO Lab

This repository builds a small, inspectable GRPO training loop for **single-turn GSM8K math questions**. The first milestone (M0) is a reproducible RLVR experiment: sampling, deterministic answer verification, group advantages, clipped loss, updates, and an independent held-out evaluation. It does not demonstrate multi-step agent behavior.

## Current state

The repository now has a pinned data and trajectory contract, a strict outcome verifier, group-relative advantages, a masked clipped GRPO loss, and a two-update end-to-end runner with fixed internal validation. The original Day 1 backward pass used generated-token negative log likelihood **only to check the model gradient path**; the later smoke step below is the first actual GRPO update. The completed two-update run did not improve strict validation accuracy.

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

[`results/m0/loss-smoke.json`](results/m0/loss-smoke.json) records 193 valid response tokens, mean pre-update ratio 1.0, maximum ratio deviation 0, clip fraction 0, approximate KL 0, and policy loss 0. The zero loss follows from this group's four zero advantages. Unit tests use nonzero advantages to verify both clipping directions and gradients.

## Mixed-reward gate before an optimizer step

```bash
uv run python -m scripts.find_trainable_group --config configs/m0.yaml --output-dir results/m0 --max-prompts 64 --order original
```

The scanner uses only the pinned training partition, fixed seed, and original row order. It stops at the first group containing both reward 0 and reward 1, then saves separate rollout, scored, and advantaged JSONL files. A run with no such group writes its scan summary and exits nonzero. The current [original-order scan](results/m0/train-smoke-scan.json) found **zero correct responses among 256 samples** from 64 prompts: 207 missing final answers, 47 invalid final answers, two valid but incorrect answers, and 133 length truncations. Its `zero_std_group_fraction` is 1.0.

To separate question difficulty from formatting, a [64-question shortest-first scan](results/m0/shortest-v1/train-smoke-scan.json) used only question word count and prompt ID for ordering; it also found zero correct responses. Two versioned prompt pilots kept the same model, dataset, and strict verifier: [format-v2](configs/m0-format-v2.yaml) allowed 256 tokens and yielded [0/64 correct](results/m0/format-v2/train-smoke-scan.json); [direct-answer](configs/m0-direct-answer.yaml) requested only the numeric final line and also yielded [0/64 correct](results/m0/direct-answer/train-smoke-scan.json). These overlapping, adaptively chosen training samples are diagnostics, not a held-out accuracy estimate.

A [format-example prompt](configs/m0-format-example.yaml) added one independent `2 + 3 = 5` example to format-v2 while keeping its pinned 0.5B model, GSM8K partition, G=4 sampling at temperature 1 with a 256-token limit, and strict verifier. Its [shortest-first scan](results/m0/format-example/train-smoke-scan.json) selected training prompt `gsm8k/main/train/01214` after seven groups: rewards `[0, 0, 1, 0]`, six zero-variance groups, and one strictly correct answer among 28 sampled responses. This is a selected training diagnostic, not an accuracy estimate. The selected [trajectories](results/m0/format-example/train-smoke-advantaged.jsonl) have advantages approximately `[-0.577, -0.577, 1.732, -0.577]`.

```bash
uv run python -m scripts.find_trainable_group --config configs/m0-format-example.yaml --output-dir results/m0/format-example --max-prompts 8 --order shortest
uv run python -m scripts.train_grpo_smoke --config configs/m0-format-example.yaml --input results/m0/format-example/train-smoke-advantaged.jsonl --scan results/m0/format-example/train-smoke-scan.json --output results/m0/format-example/train-smoke.json --checkpoint-dir results/m0/format-example/checkpoints/step-000001
```

`scripts/train_grpo_smoke.py` guards one full-parameter AdamW update at `1e-6`: it checks the pinned training row, prompt tokens, rewards, advantages, scan/config hash, and pre-update ratio before backpropagating. The [real MPS step](results/m0/format-example/train-smoke.json) used 141 valid response tokens, had a pre-update maximum ratio deviation of `3.33e-5`, nonzero pre-clip gradient norm `30.90`, and saved a local model checkpoint under Git-ignored `results/m0/format-example/checkpoints/step-000001/`. Post-update `approx_kl_old` was `0.1914` and clip fraction `0.2199`; this drift is substantial for one step, so the `1e-6` learning rate was only a smoke setting. The checkpoint contains model and tokenizer weights but no optimizer state. CPU tests with a tiny model verify parameter changes and aborts for stale old probabilities, zero reward variance, or zero gradient. That isolated smoke did not include held-out evaluation or repeated rollout/update cycles; the end-to-end run below does.

## Run the end-to-end internal experiment

```bash
uv run python -m scripts.run_m0 --config configs/m0-run.yaml --output-dir results/m0/my-new-run
```

Use an unused output directory: the runner refuses to overwrite prior results. It loads the pinned GSM8K training parquet and records the exact 7,273/200 train/validation split, loads the pinned 0.5B model once, and evaluates all 200 validation questions with greedy decoding and the unchanged strict verifier. With `configs/m0-run.yaml` (`update_epochs: 1`), each requested update searches at most 64 shortest-first **training** questions for mixed rewards, saves separate rollout/scored/advantaged JSONL, checks the pre-update ratio, updates the current model once, and saves a local checkpoint. The next search samples fresh responses from the newly updated policy. A final evaluation uses the same validation IDs, prompt, verifier, and 128-token limit. If no mixed group is found or post-update drift exceeds the configured guard, the summary reports an incomplete run rather than claiming all steps occurred. The official GSM8K test split is untouched.

The recorded [MPS run](results/m0/end-to-end-v2/summary.json) completed **2/2** fresh updates at learning rate `1e-7`, each on a different training question with rewards `[0, 0, 1, 0]`. Post-update `approx_kl_old` was `0.00393` then `0.00101`, below this run's explicit `0.05` continuation guard; this guard is an experiment choice, not a general stability guarantee. Strict internal validation was **9/200 (4.5%) both before and after**. Two questions became correct and two became incorrect; 54 response texts changed. This is a negative result for accuracy improvement at two steps, not evidence that a longer run would or would not improve. The [dataset manifest](results/m0/end-to-end-v2/dataset-split.json), per-step scans/trajectories, [initial](results/m0/end-to-end-v2/validation-initial.json) and [final](results/m0/end-to-end-v2/validation-final.json) counts are small evidence files; the two approximately 1.9 GB model checkpoints remain local and Git-ignored. Optimizer state is not saved, so this run is reproducible from the beginning but cannot resume mid-run from a checkpoint. Both rollout sampling and validation pass `use_model_defaults=False` to Transformers: the original upstream model config did not override these settings in memory, but a saved checkpoint's newer config could override them on reload. The final checkpoint reproduced its first two saved validation rows twice after reload on MPS.

The model weights are intentionally not published. From the repository root, `shasum -a 256 -c results/m0/end-to-end-v2/checkpoint-sha256.txt` verifies all 12 files of the retained final checkpoint, including both weight shards; the [checksum list](results/m0/end-to-end-v2/checkpoint-sha256.txt) is public. If those local files are unavailable, run the end-to-end command above with a fresh output directory to generate a new checkpoint and compare its validation metrics. A fresh MPS run is not guaranteed to produce byte-identical weights, so the published checksums identify the original local artifact rather than serving as an expected hash for a rerun.

## Same-rollout two-epoch experiment

The original M0 plan also specified two optimizer updates on **one** rollout (`mu=2`). [`configs/m0-mu2.yaml`](configs/m0-mu2.yaml) runs that schedule from the pinned initial model. The first update checks `ratio≈1`; the second recomputes current-policy log probabilities against the **same saved** `old_log_probs` and advantages, so its pre-update ratio need not be one. Both updates use the same AdamW state. The step records include `rollout_policy_version`, `update_epoch`, `data_age_updates`, and the path to the one saved group.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.run_m0 --config configs/m0-mu2.yaml --output-dir results/m0/my-mu2-run
```

The [completed MPS run](results/m0/end-to-end-mu2-v1/summary.json) made **2/2** updates on training question `gsm8k/main/train/01214`, reusing exactly the same four trajectories. Its baseline and first selected group match the earlier run byte-for-byte. After update 1, `approx_kl_old=0.00393` and clip fraction `0.0355`; before update 2, clip fraction is already `0.0355`, then rises to `0.0851` after that update (`approx_kl_old=0.01062`). Fixed internal validation changed from **9/200 to 8/200**: one question became correct and two became incorrect. This single-group, two-update result shows no held-out improvement and does not establish that either update schedule is generally better. [Experiment notes](docs/m0-mu2-experiment.md) link the per-step evidence. The official test split remains unused.

Each approximately 1.9 GB checkpoint stays local and Git-ignored. The [final checkpoint checksum list](results/m0/end-to-end-mu2-v1/checkpoint-sha256.txt) identifies the retained weights; run `shasum -a 256 -c results/m0/end-to-end-mu2-v1/checkpoint-sha256.txt` from the repository root to check them. Fresh MPS weights may differ byte-for-byte. `HF_HUB_OFFLINE=1` uses the already cached pinned model and dataset; without that cache, first fetch the exact configured revisions.

## Follow-up failure diagnostic

The [read-only diagnostic](docs/m0-failure-diagnostic-v1.md) uses the retained final checkpoint and a deterministic 100-question sample from the **training** partition, excluding the eight prompts scanned during the two-update run. It keeps greedy decoding, prompt, model, and strict verifier fixed. It evaluates all 100 with the original 128-token cap, then reruns only the 33 length-limited responses with a 256-token cap. The official test and fixed 200-question validation partitions are untouched; this training-partition sample is diagnostic, not a held-out accuracy estimate.

```bash
.venv/bin/python -m scripts.diagnose_m0_failures --output-dir results/m0/my-failure-diagnostic
```

The recorded [summary](results/m0/failure-diagnostic-v1/summary.json) has **6/100 strict correct**, 83 missing final-answer lines, 10 valid-but-incorrect final answers, one invalid final answer, and 33 length-limited responses at 128 tokens. These reason and length counts overlap. None of the 33 length-limited responses became strictly correct with 256 tokens; 18 still hit the longer cap. All 33 decoded short responses were prefixes of their longer counterparts. Increasing the cap alone did not recover an answer in this sample. Completed responses without `####` can contain either correct or incorrect arithmetic, so their mathematical accuracy remains unmeasured by this strict diagnostic. The [manifest](results/m0/failure-diagnostic-v1/manifest.json) and both JSONL files retain the selected IDs and per-question evidence.

The follow-up [answer-value audit](docs/m0-answer-audit-v1.md) examines the 51 responses that ended normally yet lacked a final `####` line. A reference-free annotation pass identified an explicit answer number in 47; [comparison with pinned answers](results/m0/answer-audit-v1/summary.json) found 15 matching numbers, 32 different numbers, and 4 responses without a reliable candidate. This is answer-value agreement on previously selected training-partition diagnostic questions, not a new held-out accuracy or proof that the reasoning was correct. The [per-question audit](results/m0/answer-audit-v1/audit.jsonl) and [annotations](results/m0/answer-audit-v1/annotations.jsonl) allow review. Recompute with `HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.audit_m0_answers --output-dir results/m0/my-answer-audit`.

## Prompt quality comparison after M0

A [fixed-checkpoint prompt experiment](docs/m1-prompt-two-step-v1.md) replaced the simple `2 + 3` example with one two-step arithmetic example while keeping the M0 final weights, tokenizer, greedy decoding, 128-token cap, and strict verifier fixed. It excluded 226 previously scanned or audited training IDs, selected 100 development questions and reserved 200 different questions before generation. Development strict correct was **3/100 → 7/100**, but the one-time reserved confirmation was **13/200 → 11/200** (10 newly correct, 12 lost correct). The variant reduced missing-final-line and truncated responses on the reserve, but did not improve strict answer accuracy there. Neither the original 200-question M0 validation nor the official test was used for this prompt selection. [Specs](configs/m1-prompt-two-step.yaml), [development rows](results/m1/prompt-two-step-v1/summary.json), and [reserved rows](results/m1/prompt-two-step-confirm-v1/summary.json) retain the evidence. The locally retained M0 checkpoint is needed to rerun these exact inference comparisons.

The [paired error analysis](docs/m1-prompt-error-analysis-v1.md) rechecks all 200 reserved pairs, records a clearly labeled last-number proxy for every answer, and manually annotates the failed response in every strict-correct flip. Among 10 newly correct versus 12 lost correct, correct-number/invalid-format cases balance at 4 versus 4; the remaining flips involve wrong answer values (5 versus 6) and unresolved truncation (1 versus 2). This explains the net -2 strict count without claiming the mathematical accuracy of the other 178 pairs. Recompute from the saved JSONL and pinned GSM8K train answers with `HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.analyze_m1_prompt_errors --output-dir results/m1/my-prompt-error-audit`.

## Final report and official test

The [M0 final report](docs/m0-final-report.md) connects the implementation contract, both two-update schedules, fixed validation, failure analysis, raw evidence, and limits. Before opening the official GSM8K test split, it selected the two-fresh-rollout final checkpoint on internal validation (9/200 versus 8/200 for same-rollout `mu=2`). One greedy, strict-verifier evaluation of that trained checkpoint scored **40/1,319 (3.03%)** on the official test. [Per-question rows](results/m0/official-test-v1/test-rows.jsonl), [summary](results/m0/official-test-v1/summary.json), [pinned test specification](configs/m0-final-test.yaml), and [selection/checkpoint manifest](results/m0/official-test-v1/manifest.json) are public. The initial model was not run on the official test, so this result is not a test-set estimate of training improvement. Checkpoint weights remain local; the report distinguishes exact artifact verification from rerunning the procedure.

## M0 acceptance contract

- Persist complete groups of four trajectories with prompt and response token IDs, response masks, pre-update raw-logit `old_log_probs`, outcome rewards, advantages, and policy versions.
- Verify that a newly computed policy has ratio near one before its first optimizer step; test both clipping sides, all-equal rewards, and prompt/padding masks.
- Run multiple fresh rollout/update cycles and separately check two optimizer epochs on one saved rollout. Record zero-variance groups, parser failures, response lengths, loss, clip fraction, and `approx_kl_old`.
- Evaluate the initial and final models with the same greedy decoding on a fixed internal validation split. Reserve the official test split for one final report run. Report raw correct counts, pass@1, response lengths, and failures, including a negative result if training does not improve held-out accuracy.

The source plan is [`phase1-mini-grpo.docx`](phase1-mini-grpo.docx). [`docs/grpo-dataflow.md`](docs/grpo-dataflow.md) describes the representation and ownership of each artifact. No performance or accuracy claim is made until the training and evaluation artifacts exist.
