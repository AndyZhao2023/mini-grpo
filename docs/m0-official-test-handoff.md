# M0 official-test handoff

1. **Outcome.** The selected two-fresh-rollout `step-000002` checkpoint scored 40/1,319 under the strict GSM8K verifier. The [final report](m0-final-report.md) connects this one official test run to the preceding internal-validation decision.
2. **Before.** The project had two completed two-update schedules and a fixed 200-question internal validation comparison, but no official test run or final report.
3. **Why.** Supplied: the user requested completion of the M0 final report.
4. **Core intuition.** Choose the candidate using internal validation, freeze that choice and the test contract, verify every input artifact, then generate one answer per official test question. Rechecking saved answers is not another model evaluation.
5. **Flow.** [`m0-final-test.yaml`](../configs/m0-final-test.yaml) names the unchanged run config, completed run, final checkpoint, checksum list, and pinned 1,319-row test parquet hash. [`evaluate_official_test.py`](../scripts/evaluate_official_test.py) checks their agreement, loads the original pinned tokenizer and saved model, calls the existing greedy `evaluate_model()` once over all questions, and writes a manifest, 1,319 JSONL rows, and summary. The report reads those artifacts.
6. **Explained change.** [`dataset.py`](../src/data/dataset.py) gained `load_pinned_test()` to verify the cached exact-revision parquet before constructing answer-separated `MathProblem` objects. The new CLI binds data and model provenance and reuses the existing evaluator and verifier, so no new scoring rule crosses into the report. [`plot_m0_results.py`](../scripts/plot_m0_results.py) draws the two recorded internal-validation comparisons; the report and README link the figure and raw evidence.
7. **Invariants and boundaries.** The test answer never enters `build_prompt_messages()`. Test evaluation uses greedy decoding, 128 new tokens, the unchanged strict `#### <number>` verifier, and no optimizer. Only the candidate selected before test exposure is evaluated; initial-model test accuracy is unknown. The local checkpoint's 12 files must match the published checksum list before generation.
8. **Failure and recovery.** The CLI refuses an existing output directory. If generation fails after creating the manifest, no completed summary exists; inspect the error and use a fresh output directory. The exact reported result requires the retained local checkpoint; a newly rerun model needs a new selection and checksum manifest and is a distinct experiment.
9. **Evidence.** The [official manifest](../results/m0/official-test-v1/manifest.json), [rows](../results/m0/official-test-v1/test-rows.jsonl), and [summary](../results/m0/official-test-v1/summary.json) record the MPS run. Independently checked all 1,319 row IDs against the pinned test parquet, recomputed every reward with the strict verifier, and matched counts, truncations, and mean response length to the summary. The 65-test suite, compile check, chart regeneration, and diff check passed; the chart was visually inspected.
10. **Comprehension check.** Questions and answers follow separately.

## Questions

1. Which evidence fixes the model candidate before official test answers are generated?
2. Why does `load_pinned_test()` retain reference answers if they must not enter the prompt?
3. What would change if the tokenizer were loaded from the saved checkpoint instead of the original pinned snapshot?
4. What files distinguish a completed run from a failed partial run?
5. Why does 40/1,319 not establish a test-set improvement from GRPO training?

## Answers

1. The final-test spec names the earlier completed run and checkpoint and records the selection rule based on fixed internal validation: 9/200 versus 8/200 among trained candidates. The manifest preserves that rule and the source-run/config hashes.
2. `MathProblem.reference_answer` is consumed by the deterministic verifier after generation; `build_prompt_messages()` uses only the question and system instruction.
3. The saved tokenizer could differ in preprocessing behavior. The CLI loads the original pinned tokenizer snapshot used for the training run; generation settings are also explicit in the evaluator.
4. A completed directory has `manifest.json`, `test-rows.jsonl`, and `summary.json` with status `completed`; a failed run may have only the manifest. The CLI requires a new directory for recovery.
5. The initial model was not evaluated on the official test. The 9/200 internal-validation result uses different questions and cannot be subtracted from the official test score.

## Open question

- The original training run did not reliably record throughput, exact wall time, or peak unified memory. Those performance claims require a separate instrumented experiment.
