# M0 failure diagnostic v1

1. **Outcome.** A reproducible, read-only training-partition diagnostic separates strict parser failures from length termination and tests whether more tokens recover truncated answers.
2. **Before.** The two-update run completed, but fixed validation stayed at 9/200 before and after. Its per-question rows showed format failures and truncation; it did not isolate the effect of the generation cap.
3. **Why.** Inferred from the request to move to the next experiment: identify the dominant observable failures before spending more compute on training.
4. **Core intuition.** Hold the final model, prompt, and verifier fixed. Change only the maximum response length on questions that actually reached the original cap.
5. **Flow.** Hash-rank the pinned training partition and select 100 questions after excluding the eight prompts scanned in the two-update run. Greedy-evaluate all 100 at 128 tokens. Rerun the 33 length-limited questions at 256 tokens. Compare matching prompt IDs.
6. **Explained change.** [`scripts/diagnose_m0_failures.py`](../scripts/diagnose_m0_failures.py) reuses the pinned loader and existing `evaluate_model()`; it writes a selection manifest, one JSONL per cap, and a summary. [`tests/test_diagnose_m0_failures.py`](../tests/test_diagnose_m0_failures.py) checks stable selection, overlapping failure dimensions, and exact pairing of truncated reruns. No optimizer is constructed or called.
7. **Invariants and boundaries.** The 100 IDs are disjoint from the 200 validation IDs and from this run's eight scanned training IDs. The official GSM8K test split is not loaded. Both arms use the same final `step-000002` checkpoint, greedy prompt, and strict `#### <number>` verifier. The 256-token arm contains only 128-token truncations.
8. **Failure and recovery.** The command refuses an existing output directory. Interrupted runs may leave partial JSONL but no completed `summary.json`; rerun into a fresh directory. It requires local MPS and the retained checkpoint. Reloading the checkpoint tokenizer emits a Transformers regex warning; its prompt token IDs matched the pinned original tokenizer for all 100 selected questions, but broader tokenizer equivalence was not established.
9. **Evidence.** The 12 checkpoint files passed `shasum -a 256 -c results/m0/end-to-end-v2/checkpoint-sha256.txt`; 57 unit tests passed. The [`manifest`](../results/m0/failure-diagnostic-v1/manifest.json), [128-token rows](../results/m0/failure-diagnostic-v1/evaluation-128.jsonl), [256-token truncated rows](../results/m0/failure-diagnostic-v1/evaluation-256-truncated.jsonl), and [summary](../results/m0/failure-diagnostic-v1/summary.json) are the actual MPS outputs. At 128 tokens: 6/100 strictly correct, 83 missing final-answer lines, 10 valid incorrect, one invalid, and 33 length-limited. Of the 33 extended responses, zero became strictly correct and 18 still hit the 256-token cap; all decoded short texts were prefixes of the long texts.
10. **Comprehension check.** Questions and answers follow separately.

## Questions

1. Why are the 100 questions unsuitable as a new held-out accuracy estimate?
2. Which variables change when a truncated answer is rerun at 256 tokens?
3. Why do `missing_final_answer` and `length` counts overlap?
4. What files remain if execution stops halfway through the first arm, and how should it be retried?
5. What does zero newly correct at 256 tokens establish, and what does it leave unresolved?

## Answers

1. They come from the training partition and are used for diagnosis after prior experiments, while the fixed 200-question validation partition is the held-out comparison in this project.
2. Only the generation cap changes; the model checkpoint, selected question, prompt, greedy decoding, and verifier stay fixed.
3. `finish_reason` records how generation stopped, while `reward_reason` records whether the final-answer contract was met; one response can fail both dimensions.
4. The manifest and some `evaluation-128.jsonl` rows may remain, but no completed summary exists. Use a new output directory rather than overwriting or treating partial rows as a finished run.
5. More tokens did not recover a strict answer for those 33 questions under this model and prompt. It does not determine whether unformatted answers contain correct arithmetic, nor whether a different prompt or model would help.

## Open questions

- A completed response without the required final line may still contain the correct number. A spot check found both a correct number in prose (`gsm8k/main/train/06046`, final prose says 832 and reference is 832) and wrong or incomplete arithmetic in other records. This is illustrative, not a measured rate. A blinded manual audit or a separately specified extraction diagnostic is needed before assigning mathematical error rates to the 83 format failures.
- The 100-prompt tokenizer comparison checks prompt encoding only. The checkpoint tokenizer warning and any effect on other text require separate investigation before comparing reloaded-checkpoint metrics to an in-memory model run as if all preprocessing were identical.
