# M0 same-rollout two-epoch experiment

1. **Outcome.** The original `mu=2` M0 schedule now has a recorded MPS run: one selected GSM8K training group receives two consecutive GRPO optimizer updates. Fixed internal validation changes from 9/200 to 8/200.
2. **Before.** `configs/m0-run.yaml` ran two fresh rollout/update cycles, one optimizer update per group; its validation was 9/200 before and after. The source plan additionally requested two updates on the same group.
3. **Why.** Supplied by the user's request to complete the remaining original M0 acceptance experiment.
4. **Core intuition.** A rollout freezes four responses, rewards, advantages, and `old_log_probs`. After the first optimizer step, the second step recomputes `new_log_probs` with changed weights but still compares them with that same frozen rollout. Its ratio is therefore expected to differ from one before the second step.
5. **Flow.** The pinned initial model greedily answers the fixed 200-question internal validation set (9 correct). A shortest-first training scan selects `gsm8k/main/train/01214` with rewards `[0, 0, 1, 0]`. Its `step-000000` rollout is saved once in `steps/step-000001/`. AdamW then produces `step-000001` and `step-000002` from that group without another sampling call. The final model answers the same 200 validation questions (8 correct).
6. **Explained change.** `scripts/run_m0.py` groups optimizer updates by `algorithm.update_epochs`; it only scans and writes trajectories at the first epoch, then passes the saved group to later epochs while retaining the optimizer state. `src/grpo/trainer.py` checks ratio near one only for the first epoch; later epochs still calculate and record the ratio, clipping, gradient norm, and policy drift. `configs/m0-mu2.yaml` pins `update_epochs: 2` and two total updates. The prior `update_epochs: 1` route remains available. CPU tests exercise both routes and the drift guard.
7. **Invariants and boundaries.** Each update has its own checkpoint and `step.json`; both step records name `rollout_policy_version=step-000000`, one trajectory directory, and data ages 0 and 1. The original four `old_log_probs` and advantages are not recomputed for epoch 2. The initial ratio guard remains on for every fresh rollout. Training scans use only the training partition; both evaluations use only the same 200 internal validation IDs. The official test split is untouched.
8. **Failure and recovery.** The CLI refuses an existing output directory. A missing mixed-reward group or excessive drift before a remaining update produces an incomplete summary. Checkpoints omit optimizer state, so resuming midway would change the experiment; rerun from the pinned initial model into a fresh directory. The first launch stopped before model load because a Hugging Face auxiliary-template API call returned 401 for an expired token. With the exact model and dataset revisions already cached, `HF_HUB_OFFLINE=1` loaded the local snapshot and the fresh run completed.
9. **Evidence.** The actual MPS run is [`results/m0/end-to-end-mu2-v1/summary.json`](../results/m0/end-to-end-mu2-v1/summary.json). The [first step](../results/m0/end-to-end-mu2-v1/steps/step-000001/step.json) has pre-update max ratio deviation `3.33e-5`, post-update clip fraction `0.0355`, and `approx_kl_old=0.00393`. The [second step](../results/m0/end-to-end-mu2-v1/steps/step-000002/step.json) has pre-update max ratio deviation `0.534`, pre/post clip fractions `0.0355/0.0851`, and post-update `approx_kl_old=0.01062`. The baseline JSONL and first selected `advantaged.jsonl` are byte-identical to the earlier two-fresh-rollout run. The final [validation counts](../results/m0/end-to-end-mu2-v1/validation-final.json) are 8/200, with one gained and two lost correct answers relative to this run's baseline; 58 response texts changed. Independently verified: 61 unit tests passed, compile and `git diff --check` passed, artifact consistency checks passed, and all 12 files in the retained final checkpoint passed `shasum -a 256 -c` against the [SHA-256 list](../results/m0/end-to-end-mu2-v1/checkpoint-sha256.txt).
10. **Comprehension check.** Questions and answers follow separately.

## Questions

1. Why is the second epoch's pre-update ratio allowed to differ from one?
2. Which artifact proves that both updates refer to the same sampled policy?
3. When does the runner sample again if `update_epochs` is two and more update groups are requested?
4. What happens if the first update exceeds the configured drift guard?
5. What does 9/200 to 8/200 establish, and what cannot be concluded from this one selected group?

## Answers

1. The policy weights changed after epoch 1 while `old_log_probs` remain tied to the original rollout. The ratio still compares current against original policy token probabilities.
2. Both `step.json` records set `rollout_policy_version` to `step-000000` and `rollout_artifact_dir` to the first step's directory; the second step has `data_age_updates=1`.
3. At the first epoch of the next group, after both updates on the current group; that rollout uses the latest policy version.
4. It writes the first checkpoint and step record, skips the remaining update, runs final internal validation, and reports `incomplete_post_step_drift`.
5. It establishes no held-out gain for this concrete two-update run. It cannot establish whether `mu=2` is generally worse than fresh rollouts or what a longer run would do.

## Open questions

- These two schedules share the baseline and first selected group, but then use different training exposure. A broader comparison would require multiple groups and seeds rather than generalizing from one `mu=2` group.
- The final checkpoint has model/tokenizer weights but no AdamW state. A mid-run resume feature would need a separate contract and validation.
