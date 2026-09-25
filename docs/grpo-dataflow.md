# M0 data flow and representation contract

Notation: `B` is prompts per rollout batch, `G=4` is responses per prompt, `P` is prompt-token length, and `T<=128` is generated-token length. Variable lengths are padded only in memory. A JSONL line stores one trajectory with native Python/JSON scalar and list types; it never stores a tensor or a device handle.

```text
GSM8K train question + stable row ID [B] (CPU)
  → fixed system/user chat template → prompt token IDs [B,P] (CPU → MPS)
  → policy rollout → response IDs [B,G,T], EOS/length status (MPS → CPU)
  → raw policy forward before update → old_log_probs [B,G,T] (MPS → CPU)
  → deterministic answer verifier → reward and reason [B,G] (CPU)
  → group normalization → advantage [B,G] (CPU or MPS)
  → broadcast advantage over valid response tokens [B,G,T] (MPS)
  → clipped GRPO loss → gradient → optimizer step (MPS)
  → next rollout with incremented policy_version
```

| Artifact | Producer | Consumer | Storage and checks |
| --- | --- | --- | --- |
| Question, ID, prompt text | GSM8K loader and chat template | Rollout, verifier, JSONL | CPU; prompt excludes reference answer |
| `prompt_token_ids` | Tokenizer | Generator and log-prob forward | CPU JSONL; move to MPS for model calls |
| `response_token_ids`, `response_mask`, `finish_reason` | Rollout | Old-policy forward, loss, JSONL | CPU JSONL; EOS is a valid generated token; padding is masked out |
| `old_log_probs` | Raw model logits, no grad, **before** any step | Ratio and diagnostics | One value per response ID; detached CPU JSONL; regenerated for each policy version |
| `reward`, `reward_reason` | Deterministic verifier | Group normalization, diagnostics | CPU JSONL; complete after verification |
| `advantage` | Population mean/std over each full group | Every valid token in its response | CPU JSONL; all-equal reward group gets exactly zero |
| `policy_version`, model/tokenizer revisions | Trainer and config | Group validation, reproducibility | Text; every response in a group shares one version |

`Trajectory` allows `old_log_probs`, `reward`, and `advantage` to be absent while a rollout is being assembled. A complete training group must have all three before an update. `validate_group(..., require_complete=True)` enforces this boundary. The JSONL writer validates each row; the reader reports malformed line numbers. `response_mask` is a prefix of ones followed by zeros, so it can represent EOS, length truncation, and padding without assigning loss to pad IDs.

The Day 1 model smoke uses two fixed arithmetic prompts, not GSM8K rows. Its backward pass establishes that the pinned model can generate and propagate gradients on this machine. It does not verify `old_log_probs`, reward, or the GRPO loss; those are Day 2–3 checks.

The train/internal-validation split is fixed from GSM8K `train` with seed 42 and 200 held-out questions. The official `test` split is run only in the final report. Evaluation uses greedy decoding and `max_new_tokens=128`; training reward is diagnostic and is not substituted for held-out pass@1.
