"""Greedy, answer-free evaluation on a fixed GSM8K validation partition."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch
from transformers import GenerationConfig

from src.data.dataset import MathProblem, build_prompt_messages
from src.reward.math_verifier import verify_answer
from src.rollout.hf_rollout import trim_generated_response


def evaluate_model(
    model, tokenizer, problems: Sequence[MathProblem], config: dict,
) -> tuple[list[dict], dict]:
    """Return one result per question and aggregate strict pass@1 counts."""
    if not problems:
        raise ValueError("validation problems must be nonempty")
    evaluation = config["evaluation"]
    if evaluation["do_sample"] is not False:
        raise ValueError("evaluation must use greedy decoding")
    max_new_tokens = evaluation["max_new_tokens"]
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("evaluation.max_new_tokens must be positive")
    eos_id = tokenizer.eos_token_id
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("tokenizer must define one eos_token_id")
    device = config["model"]["device"]
    generation = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        eos_token_id=eos_id, pad_token_id=eos_id,
    )
    model.eval()
    rows: list[dict] = []
    reasons: Counter[str] = Counter()
    for problem in problems:
        prompt_ids = tokenizer.apply_chat_template(
            build_prompt_messages(problem, config["prompt"]["system"]),
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(device)
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
            raise ValueError("chat template must produce one nonempty prompt")
        with torch.no_grad():
            # Otherwise Transformers may replace greedy settings with model defaults.
            sequences = model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                generation_config=generation, use_model_defaults=False,
            )
        if sequences.ndim != 2 or sequences.shape[0] != 1 or not torch.equal(
            sequences[0, :prompt_ids.shape[1]], prompt_ids[0],
        ):
            raise ValueError("evaluation generation must preserve one input prompt")
        response_ids, finish_reason = trim_generated_response(
            sequences[0, prompt_ids.shape[1]:].tolist(), eos_id,
        )
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        checked = verify_answer(response_text, problem.reference_answer)
        reasons[checked.reason] += 1
        rows.append({
            "prompt_id": problem.prompt_id,
            "response_text": response_text,
            "response_tokens": len(response_ids),
            "finish_reason": finish_reason,
            "reward": checked.reward,
            "reward_reason": checked.reason,
        })
    correct = reasons["correct_answer"]
    return rows, {
        "question_count": len(rows),
        "correct_count": correct,
        "accuracy": correct / len(rows),
        "reason_counts": dict(reasons),
        "truncation_count": sum(row["finish_reason"] == "length" for row in rows),
        "mean_response_tokens": sum(row["response_tokens"] for row in rows) / len(rows),
    }
