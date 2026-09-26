"""Sample grouped responses and score them under the pre-update raw policy."""

from __future__ import annotations

import math

import torch
from transformers import GenerationConfig

from src.data.dataset import MathProblem, build_prompt_messages
from src.data.schema import Trajectory, validate_group


def trim_generated_response(tokens: list[int], eos_token_id: int) -> tuple[list[int], str]:
    """Keep the first EOS as a valid token; discard EOS-valued output padding."""
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("eos_token_id must be a nonnegative integer")
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("generated response must contain nonnegative token IDs")
    if eos_token_id in tokens:
        return tokens[:tokens.index(eos_token_id) + 1], "eos"
    return tokens, "length"


def gather_response_log_probs(
    logits: torch.Tensor, response_ids: torch.Tensor, prompt_length: int,
) -> torch.Tensor:
    """Score response token t from causal logits at prompt_length - 1 + t."""
    if logits.ndim != 3 or response_ids.ndim != 2:
        raise ValueError("logits must be [B,L,V] and response_ids must be [B,T]")
    if (
        type(prompt_length) is not int or prompt_length < 1
        or response_ids.shape[1] < 1
        or logits.shape[0] != response_ids.shape[0]
        or logits.shape[1] < prompt_length + response_ids.shape[1] - 1
    ):
        raise ValueError("prompt_length or response shape is inconsistent with logits")
    selected = logits[:, prompt_length - 1:prompt_length - 1 + response_ids.shape[1], :]
    return selected.float().log_softmax(dim=-1).gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)


def compute_old_log_probs(
    model, prompt_ids: list[int], response_ids: list[int], device: str,
) -> list[float]:
    """Recompute raw-token log probabilities with dropout and autograd off."""
    if not prompt_ids or not response_ids:
        raise ValueError("prompt and response token IDs must be nonempty")
    model.eval()
    full_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        raw_logits = model(
            input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
        ).logits
        log_probs = gather_response_log_probs(
            raw_logits, full_ids[:, len(prompt_ids):], len(prompt_ids),
        )
    values = log_probs[0].detach().cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("old policy produced non-finite log probabilities")
    return values


def sample_group(
    model, tokenizer, problem: MathProblem, config: dict, policy_version: str,
) -> list[Trajectory]:
    """Generate and raw-score G responses to one pinned training question."""
    generation = config["generation"]
    model_config = config["model"]
    if not policy_version:
        raise ValueError("policy_version must be nonempty")
    if generation["extra_logits_processors"] != []:
        raise ValueError("M0 does not permit extra logits processors")
    if not (
        generation["do_sample"] is True
        and generation["temperature"] == 1.0
        and generation["top_p"] == 1.0
        and generation["top_k"] == 0
        and generation["repetition_penalty"] == 1.0
    ):
        raise ValueError("M0 sampling must use temperature=1, top_p=1, top_k=0, repetition_penalty=1")
    group_size, max_new_tokens = generation["group_size"], generation["max_new_tokens"]
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be positive")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    eos_id = tokenizer.eos_token_id
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("tokenizer must define one eos_token_id")

    model.eval()
    prompt_ids = tokenizer.apply_chat_template(
        build_prompt_messages(problem, config["prompt"]["system"]),
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model_config["device"])
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError("chat template must produce one nonempty prompt")
    prompt_length = prompt_ids.shape[1]
    prompt_token_ids = prompt_ids[0].tolist()
    sampling = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=generation["repetition_penalty"],
        num_beams=1,
        num_return_sequences=group_size,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    with torch.no_grad():
        sequences = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            generation_config=sampling,
        )
    if sequences.ndim != 2 or sequences.shape[0] != group_size:
        raise ValueError("generation did not return one complete group")

    group: list[Trajectory] = []
    for sample_index, sequence in enumerate(sequences):
        if sequence[:prompt_length].tolist() != prompt_token_ids:
            raise ValueError("generated sequence does not start with the input prompt")
        response_ids, finish_reason = trim_generated_response(
            sequence[prompt_length:].tolist(), eos_id,
        )
        old_log_probs = compute_old_log_probs(
            model, prompt_token_ids, response_ids, model_config["device"],
        )
        group.append(Trajectory(
            schema_version=1,
            prompt_id=problem.prompt_id,
            group_id=f"{policy_version}/{problem.prompt_id}",
            sample_index=sample_index,
            group_size=group_size,
            prompt_text=problem.question,
            prompt_token_ids=prompt_token_ids,
            response_text=tokenizer.decode(response_ids, skip_special_tokens=True),
            response_token_ids=response_ids,
            response_mask=[1] * len(response_ids),
            finish_reason=finish_reason,
            policy_version=policy_version,
            model_revision=model_config["revision"],
            tokenizer_revision=model_config["tokenizer_revision"],
            old_log_probs=old_log_probs,
        ))
    validate_group(group)
    return group
