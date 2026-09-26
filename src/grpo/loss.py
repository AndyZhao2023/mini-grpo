"""Response-masked, sequence-normalized clipped GRPO policy loss."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.data.schema import Trajectory, validate_group


@dataclass(frozen=True)
class PackedGroup:
    input_ids: torch.Tensor  # [G, P+T]
    attention_mask: torch.Tensor  # [G, P+T]
    response_ids: torch.Tensor  # [G, T]
    response_mask: torch.Tensor  # [1, G, T], bool
    old_log_probs: torch.Tensor  # [1, G, T]
    advantages: torch.Tensor  # [1, G]
    prompt_length: int


@dataclass(frozen=True)
class GrpoLossResult:
    loss: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl_old: torch.Tensor
    mean_ratio: torch.Tensor
    max_ratio_deviation: torch.Tensor
    valid_token_count: int


def pack_group(
    group: list[Trajectory], pad_token_id: int, device: torch.device | str,
) -> PackedGroup:
    """Move one complete JSONL group to tensors; prompt and padding stay outside loss."""
    validate_group(group, require_complete=True)
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    prompt_ids = group[0].prompt_token_ids
    prompt_length = len(prompt_ids)
    max_response_length = max(sum(item.response_mask) for item in group)
    input_ids, attention_masks, response_ids = [], [], []
    response_masks, old_log_probs, advantages = [], [], []
    for item in group:
        length = sum(item.response_mask)
        padding = max_response_length - length
        active_ids = item.response_token_ids[:length]
        response_ids.append(active_ids + [pad_token_id] * padding)
        input_ids.append(prompt_ids + response_ids[-1])
        attention_masks.append([1] * (prompt_length + length) + [0] * padding)
        response_masks.append([True] * length + [False] * padding)
        old_log_probs.append(item.old_log_probs[:length] + [0.0] * padding)
        advantages.append(item.advantage)
    return PackedGroup(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attention_masks, dtype=torch.long, device=device),
        response_ids=torch.tensor(response_ids, dtype=torch.long, device=device),
        response_mask=torch.tensor([response_masks], dtype=torch.bool, device=device),
        old_log_probs=torch.tensor([old_log_probs], dtype=torch.float32, device=device),
        advantages=torch.tensor([advantages], dtype=torch.float32, device=device),
        prompt_length=prompt_length,
    )


def clipped_grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_epsilon: float,
    reduction: str = "response_then_group_then_batch",
) -> GrpoLossResult:
    """Average clipped surrogate over valid tokens, then responses, then groups.

    Inputs have shapes [B,G,T], [B,G,T], [B,G], and [B,G,T].
    Only new_log_probs carries the policy gradient; old values are detached.
    """
    if reduction != "response_then_group_then_batch":
        raise ValueError("unsupported GRPO loss reduction")
    if (
        type(clip_epsilon) not in (int, float)
        or not math.isfinite(clip_epsilon)
        or not 0 < clip_epsilon < 1
    ):
        raise ValueError("clip_epsilon must be finite and in (0, 1)")
    if (
        new_log_probs.ndim != 3
        or min(new_log_probs.shape) < 1
        or old_log_probs.shape != new_log_probs.shape
        or response_mask.shape != new_log_probs.shape
        or advantages.shape != new_log_probs.shape[:2]
    ):
        raise ValueError("expected log_probs and mask [B,G,T], advantages [B,G]")
    if (
        not new_log_probs.is_floating_point()
        or not old_log_probs.is_floating_point()
        or not advantages.is_floating_point()
        or response_mask.dtype != torch.bool
    ):
        raise ValueError("log probabilities and advantages must be floating; mask must be bool")
    if any(
        tensor.device != new_log_probs.device
        for tensor in (old_log_probs, advantages, response_mask)
    ):
        raise ValueError("all GRPO loss tensors must be on the same device")
    counts = response_mask.sum(dim=-1)
    if (counts == 0).any().item():
        raise ValueError("every response needs at least one valid token")
    if (
        not torch.isfinite(new_log_probs[response_mask]).all().item()
        or not torch.isfinite(old_log_probs[response_mask]).all().item()
        or not torch.isfinite(advantages).all().item()
    ):
        raise ValueError("valid GRPO loss inputs must be finite")

    log_ratio = torch.where(
        response_mask, new_log_probs - old_log_probs.detach(),
        torch.zeros_like(new_log_probs),
    )
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio[response_mask]).all().item():
        raise ValueError("GRPO ratio must be finite on valid response tokens")
    advantage_tokens = advantages.detach().unsqueeze(-1)
    surrogate = torch.minimum(
        ratio * advantage_tokens,
        ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage_tokens,
    )
    masked_surrogate = torch.where(response_mask, surrogate, torch.zeros_like(surrogate))
    response_losses = -masked_surrogate.sum(dim=-1) / counts
    loss = response_losses.mean(dim=-1).mean()
    if not torch.isfinite(loss).item():
        raise ValueError("GRPO loss must be finite")

    with torch.no_grad():
        valid_ratio = ratio[response_mask]
        valid_log_ratio = log_ratio[response_mask]
        clip_fraction = (
            (valid_ratio < 1 - clip_epsilon) | (valid_ratio > 1 + clip_epsilon)
        ).float().mean()
        approx_kl_old = ((valid_ratio - 1) - valid_log_ratio).mean()
        mean_ratio = valid_ratio.mean()
        max_ratio_deviation = (valid_ratio - 1).abs().max()
    return GrpoLossResult(
        loss, clip_fraction, approx_kl_old, mean_ratio,
        max_ratio_deviation, int(response_mask.sum().item()),
    )
