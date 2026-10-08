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
    """Pad one complete rollout group and move it to tensors for GRPO loss.

    For example, take a shared prompt [1, 2], pad_token_id=9, and two responses:

        sample 0: ids=[7, 9, 0], mask=[1, 1, 0], old_log_probs=[-1, -2, 0], A=1
        sample 1: ids=[8, 6, 9], mask=[1, 1, 1], old_log_probs=[-3, -4, -5], A=-1

    The valid response lengths are 2 and 3, so both rows are padded to T=3.
    Sample 0's stored trailing 0 is discarded before padding with token 9:

        input_ids       = [[1, 2, 7, 9, 9], [1, 2, 8, 6, 9]]  # [G, P+T]
        attention_mask  = [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]  # [G, P+T]
        response_ids    = [[7, 9, 9], [8, 6, 9]]              # [G, T]
        response_mask   = [[[True, True, False],              # [1, G, T], bool
                            [True, True, True]]]
        old_log_probs   = [[[-1.0, -2.0, 0.0],                 # [1, G, T]
                            [-3.0, -4.0, -5.0]]]
        advantages      = [[1.0, -1.0]]                       # [1, G]
        prompt_length   = 2

    Here the first 9 in sample 0 is a generated EOS; the second is padding.
    The masks distinguish them. The leading size-1 axis is the batch of one
    prompt group. GRPO loss uses only response positions with a True mask.
    """
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
    """Compute the clipped GRPO policy loss for a batch of response groups.

    B is the number of prompt groups, G the responses per prompt, and T the
    padded response length. Prompt tokens are not part of the T axis.

    Args:
        new_log_probs: Float tensor [B, G, T] with current-policy log p at
            response positions; padded entries are ignored.
        old_log_probs: Float tensor [B, G, T] with saved pre-update log p at
            the same valid positions; padded entries are ignored.
        advantages: Float tensor [B, G] with one group-relative value per
            response; it is broadcast over that response's T positions.
        response_mask: Bool tensor [B, G, T]. True marks a generated token
            (including EOS); False marks padding excluded from the loss.
        clip_epsilon: Scalar number in (0, 1), not a tensor.
        reduction: Name of the supported response/group/batch averaging order.

    All four tensor arguments must be on the same device. The returned
    GrpoLossResult contains scalar (0-D) tensors for the loss and diagnostics,
    plus a Python int for valid_token_count.

    For each valid token, r = exp(new_log_prob - old_log_prob). Compare r * A
    with clamp(r, 1 - epsilon, 1 + epsilon) * A and take the smaller value.
    With epsilon=0.2, A=+1 and r=1.5 give min(1.5, 1.2)=1.2; A=-1 and
    r=0.5 give min(-0.5, -0.8)=-0.8. The loss is the negative of this
    surrogate: average valid tokens within each response, average responses
    within each group, then average the B groups. Thus a short response has
    the same group-level weight as a long response, and padding has none.

    Only new_log_probs carries policy gradients. Old log probabilities and
    advantages are fixed data. Returned diagnostics count only valid tokens:
    clip_fraction is the fraction of ratios outside the clip interval;
    approx_kl_old is mean((r - 1) - log(r)); mean_ratio and
    max_ratio_deviation summarize r and abs(r - 1). These diagnostics average
    over tokens, whereas the loss averages responses before groups.
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
    # Each [B, G] count is the number of generated tokens in one response.
    counts = response_mask.sum(dim=-1)
    if (counts == 0).any().item():
        raise ValueError("every response needs at least one valid token")
    if (
        not torch.isfinite(new_log_probs[response_mask]).all().item()
        or not torch.isfinite(old_log_probs[response_mask]).all().item()
        or not torch.isfinite(advantages).all().item()
    ):
        raise ValueError("valid GRPO loss inputs must be finite")

    # Set padding log-ratios to zero (ratio=1); only valid positions can
    # contribute to the objective or the diagnostics below.
    log_ratio = torch.where(
        response_mask, new_log_probs - old_log_probs.detach(),
        torch.zeros_like(new_log_probs),
    )
    ratio = log_ratio.exp()
    if not torch.isfinite(ratio[response_mask]).all().item():
        raise ValueError("GRPO ratio must be finite on valid response tokens")
    # Broadcast one fixed advantage per response across its T token positions.
    advantage_tokens = advantages.detach().unsqueeze(-1)
    surrogate = torch.minimum(
        ratio * advantage_tokens,
        ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage_tokens,
    )
    # First normalize by each response's own valid length, then by G and B.
    masked_surrogate = torch.where(response_mask, surrogate, torch.zeros_like(surrogate))
    response_losses = -masked_surrogate.sum(dim=-1) / counts
    loss = response_losses.mean(dim=-1).mean()
    if not torch.isfinite(loss).item():
        raise ValueError("GRPO loss must be finite")

    # Diagnostics use all valid tokens together and never enter backpropagation.
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
