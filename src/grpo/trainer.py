"""One guarded full-parameter GRPO optimizer step on a mixed-reward group."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.data.schema import Trajectory, validate_group
from src.grpo.loss import clipped_grpo_loss, pack_group
from src.rollout.hf_rollout import gather_response_log_probs


@dataclass(frozen=True)
class StepMetrics:
    pre_loss: float
    post_loss: float
    pre_max_ratio_deviation: float
    post_max_ratio_deviation: float
    pre_clip_fraction: float
    post_clip_fraction: float
    post_approx_kl_old: float
    gradient_norm_before_clip: float
    valid_token_count: int


def run_one_grpo_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    group: list[Trajectory],
    *,
    pad_token_id: int,
    device: torch.device | str,
    clip_epsilon: float,
    loss_reduction: str,
    max_grad_norm: float,
    ratio_tolerance: float = 1e-3,
) -> StepMetrics:
    """Use saved old log-probs for one update, then measure policy drift."""
    validate_group(group, require_complete=True)
    if len({item.reward for item in group}) < 2:
        raise ValueError("one-step training requires a mixed-reward group")
    if all(item.advantage == 0.0 for item in group):
        raise ValueError("one-step training requires nonzero advantages")
    if (
        type(max_grad_norm) not in (int, float)
        or not math.isfinite(max_grad_norm)
        or max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be finite and positive")
    if (
        type(ratio_tolerance) not in (int, float)
        or not math.isfinite(ratio_tolerance)
        or ratio_tolerance <= 0
    ):
        raise ValueError("ratio_tolerance must be finite and positive")
    packed = pack_group(group, pad_token_id, device)
    model.eval()  # Disable dropout while retaining autograd.
    optimizer.zero_grad(set_to_none=True)

    def current_log_probs() -> torch.Tensor:
        logits = model(
            input_ids=packed.input_ids,
            attention_mask=packed.attention_mask,
        ).logits
        return gather_response_log_probs(
            logits, packed.response_ids, packed.prompt_length,
        ).unsqueeze(0)

    pre = clipped_grpo_loss(
        current_log_probs(), packed.old_log_probs, packed.advantages,
        packed.response_mask, clip_epsilon, loss_reduction,
    )
    pre_deviation = pre.max_ratio_deviation.item()
    if pre_deviation > ratio_tolerance:
        raise RuntimeError("pre-update ratio differs from 1 beyond tolerance")
    pre_loss = pre.loss.item()
    pre_clip_fraction = pre.clip_fraction.item()
    pre.loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm, error_if_nonfinite=True,
    ).item()
    if not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("GRPO backward produced no finite nonzero gradient")
    optimizer.step()

    with torch.no_grad():
        post = clipped_grpo_loss(
            current_log_probs(), packed.old_log_probs, packed.advantages,
            packed.response_mask, clip_epsilon, loss_reduction,
        )
    return StepMetrics(
        pre_loss=pre_loss,
        post_loss=post.loss.item(),
        pre_max_ratio_deviation=pre_deviation,
        post_max_ratio_deviation=post.max_ratio_deviation.item(),
        pre_clip_fraction=pre_clip_fraction,
        post_clip_fraction=post.clip_fraction.item(),
        post_approx_kl_old=post.approx_kl_old.item(),
        gradient_norm_before_clip=gradient_norm,
        valid_token_count=pre.valid_token_count,
    )
