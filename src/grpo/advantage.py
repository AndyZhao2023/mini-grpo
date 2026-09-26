"""Normalize outcome rewards within one complete rollout group."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

from src.data.schema import Trajectory, validate_group


@dataclass(frozen=True)
class AdvantageResult:
    group: list[Trajectory]
    reward_mean: float
    reward_std: float
    zero_variance: bool


def normalize_group_advantages(
    group: list[Trajectory], algorithm_config: Mapping[str, object],
) -> AdvantageResult:
    """Set A_i=(r_i-mean)/(population_std+epsilon); equal rewards yield exact zero."""
    validate_group(group)
    if algorithm_config["advantage_std"] != "population":
        raise ValueError("advantage_std must be population")
    epsilon = algorithm_config["advantage_epsilon"]
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("advantage_epsilon must be a finite positive number")
    if any(item.reward is None or not item.reward_reason for item in group):
        raise ValueError("group must have rewards and reward reasons before normalization")
    if any(item.advantage is not None for item in group):
        raise ValueError("group already has advantages")
    rewards = [float(item.reward) for item in group]
    reward_mean = math.fsum(rewards) / len(rewards)
    zero_variance = all(reward == rewards[0] for reward in rewards)
    reward_std = 0.0 if zero_variance else math.sqrt(
        math.fsum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards)
    )
    if not math.isfinite(reward_mean) or not math.isfinite(reward_std):
        raise ValueError("reward normalization produced non-finite statistics")
    advantages = (
        [0.0] * len(rewards) if zero_variance else
        [(reward - reward_mean) / (reward_std + epsilon) for reward in rewards]
    )
    normalized = [replace(item, advantage=value) for item, value in zip(group, advantages)]
    validate_group(normalized, require_complete=True)
    return AdvantageResult(normalized, reward_mean, reward_std, zero_variance)
