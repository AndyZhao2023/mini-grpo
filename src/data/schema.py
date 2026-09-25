"""Validated, JSON-native trajectory records for one sampled response.

Model tensors are intentionally outside this module. A list may include padded
response IDs, but the mask marks only generated tokens, including EOS.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def _nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _token_ids(name: str, values: object) -> None:
    if not isinstance(values, list) or not values or any(
        type(value) is not int or value < 0 for value in values
    ):
        raise ValueError(f"{name} must be a nonempty list of nonnegative integer IDs")


def _finite(name: str, value: object) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass
class Trajectory:
    schema_version: int
    prompt_id: str
    group_id: str
    sample_index: int
    group_size: int
    prompt_text: str
    prompt_token_ids: list[int]
    response_text: str
    response_token_ids: list[int]
    response_mask: list[int]
    finish_reason: str
    policy_version: str
    model_revision: str
    tokenizer_revision: str
    old_log_probs: list[float] | None = None
    reward: float | None = None
    reward_reason: str | None = None
    advantage: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        for name in (
            "prompt_id", "group_id", "prompt_text", "policy_version",
            "model_revision", "tokenizer_revision",
        ):
            _nonempty_string(name, getattr(self, name))
        if not isinstance(self.response_text, str):
            raise ValueError("response_text must be a string")
        if type(self.group_size) is not int or self.group_size < 1:
            raise ValueError("group_size must be a positive integer")
        if (type(self.sample_index) is not int or not 0 <= self.sample_index < self.group_size):
            raise ValueError("sample_index must be in [0, group_size)")
        _token_ids("prompt_token_ids", self.prompt_token_ids)
        _token_ids("response_token_ids", self.response_token_ids)
        if (
            not isinstance(self.response_mask, list)
            or len(self.response_mask) != len(self.response_token_ids)
            or any(type(value) is not int or value not in (0, 1) for value in self.response_mask)
            or not self.response_mask[0]
            or any(left < right for left, right in zip(self.response_mask, self.response_mask[1:]))
        ):
            raise ValueError("response_mask must align with response IDs and be 1s then 0s")
        if self.finish_reason not in ("eos", "length"):
            raise ValueError("finish_reason must be 'eos' or 'length'")
        if self.old_log_probs is not None:
            if not isinstance(self.old_log_probs, list) or len(self.old_log_probs) != len(self.response_token_ids):
                raise ValueError("old_log_probs must align with response IDs")
            for value in self.old_log_probs:
                _finite("old_log_probs", value)
        for name in ("reward", "advantage"):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value)
        if self.reward_reason is not None and not isinstance(self.reward_reason, str):
            raise ValueError("reward_reason must be a string or null")

    @classmethod
    def from_dict(cls, value: dict) -> Trajectory:
        if not isinstance(value, dict):
            raise ValueError("trajectory must be a JSON object")
        return cls(**value)

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def validate_group(trajectories: list[Trajectory], *, require_complete: bool = False) -> None:
    """Validate exactly one full rollout group for a single prompt/version."""
    if not trajectories:
        raise ValueError("group must contain at least one trajectory")
    first = trajectories[0]
    expected = first.group_size
    if len(trajectories) != expected:
        raise ValueError(f"group_size={expected} but found {len(trajectories)} trajectories")
    for item in trajectories:
        item.validate()
        for name in (
            "group_id", "group_size", "prompt_id", "prompt_text", "prompt_token_ids",
            "policy_version", "model_revision", "tokenizer_revision",
        ):
            if getattr(item, name) != getattr(first, name):
                raise ValueError(f"group has inconsistent {name}")
        if require_complete:
            for name in ("old_log_probs", "reward", "advantage"):
                if getattr(item, name) is None:
                    raise ValueError(f"complete group requires {name}")
    if {item.sample_index for item in trajectories} != set(range(expected)):
        raise ValueError("group sample_index values must be unique and cover [0, group_size)")


def write_jsonl(path: str | Path, trajectories: Iterable[Trajectory]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in trajectories:
            file.write(json.dumps(item.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: str | Path) -> list[Trajectory]:
    records: list[Trajectory] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                records.append(Trajectory.from_dict(json.loads(line)))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid trajectory at line {line_number}: {error}") from error
    return records
