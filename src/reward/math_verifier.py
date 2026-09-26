"""Strict GSM8K final-answer parsing and deterministic 0/1 rewards."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal

from src.data.dataset import MathProblem
from src.data.schema import Trajectory, validate_group


_FINAL_ANSWER = re.compile(
    r"####[ \t]+(?P<number>[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)"
)


@dataclass(frozen=True)
class VerificationResult:
    reward: float
    reason: str


def parse_final_answer(text: str) -> Decimal | None:
    """Read only a complete `#### <number>` on the last nonblank line."""
    if not isinstance(text, str):
        raise TypeError("answer text must be a string")
    lines = text.strip().splitlines()
    if not lines:
        return None
    match = _FINAL_ANSWER.fullmatch(lines[-1].strip())
    if match is None:
        return None
    return Decimal(match.group("number").replace(",", ""))


def verify_answer(response_text: str, reference_answer: str) -> VerificationResult:
    """Score a generated answer against a trusted GSM8K reference answer."""
    expected = parse_final_answer(reference_answer)
    if expected is None:
        raise ValueError("reference answer has no valid final #### number")
    predicted = parse_final_answer(response_text)
    if predicted is None:
        last_line = response_text.strip().splitlines()[-1].strip() if response_text.strip() else ""
        reason = "invalid_final_answer" if last_line.startswith("####") else "missing_final_answer"
        return VerificationResult(0.0, reason)
    if predicted == expected:
        return VerificationResult(1.0, "correct_answer")
    return VerificationResult(0.0, "incorrect_answer")


def verify_group(problem: MathProblem, group: list[Trajectory]) -> list[Trajectory]:
    """Attach rewards without changing the saved, unscored rollout records."""
    validate_group(group)
    if group[0].prompt_id != problem.prompt_id or group[0].prompt_text != problem.question:
        raise ValueError("trajectory group does not match the reference problem")
    if parse_final_answer(problem.reference_answer) is None:
        raise ValueError("reference answer has no valid final #### number")
    scored = []
    for item in group:
        result = verify_answer(item.response_text, problem.reference_answer)
        scored.append(replace(item, reward=result.reward, reward_reason=result.reason))
    validate_group(scored)
    return scored
