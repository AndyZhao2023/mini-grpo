import json
import tempfile
import unittest
from pathlib import Path

from src.data.schema import Trajectory, read_jsonl, validate_group, write_jsonl


def row(index: int = 0, **changes: object) -> dict:
    value = {
        "schema_version": 1,
        "prompt_id": "smoke-1",
        "group_id": "rollout-0/smoke-1",
        "sample_index": index,
        "group_size": 2,
        "prompt_text": "What is 1 + 1?",
        "prompt_token_ids": [11, 12],
        "response_text": "#### 2",
        "response_token_ids": [21, 22, 0],
        "response_mask": [1, 1, 0],
        "finish_reason": "eos",
        "policy_version": "step-0",
        "model_revision": "model-sha",
        "tokenizer_revision": "tokenizer-sha",
        "old_log_probs": [-0.1, -0.2, 0.0],
        "reward": 1.0,
        "reward_reason": "exact",
        "advantage": 1.0,
    }
    value.update(changes)
    return value


class TrajectoryTests(unittest.TestCase):
    def test_jsonl_round_trip_and_complete_group(self) -> None:
        group = [Trajectory.from_dict(row(i)) for i in range(2)]
        validate_group(group, require_complete=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            write_jsonl(path, group)
            self.assertEqual(read_jsonl(path), group)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_mask_and_log_prob_lengths_must_match_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "response_mask"):
            Trajectory.from_dict(row(response_mask=[1, 1]))
        with self.assertRaisesRegex(ValueError, "old_log_probs"):
            Trajectory.from_dict(row(old_log_probs=[-0.1]))
        with self.assertRaisesRegex(ValueError, "response_mask"):
            Trajectory.from_dict(row(response_mask=[1, 0, 1]))

    def test_non_finite_values_are_rejected(self) -> None:
        for field in ("reward", "advantage"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                Trajectory.from_dict(row(**{field: float("nan")}))
        with self.assertRaisesRegex(ValueError, "old_log_probs"):
            Trajectory.from_dict(row(old_log_probs=[0.0, float("inf"), 0.0]))

    def test_mixed_policy_and_duplicate_sample_are_rejected(self) -> None:
        first = Trajectory.from_dict(row(0))
        with self.assertRaisesRegex(ValueError, "policy_version"):
            validate_group([first, Trajectory.from_dict(row(1, policy_version="step-1"))])
        with self.assertRaisesRegex(ValueError, "sample_index"):
            validate_group([first, Trajectory.from_dict(row(0))])

    def test_incomplete_group_rejected_for_training(self) -> None:
        group = [Trajectory.from_dict(row(i, old_log_probs=None)) for i in range(2)]
        validate_group(group)
        with self.assertRaisesRegex(ValueError, "old_log_probs"):
            validate_group(group, require_complete=True)

    def test_truncated_jsonl_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text(json.dumps(row()) + "\n{bad", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                read_jsonl(path)

    def test_bool_is_not_a_token_id_or_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt_token_ids"):
            Trajectory.from_dict(row(prompt_token_ids=[True]))
        with self.assertRaisesRegex(ValueError, "response_mask"):
            Trajectory.from_dict(row(response_mask=[True, 1, 0]))


if __name__ == "__main__":
    unittest.main()
