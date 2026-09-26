import unittest
import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from src.data.dataset import MathProblem
from src.data.schema import Trajectory, read_jsonl, write_jsonl
from src.reward.math_verifier import parse_final_answer, verify_answer, verify_group
from scripts.score_rollout import main as score_main


def make_group(responses: list[str]) -> list[Trajectory]:
    return [
        Trajectory(
            schema_version=1,
            prompt_id="gsm8k/main/train/00000",
            group_id="step-000000/gsm8k/main/train/00000",
            sample_index=index,
            group_size=len(responses),
            prompt_text="How many clips?",
            prompt_token_ids=[1, 2],
            response_text=response,
            response_token_ids=[3, 4],
            response_mask=[1, 1],
            finish_reason="eos",
            policy_version="step-000000",
            model_revision="model-sha",
            tokenizer_revision="tokenizer-sha",
            old_log_probs=[-1.0, -2.0],
        )
        for index, response in enumerate(responses)
    ]


class MathVerifierTests(unittest.TestCase):
    def test_parses_only_complete_final_line_and_normalizes_number(self) -> None:
        self.assertEqual(parse_final_answer("work\n#### -1,234.50\n"), Decimal("-1234.50"))
        self.assertEqual(parse_final_answer("#### +72.0"), Decimal("72.0"))
        for response in (
            "48 + 24 = 72", "#### 72\nextra", "#### 72 clips", "#### 1,23",
            "#### 72.0.0", "#### NaN", "#### 7e1", "#### ٧٢",
        ):
            with self.subTest(response=response):
                self.assertIsNone(parse_final_answer(response))

    def test_scores_exact_numeric_answer_and_reports_failure_reasons(self) -> None:
        reference = "Natalia sold clips.\n#### 72"
        cases = [
            ("calculation\n#### 72.0", 1.0, "correct_answer"),
            ("#### 73", 0.0, "incorrect_answer"),
            ("48/2 = 24. 48 + 24 = 72.", 0.0, "missing_final_answer"),
            ("#### seventy-two", 0.0, "invalid_final_answer"),
        ]
        for response, reward, reason in cases:
            with self.subTest(response=response):
                result = verify_answer(response, reference)
                self.assertEqual((result.reward, result.reason), (reward, reason))

    def test_invalid_reference_is_an_error_not_a_zero_reward(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference answer"):
            verify_answer("#### 72", "answer: 72")

    def test_scores_full_group_without_mutating_rollout(self) -> None:
        problem = MathProblem("gsm8k/main/train/00000", "How many clips?", "work\n#### 72")
        group = make_group(["#### 72", "#### 73", "72", "#### no"])
        scored = verify_group(problem, group)
        self.assertEqual([item.reward for item in scored], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(
            [item.reward_reason for item in scored],
            ["correct_answer", "incorrect_answer", "missing_final_answer", "invalid_final_answer"],
        )
        self.assertTrue(all(item.reward is None for item in group))
        self.assertTrue(all(item.advantage is None for item in scored))
        self.assertEqual(scored[0].old_log_probs, group[0].old_log_probs)

    def test_rejects_wrong_reference_problem_and_incomplete_group(self) -> None:
        group = make_group(["#### 72", "#### 72"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_group(MathProblem(group[0].prompt_id, "Different question", "#### 72"), group)
        with self.assertRaisesRegex(ValueError, "group_size"):
            verify_group(MathProblem(group[0].prompt_id, group[0].prompt_text, "#### 72"), group[:1])
        with self.assertRaisesRegex(ValueError, "inconsistent prompt_text"):
            verify_group(
                MathProblem(group[0].prompt_id, group[0].prompt_text, "#### 72"),
                [group[0], replace(group[1], prompt_text="Different question")],
            )

    def test_score_command_uses_training_partition_and_preserves_input(self) -> None:
        problem = MathProblem("gsm8k/main/train/00000", "How many clips?", "#### 72")
        group = make_group(["#### 72", "72"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, input_path, output_path = (
                root / "config.yaml", root / "rollout.jsonl", root / "scored.jsonl"
            )
            config_path.write_text(yaml.safe_dump({
                "generation": {"group_size": 2},
                "model": {"revision": "model-sha", "tokenizer_revision": "tokenizer-sha"},
                "data": {},
            }))
            write_jsonl(input_path, group)
            with patch("scripts.score_rollout.load_pinned_train", return_value=SimpleNamespace(training=())):
                with self.assertRaisesRegex(ValueError, "not in the pinned training partition"):
                    score_main(config_path, input_path, output_path)
            self.assertFalse(output_path.exists())
            with patch("scripts.score_rollout.load_pinned_train", return_value=SimpleNamespace(training=(problem,))):
                score_main(config_path, input_path, output_path)
            self.assertEqual([item.reward for item in read_jsonl(output_path)], [1.0, 0.0])
            self.assertEqual(read_jsonl(input_path), group)


if __name__ == "__main__":
    unittest.main()
