import unittest

from scripts.diagnose_m0_failures import compare_extended, select_problems, summarize
from src.data.dataset import MathProblem


class DiagnosticTests(unittest.TestCase):
    def test_selection_is_stable_and_excludes_scanned_ids(self):
        training = tuple(MathProblem(f"train/{i}", f"question {i}", "#### 1") for i in range(8))
        selected = select_problems(training, {"train/3"}, 4)
        self.assertEqual(selected, select_problems(tuple(reversed(training)), {"train/3"}, 4))
        self.assertEqual(len({item.prompt_id for item in selected}), 4)
        self.assertNotIn("train/3", {item.prompt_id for item in selected})

    def test_summary_keeps_truncation_and_verifier_reasons_separate(self):
        rows = [
            {"prompt_id": "a", "reward": 0.0, "reward_reason": "missing_final_answer", "finish_reason": "length"},
            {"prompt_id": "b", "reward": 0.0, "reward_reason": "incorrect_answer", "finish_reason": "eos"},
            {"prompt_id": "c", "reward": 1.0, "reward_reason": "correct_answer", "finish_reason": "eos"},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["correct_count"], 1)
        self.assertEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["finish_by_reward_reason"]["length:missing_final_answer"], 1)

    def test_extended_comparison_requires_exact_truncated_subset(self):
        short = [
            {"prompt_id": "a", "reward": 0.0, "reward_reason": "missing_final_answer", "finish_reason": "length", "response_text": "work"},
            {"prompt_id": "b", "reward": 1.0, "reward_reason": "correct_answer", "finish_reason": "eos", "response_text": "#### 1"},
        ]
        long = [{"prompt_id": "a", "reward": 1.0, "reward_reason": "correct_answer", "finish_reason": "eos", "response_text": "work\n#### 1"}]
        result = compare_extended(short, long)
        self.assertEqual(result["rerun_count"], 1)
        self.assertEqual(result["new_correct_count"], 1)
        self.assertEqual(result["decoded_short_text_is_long_prefix_count"], 1)
        with self.assertRaisesRegex(ValueError, "match exactly"):
            compare_extended(short, [])


if __name__ == "__main__":
    unittest.main()
