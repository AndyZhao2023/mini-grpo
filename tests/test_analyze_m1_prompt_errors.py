import unittest

from decimal import Decimal

from scripts.analyze_m1_prompt_errors import analyze_pairs, value_proxy
from src.data.dataset import MathProblem


def row(prompt_id, response, reward, reason, finish="eos"):
    return {
        "prompt_id": prompt_id, "response_text": response,
        "response_tokens": len(response.split()), "finish_reason": finish,
        "reward": reward, "reward_reason": reason,
    }


class PromptErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.problems = {
            "a": MathProblem("a", "question a", "#### 5"),
            "b": MathProblem("b", "question b", "#### 9"),
            "c": MathProblem("c", "question c", "#### 2"),
        }
        self.baseline = [
            row("a", "The answer is 5.", 0.0, "missing_final_answer"),
            row("b", "#### 9", 1.0, "correct_answer"),
            row("c", "#### 1", 0.0, "incorrect_answer"),
        ]
        self.variant = [
            row("a", "#### 5", 1.0, "correct_answer"),
            row("b", "#### 8", 0.0, "incorrect_answer"),
            row("c", "#### 4", 0.0, "incorrect_answer"),
        ]
        self.annotations = [
            {"prompt_id": "a", "failed_arm": "baseline",
             "failure_class": "correct_value_wrong_format", "candidate": "5",
             "evidence": "answer is 5"},
            {"prompt_id": "b", "failed_arm": "two_step_example",
             "failure_class": "wrong_value", "candidate": "8", "evidence": "#### 8"},
        ]

    def test_classifies_every_pair_and_only_annotates_flips(self):
        paired, summary = analyze_pairs(
            self.baseline, self.variant, self.annotations, self.problems,
        )
        self.assertEqual([item["change"] for item in paired], [
            "newly_correct", "lost_correct", "incorrect_in_both",
        ])
        self.assertIsNone(paired[2]["failed_response_annotation"])
        self.assertEqual(paired[0]["baseline_value_proxy"]["class"], "last_number_matches_reference")
        self.assertEqual(summary["flip_failure_class_counts"]["newly_correct"], {
            "correct_value_wrong_format": 1,
        })
        self.assertEqual(summary["paired"]["lost_strict_correct"], 1)

    def test_requires_complete_annotations_and_original_scores(self):
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            analyze_pairs(self.baseline, self.variant, self.annotations[:1], self.problems)
        changed = [dict(item) for item in self.baseline]
        changed[0]["reward_reason"] = "incorrect_answer"
        with self.assertRaisesRegex(ValueError, "saved verifier"):
            analyze_pairs(changed, self.variant, self.annotations, self.problems)

    def test_rejects_unsupported_or_misclassified_number(self):
        changed = [dict(item) for item in self.annotations]
        changed[0]["candidate"] = "6"
        with self.assertRaisesRegex(ValueError, "candidate is not supported"):
            analyze_pairs(self.baseline, self.variant, changed, self.problems)
        changed[0]["candidate"] = "5"
        changed[0]["failure_class"] = "wrong_value"
        with self.assertRaisesRegex(ValueError, "disagree with reference"):
            analyze_pairs(self.baseline, self.variant, changed, self.problems)
        changed = [dict(item) for item in self.annotations]
        changed[1]["candidate"] = "7"
        changed[1]["evidence"] = "7"
        variant = [dict(item) for item in self.variant]
        variant[1]["response_text"] = "7\n#### 8"
        with self.assertRaisesRegex(ValueError, "differs from the valid final"):
            analyze_pairs(self.baseline, variant, changed, self.problems)

    def test_truncated_response_has_no_claimed_answer(self):
        baseline = [row("a", "5 +", 0.0, "missing_final_answer", "length")]
        variant = [self.variant[0]]
        annotation = [{"prompt_id": "a", "failed_arm": "baseline",
                       "failure_class": "truncated_unresolved", "candidate": None,
                       "evidence": "5 +"}]
        _, summary = analyze_pairs(baseline, variant, annotation, self.problems)
        self.assertEqual(summary["flip_failure_class_counts"]["newly_correct"], {
            "truncated_unresolved": 1,
        })
        annotation[0]["candidate"] = "5"
        with self.assertRaisesRegex(ValueError, "must have no candidate"):
            analyze_pairs(baseline, variant, annotation, self.problems)

    def test_last_number_is_only_a_proxy_and_length_abstains(self):
        self.assertEqual(value_proxy(
            row("a", "5 is an intermediate value. Then 7.", 0.0, "missing_final_answer"),
            Decimal("5"),
        )["class"], "last_number_differs")
        self.assertEqual(value_proxy(
            row("a", "The answer is $5.", 0.0, "missing_final_answer"), Decimal("5"),
        )["candidate"], "5")
        self.assertEqual(value_proxy(
            row("a", "5 +", 0.0, "missing_final_answer", "length"), Decimal("5"),
        )["class"], "truncated_unresolved")


if __name__ == "__main__":
    unittest.main()
