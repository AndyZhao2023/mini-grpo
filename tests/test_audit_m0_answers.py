import unittest

from scripts.audit_m0_answers import compare_answers, summarize
from src.data.dataset import MathProblem


class AnswerAuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"prompt_id": "a", "finish_reason": "eos", "reward_reason": "missing_final_answer",
             "response_text": "The answer is $1,200."},
            {"prompt_id": "b", "finish_reason": "eos", "reward_reason": "missing_final_answer",
             "response_text": "It is 7.5 units."},
            {"prompt_id": "c", "finish_reason": "eos", "reward_reason": "missing_final_answer",
             "response_text": "The answer is ####."},
            {"prompt_id": "d", "finish_reason": "length", "reward_reason": "missing_final_answer",
             "response_text": "Still calculating"},
        ]
        self.problems = {
            "a": MathProblem("a", "question a", "#### 1200"),
            "b": MathProblem("b", "question b", "#### 8"),
            "c": MathProblem("c", "question c", "#### 4"),
            "d": MathProblem("d", "question d", "#### 2"),
        }
        self.annotations = [
            {"prompt_id": "a", "kind": "terminal", "candidate": "1200",
             "evidence": "$1,200."},
            {"prompt_id": "b", "kind": "terminal", "candidate": "7.5",
             "evidence": "7.5 units"},
            {"prompt_id": "c", "kind": "unresolved", "candidate": None,
             "evidence": "####."},
        ]

    def test_compares_explicit_numbers_and_abstains_without_one(self):
        compared = compare_answers(self.rows, self.annotations, self.problems)
        self.assertEqual([item["comparison"] for item in compared], [
            "matches_reference", "different_from_reference", "unresolved",
        ])
        counts = summarize(self.rows, compared)
        self.assertEqual(counts["completed_missing_final_count"], 3)
        self.assertEqual(counts["truncated_missing_final_count"], 1)
        self.assertEqual(counts["terminal_match_count"], 1)

    def test_requires_exact_coverage_and_source_evidence(self):
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            compare_answers(self.rows, self.annotations[:-1], self.problems)
        changed = [dict(item) for item in self.annotations]
        changed[0]["candidate"] = "2"
        with self.assertRaisesRegex(ValueError, "absent from evidence"):
            compare_answers(self.rows, changed, self.problems)
        changed[0]["candidate"] = "1200"
        changed[0]["evidence"] = "made-up answer"
        with self.assertRaisesRegex(ValueError, "absent from response"):
            compare_answers(self.rows, changed, self.problems)


if __name__ == "__main__":
    unittest.main()
