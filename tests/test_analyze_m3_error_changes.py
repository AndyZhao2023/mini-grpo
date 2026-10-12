import unittest

from scripts.analyze_m3_error_changes import analyze_pairs, target_class
from src.data.dataset import MathProblem


def row(prompt_id, text, reason, *, finish="eos"):
    return {
        "prompt_id": prompt_id, "response_text": text,
        "reward": float(reason == "correct_answer"),
        "reward_reason": reason, "finish_reason": finish,
    }


def annotation(prompt_id, kind, candidate, evidence, mechanism=None, final_evidence=None):
    return {
        "prompt_id": prompt_id, "initial_kind": kind,
        "initial_candidate": candidate, "initial_evidence": evidence,
        "gain_mechanism": mechanism, "final_evidence": final_evidence,
    }


class M3ErrorAuditTests(unittest.TestCase):
    def setUp(self):
        self.problems = {
            name: MathProblem(name, f"question {name}", "#### 7")
            for name in "abcdef"
        }
        self.initial = [
            row("a", "4 + 3 = 7.", "missing_final_answer"),
            row("b", "5.", "missing_final_answer"),
            row("c", "x = 3 + 4", "missing_final_answer"),
            row("d", "5.", "missing_final_answer"),
            row("e", "still working", "missing_final_answer", finish="length"),
            row("f", "6.", "missing_final_answer"),
        ]
        self.final = [
            row("a", "4 + 3 = 7\n#### 7", "correct_answer"),
            row("b", "#### 7", "correct_answer"),
            row("c", "x = 7\n#### 7", "correct_answer"),
            row("d", "#### 5", "incorrect_answer"),
            row("e", "#### 4", "incorrect_answer"),
            row("f", "6 remains.", "missing_final_answer"),
        ]
        self.annotations = [
            annotation("a", "terminal", "7", "4 + 3 = 7", "same_value_and_derivation", "4 + 3 = 7\n#### 7"),
            annotation("b", "terminal", "5", "5.", "changed_answer_value", "#### 7"),
            annotation("c", "unresolved", None, "x = 3 + 4", "completed_answer", "x = 7\n#### 7"),
            annotation("d", "terminal", "5", "5."),
            annotation("e", "unresolved", None, "still working"),
        ]

    def test_classifies_selected_changes_without_treating_other_pairs_as_math_errors(self):
        paired, summary = analyze_pairs(
            self.initial, self.final, self.annotations, self.problems,
        )
        self.assertEqual(summary["target_counts"], {
            "new_wrong_final": 2, "newly_correct": 3,
        })
        self.assertEqual(summary["gain_mechanism_counts"], {
            "changed_answer_value": 1, "completed_answer": 1,
            "same_value_and_derivation": 1,
        })
        self.assertEqual(summary["new_wrong_final_initial_relation_counts"], {
            "same_wrong_value": 1, "unresolved_initial": 1,
        })
        self.assertIsNone(paired[-1]["annotation"])
        self.assertIsNone(target_class(self.initial[-1], self.final[-1]))

    def test_rejects_missing_or_unsupported_annotations(self):
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            analyze_pairs(self.initial, self.final, self.annotations[:-1], self.problems)
        changed = [dict(item) for item in self.annotations]
        changed[0]["initial_candidate"] = "8"
        with self.assertRaisesRegex(ValueError, "absent from evidence"):
            analyze_pairs(self.initial, self.final, changed, self.problems)
        changed = [dict(item) for item in self.annotations]
        changed[0]["gain_mechanism"] = "changed_answer_value"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            analyze_pairs(self.initial, self.final, changed, self.problems)
        changed = [dict(item) for item in self.annotations]
        changed[0]["final_evidence"] = "invented"
        with self.assertRaisesRegex(ValueError, "final evidence"):
            analyze_pairs(self.initial, self.final, changed, self.problems)

    def test_rechecks_saved_strict_rewards(self):
        changed = [dict(item) for item in self.final]
        changed[0]["reward_reason"] = "incorrect_answer"
        with self.assertRaisesRegex(ValueError, "saved verifier result differs"):
            analyze_pairs(self.initial, changed, self.annotations, self.problems)


if __name__ == "__main__":
    unittest.main()
