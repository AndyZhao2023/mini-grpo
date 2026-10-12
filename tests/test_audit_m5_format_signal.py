import unittest

from scripts.audit_m5_format_signal import summarize, validate_annotations


TARGET = {"arm": "shortest", "prompt_id": "train/1", "sample_index": 0}
RESPONSE = {"finish_reason": "eos", "response_text": "Answer: 1,200.",
            "reward_reason": "missing_final_answer"}
ANNOTATION = {**TARGET, "kind": "terminal", "candidate": "1200",
              "evidence": "Answer: 1,200."}


class M5FormatSignalAuditTests(unittest.TestCase):
    def test_annotation_requires_frozen_coverage_and_grounded_candidate(self) -> None:
        responses = {("shortest", "train/1", 0): RESPONSE}
        self.assertEqual(
            validate_annotations([TARGET], [ANNOTATION], responses)[("shortest", "train/1", 0)],
            ANNOTATION,
        )
        with self.assertRaisesRegex(ValueError, "ordered, unique"):
            validate_annotations([TARGET], [], responses)
        with self.assertRaisesRegex(ValueError, "does not occur"):
            validate_annotations([TARGET], [{**ANNOTATION, "candidate": "1201"}], responses)
        with self.assertRaisesRegex(ValueError, "evidence does not occur"):
            validate_annotations([TARGET], [{**ANNOTATION, "evidence": "not in answer"}], responses)
        with self.assertRaisesRegex(ValueError, "invalid-format completed"):
            validate_annotations(
                [TARGET], [ANNOTATION],
                {("shortest", "train/1", 0): {**RESPONSE, "response_text": "#### 1200"}},
            )

    def test_summary_identifies_wrong_positive_and_correct_negative(self) -> None:
        rows = [{
            "arm": "shortest", "prompt_id": "train/1", "sample_index": index,
            "comparison": comparison, "candidate_kind": kind,
            "naive_advantage": advantage,
        } for index, (comparison, kind, advantage) in enumerate((
            ("different_from_reference", "valid_final", 1.732),
            ("matches_reference", "terminal", -0.577),
            ("unresolved", "length_abstain", -0.577),
            ("different_from_reference", "terminal", -0.577),
        ))]
        result = summarize(rows, 1)
        self.assertEqual(result["counts"]["positive_advantage_known_wrong"], 1)
        self.assertEqual(result["counts"]["negative_advantage_known_correct"], 1)
        self.assertEqual(result["counts"]["groups_with_rank_inversion"], 1)


if __name__ == "__main__":
    unittest.main()
