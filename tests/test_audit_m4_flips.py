import unittest

from scripts.audit_m4_flips import to_shared_annotations


class M4FlipAuditTests(unittest.TestCase):
    def test_converts_m4_arm_names_and_unresolved_class(self):
        result = to_shared_annotations([
            {"prompt_id": "p1", "failed_arm": "m0", "failure_class": "unresolved",
             "candidate": None, "evidence": "unfinished"},
            {"prompt_id": "p2", "failed_arm": "m3", "failure_class": "wrong_value",
             "candidate": "5", "evidence": "#### 5"},
        ])
        self.assertEqual(result[0]["failed_arm"], "baseline")
        self.assertEqual(result[0]["failure_class"], "truncated_unresolved")
        self.assertEqual(result[1]["failed_arm"], "two_step_example")
        self.assertEqual(result[1]["failure_class"], "wrong_value")

    def test_rejects_unsupported_annotation(self):
        with self.assertRaises(ValueError):
            to_shared_annotations([
                {"prompt_id": "p1", "failed_arm": "m2", "failure_class": "wrong_value",
                 "candidate": "5", "evidence": "5"},
            ])


if __name__ == "__main__":
    unittest.main()
