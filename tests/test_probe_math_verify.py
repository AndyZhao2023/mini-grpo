import unittest

from sympy import Integer, Rational, Symbol

from scripts.probe_math_verify import compare_to_manual, numeric_candidate


class MathVerifyProbeTests(unittest.TestCase):
    def test_numeric_filter_rejects_fallbacks_symbols_and_conflicts(self) -> None:
        self.assertEqual(numeric_candidate([Integer(8), "8"]), Integer(8))
        self.assertIsNone(numeric_candidate(["8"]))
        self.assertIsNone(numeric_candidate([Symbol("x")]))
        self.assertIsNone(numeric_candidate([Rational(79, 100), Integer(79)]))

    def test_manual_comparison_preserves_abstention(self) -> None:
        def exact_verify(gold, prediction):
            return gold[0] == prediction[0]

        self.assertEqual(
            compare_to_manual(Integer(8), {"candidate": "8"}, exact_verify), "agreed"
        )
        self.assertEqual(
            compare_to_manual(Integer(5), {"candidate": "8"}, exact_verify), "disagreed"
        )
        self.assertEqual(
            compare_to_manual(None, {"candidate": "8"}, exact_verify), "missed"
        )
        self.assertEqual(
            compare_to_manual(Integer(3), {"candidate": None}, exact_verify),
            "unknown_extracted",
        )


if __name__ == "__main__":
    unittest.main()
