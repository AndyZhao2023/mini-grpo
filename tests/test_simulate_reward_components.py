import unittest

from scripts.simulate_reward_components import (
    answer_candidate, group_advantages, reward_vectors,
)


class RewardSimulationTests(unittest.TestCase):
    def test_candidate_abstains_on_conflicting_or_incomplete_outputs(self) -> None:
        self.assertEqual(answer_candidate("work\n#### 27", "eos")[1], "valid_final")
        self.assertEqual(
            answer_candidate("work\n#### 27\nEnd of solution.", "eos"),
            (27, "explicit_earlier_final"),
        )
        for text, finish_reason in (
            ("work\n#### 27\nThen the answer is 28.", "eos"),
            ("work\n#### 27\nEnd of solution.", "length"),
            ("#### 27\n#### 28\nEnd of solution.", "eos"),
            ("work ends at 27", "eos"),
        ):
            with self.subTest(text=text, finish_reason=finish_reason):
                self.assertEqual(answer_candidate(text, finish_reason), (None, "unknown"))

    def test_reward_scenarios_do_not_turn_unknown_into_known_answer(self) -> None:
        labels = [
            {"strict_correct": False, "format_valid": False, "answer_value_correct": None},
            {"strict_correct": False, "format_valid": True, "answer_value_correct": False},
            {"strict_correct": False, "format_valid": False, "answer_value_correct": True},
            {"strict_correct": True, "format_valid": True, "answer_value_correct": True},
        ]
        vectors = reward_vectors(labels)
        self.assertEqual(vectors["strict"], [0, 0, 0, 1])
        self.assertEqual(vectors["strict_format_0.1"], [0, 0.1, 0, 1.1])
        self.assertEqual(vectors["evidence_gated_format_0.1"], [0, 0, 1, 1.1])

    def test_small_format_weight_still_makes_full_scale_advantages(self) -> None:
        config = {"advantage_std": "population", "advantage_epsilon": 1e-8}
        baseline, baseline_zero = group_advantages([0, 0, 0, 0], config)
        high, high_zero = group_advantages([0, 0, 0.1, 0.1], config)
        low, low_zero = group_advantages([0, 0, 0.01, 0.01], config)
        self.assertTrue(baseline_zero)
        self.assertEqual(baseline, [0] * 4)
        self.assertFalse(high_zero)
        self.assertFalse(low_zero)
        self.assertAlmostEqual(high[0], -1, places=5)
        self.assertAlmostEqual(low[0], -1, places=5)
        self.assertAlmostEqual(high[2], 1, places=5)
        self.assertAlmostEqual(low[2], 1, places=5)


if __name__ == "__main__":
    unittest.main()
