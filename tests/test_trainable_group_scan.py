import unittest

from scripts.find_trainable_group import order_training_problems, scan_for_mixed_group
from src.data.dataset import MathProblem
from src.data.schema import Trajectory


ALGORITHM = {"advantage_std": "population", "advantage_epsilon": 1e-8}


def make_rollout(problem: MathProblem, responses: list[str]) -> list[Trajectory]:
    return [
        Trajectory(
            schema_version=1,
            prompt_id=problem.prompt_id,
            group_id=f"step-000000/{problem.prompt_id}",
            sample_index=index,
            group_size=len(responses),
            prompt_text=problem.question,
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


class ScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problems = [
            MathProblem(f"gsm8k/main/train/{index:05d}", f"Question {index}?", "#### 5")
            for index in range(3)
        ]

    def test_stops_at_first_mixed_group_and_normalizes_it(self) -> None:
        visited = []

        def rollout(problem: MathProblem) -> list[Trajectory]:
            visited.append(problem.prompt_id)
            answers = ["no final answer", "#### 0"] if problem == self.problems[0] else ["#### 5", "#### 0"]
            return make_rollout(problem, answers)

        result = scan_for_mixed_group(self.problems, rollout, ALGORITHM, 3)
        self.assertEqual(visited, [item.prompt_id for item in self.problems[:2]])
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual([item["zero_variance"] for item in result.attempts], [True, False])
        self.assertEqual([item.reward for item in result.selected_scored], [1.0, 0.0])
        self.assertAlmostEqual(result.selected_advantaged[0].advantage, 1.0, places=6)
        self.assertAlmostEqual(result.selected_advantaged[1].advantage, -1.0, places=6)
        self.assertTrue(all(item.reward is None for item in result.selected_rollout))

    def test_reports_no_selection_with_bounded_scan(self) -> None:
        result = scan_for_mixed_group(
            self.problems, lambda problem: make_rollout(problem, ["wrong", "wrong"]),
            ALGORITHM, 2,
        )
        self.assertEqual(len(result.attempts), 2)
        self.assertIsNone(result.selected_rollout)
        self.assertTrue(all(item["zero_variance"] for item in result.attempts))
        for max_prompts in (0, 4, True):
            with self.subTest(max_prompts=max_prompts):
                with self.assertRaisesRegex(ValueError, "max_prompts"):
                    scan_for_mixed_group(self.problems, lambda problem: [], ALGORITHM, max_prompts)

    def test_shortest_order_uses_only_question_text_and_stable_id(self) -> None:
        problems = [
            MathProblem("train/2", "A longer question?", "#### 1"),
            MathProblem("train/1", "Short?", "#### 9"),
            MathProblem("train/0", "Brief?", "#### 7"),
        ]
        self.assertEqual(
            [item.prompt_id for item in order_training_problems(problems, "shortest")],
            ["train/0", "train/1", "train/2"],
        )
        self.assertEqual(order_training_problems(problems, "original"), tuple(problems))
        with self.assertRaisesRegex(ValueError, "order"):
            order_training_problems(problems, "reward")


if __name__ == "__main__":
    unittest.main()
