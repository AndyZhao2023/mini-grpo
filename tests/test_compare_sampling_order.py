import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_sampling_order import question_seed, run_arm, select_arms
from src.data.dataset import MathProblem
from src.data.schema import Trajectory


def fake_rollout(problem, responses):
    return [
        Trajectory(
            schema_version=1, prompt_id=problem.prompt_id,
            group_id=f"step-000002/{problem.prompt_id}", sample_index=index,
            group_size=len(responses), prompt_text=problem.question,
            prompt_token_ids=[1, 2], response_text=response,
            response_token_ids=[3, 4], response_mask=[1, 1],
            finish_reason="eos", policy_version="step-000002",
            model_revision="model", tokenizer_revision="tokenizer",
            old_log_probs=[-1.0, -1.0],
        )
        for index, response in enumerate(responses)
    ]


class SamplingOrderTests(unittest.TestCase):
    def test_selection_is_stable_disjoint_and_excludes_prior_questions(self):
        training = [
            MathProblem(f"train/{index:03d}", "word " * (index + 1), "#### 5")
            for index in range(10)
        ]
        selected = select_arms(training, {"train/000"}, 3, "fixed-salt")
        self.assertEqual(
            [item.prompt_id for item in selected["shortest"]],
            ["train/001", "train/002", "train/003"],
        )
        self.assertEqual(selected, select_arms(training, {"train/000"}, 3, "fixed-salt"))
        self.assertFalse({item.prompt_id for item in selected["shortest"]} & {
            item.prompt_id for item in selected["hash_random"]
        })
        self.assertNotIn("train/000", {
            item.prompt_id for arm in selected.values() for item in arm
        })
        with self.assertRaisesRegex(ValueError, "not enough unused"):
            select_arms(training, {"train/000"}, 5, "fixed-salt")

    def test_question_seed_is_independent_of_arm_order(self):
        self.assertEqual(question_seed(42, "train/001"), question_seed(42, "train/001"))
        self.assertNotEqual(question_seed(42, "train/001"), question_seed(42, "train/002"))

    def test_runs_full_budget_including_zero_reward_groups(self):
        problems = [
            MathProblem("train/001", "short question", "#### 5"),
            MathProblem("train/002", "another question", "#### 5"),
        ]
        visited = []

        def rollout(_model, _tokenizer, problem, _config, policy_version):
            self.assertEqual(policy_version, "step-000002")
            visited.append(problem.prompt_id)
            responses = ["wrong", "wrong"] if problem == problems[0] else ["#### 5", "wrong"]
            return fake_rollout(problem, responses)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.jsonl"
            summary = run_arm(
                None, None, problems, {"seed": 42, "generation": {"group_size": 2}},
                path, rollout_fn=rollout,
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(visited, [item.prompt_id for item in problems])
        self.assertEqual([item["mixed_reward"] for item in rows], [False, True])
        self.assertEqual(summary["mixed_group_count"], 1)
        self.assertEqual(summary["correct_response_count"], 1)
        self.assertEqual(summary["sample_count"], 4)


if __name__ == "__main__":
    unittest.main()
