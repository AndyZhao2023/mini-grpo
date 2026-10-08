import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.find_trainable_group import ScanResult
from scripts.run_m0 import run_cycles
from src.data.dataset import MathProblem, TrainValidationSplit
from src.data.schema import Trajectory
from src.grpo.trainer import StepMetrics


class FakeModel:
    def __init__(self):
        self.policy_step = 0

    def save_pretrained(self, path, **kwargs):
        (Path(path) / "weights.txt").write_text(str(self.policy_step))


class FakeTokenizer:
    pad_token_id = 9
    eos_token_id = 9

    def save_pretrained(self, path):
        (Path(path) / "tokenizer.txt").write_text("fake")


CONFIG = {
    "seed": 42,
    "model": {"device": "cpu", "revision": "base", "tokenizer_revision": "base"},
    "data": {"id": "openai/gsm8k", "name": "main", "revision": "dataset", "split_seed": 42},
    "prompt": {"system": "Solve and end with #### <number>."},
    "generation": {"group_size": 2},
    "algorithm": {"advantage_std": "population", "advantage_epsilon": 1e-8,
                  "clip_epsilon": 0.2, "loss_reduction": "response_then_group_then_batch"},
    "optimizer": {"max_grad_norm": 1.0},
    "run": {"update_steps": 2, "max_prompts_per_step": 2,
            "order": "original", "max_post_approx_kl_old": 0.05},
}


def rollout(problem, policy_version):
    return [Trajectory(
        schema_version=1, prompt_id=problem.prompt_id,
        group_id=f"{policy_version}/{problem.prompt_id}", sample_index=i,
        group_size=2, prompt_text=problem.question, prompt_token_ids=[1, 2],
        response_text="#### 5" if i == 0 else "#### 0",
        response_token_ids=[3 + i, 9], response_mask=[1, 1],
        finish_reason="eos", policy_version=policy_version,
        model_revision="base", tokenizer_revision="base",
        old_log_probs=[-1.0, -1.0],
    ) for i in range(2)]


def selected_scan(problems, rollout_fn, algorithm_config, max_prompts):
    group = rollout_fn(problems[0])
    scored = [replace(item, reward=float(1 - i), reward_reason="scored")
              for i, item in enumerate(group)]
    advantaged = [replace(item, advantage=1.0 if i == 0 else -1.0)
                  for i, item in enumerate(scored)]
    attempt = {"prompt_id": problems[0].prompt_id, "rewards": [1.0, 0.0],
               "reason_counts": {"scored": 2}, "response_lengths": [2, 2],
               "truncation_count": 0, "zero_variance": False}
    return ScanResult([attempt], group, scored, advantaged)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.split = TrainValidationSplit(
            training=(MathProblem("train/1", "Q1?", "#### 5"),
                      MathProblem("train/2", "Q2?", "#### 5")),
            validation=(MathProblem("val/1", "V?", "#### 5"),),
            source_sha256="source-hash",
        )
        self.model, self.tokenizer = FakeModel(), FakeTokenizer()
        self.evaluated_steps = []

    def evaluate(self, model, tokenizer, problems, config):
        self.evaluated_steps.append(model.policy_step)
        return ([{"prompt_id": problems[0].prompt_id, "reward": float(model.policy_step > 0)}],
                {"question_count": 1, "correct_count": int(model.policy_step > 0),
                 "accuracy": float(model.policy_step > 0)})

    def update(self, model, optimizer, group, **kwargs):
        self.assertEqual(group[0].policy_version, f"step-{model.policy_step:06d}")
        model.policy_step += 1
        return StepMetrics(0.0, -0.01, 0.0, 0.01, 0.0, 0.0, 0.001, 2.0, 4)

    def test_two_fresh_rollouts_and_fixed_validation(self):
        seen = []

        def sample(model, tokenizer, problem, config, version):
            seen.append((model.policy_step, version, problem.prompt_id))
            return rollout(problem, version)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            result = run_cycles(
                self.model, self.tokenizer, object(), self.split, CONFIG, output,
                config_sha256="config-hash", sample_fn=sample,
                scan_fn=selected_scan, update_fn=self.update, evaluation_fn=self.evaluate,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completed_steps"], 2)
            self.assertEqual(self.evaluated_steps, [0, 2])
            self.assertEqual([item[1] for item in seen], ["step-000000", "step-000001"])
            self.assertNotEqual(seen[0][2], seen[1][2])
            self.assertEqual((output / "summary.json").exists(), True)
            self.assertEqual((output / "dataset-split.json").exists(), True)
            self.assertEqual((output / "steps/step-000002/step.json").exists(), True)
            self.assertEqual((output / "checkpoints/step-000002/weights.txt").read_text(), "2")

    def test_no_mixed_group_records_partial_run_without_update(self):
        def no_signal(problems, rollout_fn, algorithm_config, max_prompts):
            return ScanResult([{"prompt_id": problems[0].prompt_id,
                                "rewards": [0.0, 0.0], "reason_counts": {"wrong": 2},
                                "response_lengths": [2, 2], "truncation_count": 0,
                                "zero_variance": True}], None, None, None)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            result = run_cycles(
                self.model, self.tokenizer, object(), self.split, CONFIG, output,
                config_sha256="config-hash", scan_fn=no_signal,
                update_fn=lambda *args, **kwargs: self.fail("must not update"),
                evaluation_fn=self.evaluate,
            )
            self.assertEqual(result["status"], "incomplete_no_mixed_group")
            self.assertEqual(result["completed_steps"], 0)
            self.assertEqual(self.evaluated_steps, [0, 0])
            self.assertFalse((output / "checkpoints").exists())
            self.assertTrue((output / "steps/step-000001/scan.json").exists())

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                run_cycles(self.model, self.tokenizer, object(), self.split, CONFIG,
                           Path(tmp), config_sha256="config-hash",
                           evaluation_fn=self.evaluate)

    def test_large_first_step_drift_stops_before_second_update(self):
        def large_update(model, optimizer, group, **kwargs):
            self.assertEqual(model.policy_step, 0)
            model.policy_step += 1
            return StepMetrics(0.0, -0.01, 0.0, 0.5, 0.0, 0.25, 0.2, 2.0, 4)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            result = run_cycles(
                self.model, self.tokenizer, object(), self.split, CONFIG, output,
                config_sha256="config-hash", scan_fn=selected_scan,
                sample_fn=lambda model, tokenizer, problem, config, version: rollout(problem, version),
                update_fn=large_update, evaluation_fn=self.evaluate,
            )
            self.assertEqual(result["status"], "incomplete_post_step_drift")
            self.assertEqual(result["completed_steps"], 1)
            self.assertEqual(self.evaluated_steps, [0, 1])
            self.assertFalse((output / "checkpoints/step-000002").exists())

    def test_rejects_training_validation_overlap(self):
        bad_split = replace(self.split, validation=(self.split.training[0],))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "disjoint"):
                run_cycles(self.model, self.tokenizer, object(), bad_split, CONFIG,
                           Path(tmp) / "run", config_sha256="config-hash",
                           evaluation_fn=self.evaluate)


if __name__ == "__main__":
    unittest.main()
