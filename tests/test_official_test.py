import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_official_test import (
    evaluate_to_artifacts, validate_candidate, verify_checkpoint,
)
from src.data.dataset import MathProblem


class OfficialTestTests(unittest.TestCase):
    def test_checkpoint_integrity_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            checkpoint.mkdir()
            weights = checkpoint / "weights.bin"
            weights.write_bytes(b"original")
            checksums = Path(tmp) / "checksums.txt"
            checksums.write_text(f"{hashlib.sha256(weights.read_bytes()).hexdigest()}  {weights}\n")
            self.assertEqual(verify_checkpoint(checksums, checkpoint),
                             hashlib.sha256(checksums.read_bytes()).hexdigest())
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                verify_checkpoint(checksums, checkpoint)

    def test_candidate_must_match_completed_run_and_checkpoint(self):
        run = {
            "status": "completed", "config_sha256": "config-hash",
            "dataset_revision": "dataset-revision", "initial_model_revision": "model-revision",
            "steps": [{"policy_after": "step-000002", "checkpoint_dir": "/tmp/run/checkpoints/step-000002"}],
        }
        config = {
            "data": {"revision": "dataset-revision"},
            "model": {"revision": "model-revision"},
            "evaluation": {"do_sample": False, "run_official_test_once": True},
        }
        validate_candidate(run, config, "config-hash", Path("/tmp/run/checkpoints/step-000002"))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_candidate(run, config, "config-hash", Path("/tmp/other"))
        with self.assertRaisesRegex(ValueError, "config"):
            validate_candidate(run, config, "wrong", Path("/tmp/run/checkpoints/step-000002"))

    def test_writes_exactly_one_evaluation_and_refuses_existing_output(self):
        problems = [MathProblem("gsm8k/main/test/00000", "Q1?", "#### 1"),
                    MathProblem("gsm8k/main/test/00001", "Q2?", "#### 2")]
        calls = []

        def fake_evaluate(model, tokenizer, received, config):
            calls.append(tuple(item.prompt_id for item in received))
            return ([
                {"prompt_id": problems[0].prompt_id, "response_text": "#### 1",
                 "response_tokens": 3, "finish_reason": "eos", "reward": 1.0,
                 "reward_reason": "correct_answer"},
                {"prompt_id": problems[1].prompt_id, "response_text": "#### 0",
                 "response_tokens": 3, "finish_reason": "eos", "reward": 0.0,
                 "reward_reason": "incorrect_answer"},
            ], {"question_count": 2, "correct_count": 1, "accuracy": 0.5,
                "reason_counts": {"correct_answer": 1, "incorrect_answer": 1},
                "truncation_count": 0, "mean_response_tokens": 3.0})

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "official"
            result = evaluate_to_artifacts(
                object(), object(), problems, {"evaluation": {"do_sample": False}},
                output, {"source_split": "test", "checkpoint_policy_version": "step-000002"},
                evaluation_fn=fake_evaluate,
            )
            self.assertEqual(calls, [tuple(item.prompt_id for item in problems)])
            self.assertEqual(result["metrics"]["correct_count"], 1)
            self.assertEqual(len((output / "test-rows.jsonl").read_text().splitlines()), 2)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["source_split"], "test")
            with self.assertRaises(FileExistsError):
                evaluate_to_artifacts(object(), object(), problems, {}, output, {},
                                      evaluation_fn=fake_evaluate)
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
