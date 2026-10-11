import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_prompt_quality import (
    collect_prior_ids, compare_paired, select_development_and_reserve, summarize,
)
from src.data.dataset import MathProblem


class PromptQualityTests(unittest.TestCase):
    def test_new_sets_are_stable_disjoint_and_exclude_seen_training_ids(self):
        training = tuple(MathProblem(f"train/{i}", f"question {i}", "#### 1") for i in range(12))
        first = select_development_and_reserve(training, {"train/3", "train/5"}, "fixed", 3, 4)
        second = select_development_and_reserve(tuple(reversed(training)), {"train/3", "train/5"}, "fixed", 3, 4)
        self.assertEqual(first, second)
        dev, reserved = first
        self.assertEqual(len(dev), 3)
        self.assertEqual(len(reserved), 4)
        self.assertFalse({item.prompt_id for item in dev} & {item.prompt_id for item in reserved})
        self.assertFalse({item.prompt_id for item in dev + reserved} & {"train/3", "train/5"})

    def test_prior_evidence_is_bound_to_dataset_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "manifest.json"
            scan = Path(folder) / "scan.json"
            manifest.write_text(json.dumps({
                "source_split": "train", "source_sha256": "abc", "prompt_ids": ["train/2"],
            }))
            scan.write_text(json.dumps({
                "source_sha256": "abc", "attempts": [{"prompt_id": "train/1"}],
            }))
            spec = {"prior_diagnostic_manifest": str(manifest), "prior_scan_files": [str(scan)]}
            ids, hashes = collect_prior_ids(spec, "abc")
            self.assertEqual(ids, {"train/1", "train/2"})
            self.assertEqual(len(hashes), 2)
            with self.assertRaisesRegex(ValueError, "different dataset hash"):
                collect_prior_ids(spec, "wrong")

    def test_paired_counts_keep_gains_and_regressions_separate(self):
        baseline = [
            {"prompt_id": "a", "reward": 1.0, "response_text": "a", "reward_reason": "correct_answer", "finish_reason": "eos", "response_tokens": 4},
            {"prompt_id": "b", "reward": 0.0, "response_text": "b", "reward_reason": "missing_final_answer", "finish_reason": "length", "response_tokens": 8},
        ]
        variant = [
            {"prompt_id": "a", "reward": 0.0, "response_text": "changed", "reward_reason": "incorrect_answer", "finish_reason": "eos", "response_tokens": 5},
            {"prompt_id": "b", "reward": 1.0, "response_text": "#### 2", "reward_reason": "correct_answer", "finish_reason": "eos", "response_tokens": 6},
        ]
        self.assertEqual(compare_paired(baseline, variant)["newly_strict_correct"], 1)
        self.assertEqual(compare_paired(baseline, variant)["lost_strict_correct"], 1)
        self.assertEqual(summarize(baseline)["truncated_count"], 1)
        with self.assertRaisesRegex(ValueError, "same ordered questions"):
            compare_paired(baseline, list(reversed(variant)))


if __name__ == "__main__":
    unittest.main()
