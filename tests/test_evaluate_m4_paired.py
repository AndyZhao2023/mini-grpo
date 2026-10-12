import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_m4_paired import (
    load_verified_prefix, select_new_questions, summarize_paired,
    validate_lineage,
)
from src.data.dataset import MathProblem


def problem(number, answer="7"):
    return MathProblem(f"train/{number}", f"question {number}", f"#### {answer}")


def row(prompt_id, text, reason):
    return {
        "prompt_id": prompt_id, "response_text": text,
        "response_tokens": 3, "finish_reason": "eos",
        "reward": float(reason == "correct_answer"), "reward_reason": reason,
    }


class M4SelectionTests(unittest.TestCase):
    def test_selection_is_stable_and_excludes_every_prior_id(self):
        questions = tuple(problem(index) for index in range(30))
        a = select_new_questions(questions, {"train/0", "train/1"}, "fixed-salt", 8)
        b = select_new_questions(tuple(reversed(questions)), {"train/0", "train/1"}, "fixed-salt", 8)
        self.assertEqual([item.prompt_id for item in a], [item.prompt_id for item in b])
        self.assertEqual(len(a), 8)
        self.assertFalse({item.prompt_id for item in a} & {"train/0", "train/1"})

    def test_selection_rejects_invalid_budget_or_unknown_exclusion(self):
        questions = tuple(problem(index) for index in range(3))
        with self.assertRaises(ValueError):
            select_new_questions(questions, {"train/99"}, "salt", 1)
        with self.assertRaises(ValueError):
            select_new_questions(questions, set(), "salt", 4)
        with self.assertRaises(ValueError):
            select_new_questions(questions, set(), "", 1)

    def test_lineage_rejects_a_changed_prompt_or_wrong_final_checkpoint(self):
        m0_config = {"data": {}, "prompt": {"system": "fixed"},
                     "evaluation": {"do_sample": False, "max_new_tokens": 128,
                                    "run_official_test_once": True},
                     "model": {"id": "model", "tokenizer_revision": "tok", "device": "mps", "dtype": "float32"}}
        m3_config = {**m0_config, "model": {**m0_config["model"], "revision": "m0-checkpoint"},
                     "evaluation": {**m0_config["evaluation"], "run_official_test_once": False}}
        m0_run = {"status": "completed", "completed_steps": 2, "config_sha256": "m0-config",
                  "source_sha256": "data", "steps": [{"checkpoint_dir": "m0-checkpoint"}]}
        m3_run = {"status": "completed", "completed_steps": 10, "config_sha256": "m3-config",
                  "source_sha256": "data", "initial_model_revision": "m0-checkpoint",
                  "steps": [{"checkpoint_dir": "m3-checkpoint"}]}
        manifest = {"config_sha256": "m3-config", "source_sha256": "data",
                    "source_run_summary_sha256": "m0-summary", "starting_checkpoint": "m0-checkpoint"}
        args = (m0_config, m3_config, m0_run, m3_run, manifest,
                "m0-config", "m3-config", "m0-summary", "data")
        self.assertEqual(validate_lineage(*args), (Path("m0-checkpoint"), Path("m3-checkpoint")))
        bad = {**m3_config, "prompt": {"system": "changed"}}
        with self.assertRaises(ValueError):
            validate_lineage(m0_config, bad, *args[2:])
        bad_run = {**m3_run, "steps": [{"checkpoint_dir": "m0-checkpoint"}]}
        with self.assertRaises(ValueError):
            validate_lineage(m0_config, m3_config, m0_run, bad_run, *args[4:])


class M4EvidenceTests(unittest.TestCase):
    def test_verified_prefix_rejects_changed_reward_and_wrong_order(self):
        questions = (problem(0), problem(1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"prompt_id":"train/0","response_text":"#### 7","response_tokens":3,"finish_reason":"eos","reward":1.0,"reward_reason":"correct_answer"}\n')
            self.assertEqual(len(load_verified_prefix(path, questions)), 1)
            path.write_text(path.read_text().replace('"reward":1.0', '"reward":0.0'))
            with self.assertRaises(ValueError):
                load_verified_prefix(path, questions)
            path.write_text(path.read_text().replace('"prompt_id":"train/0"', '"prompt_id":"train/1"'))
            with self.assertRaises(ValueError):
                load_verified_prefix(path, questions)

    def test_paired_summary_counts_all_four_cells_and_rechecks_rewards(self):
        questions = (problem(0), problem(1), problem(2), problem(3))
        baseline = [
            row("train/0", "#### 7", "correct_answer"),
            row("train/1", "no final", "missing_final_answer"),
            row("train/2", "#### 7", "correct_answer"),
            row("train/3", "#### 5", "incorrect_answer"),
        ]
        final = [
            row("train/0", "#### 7", "correct_answer"),
            row("train/1", "#### 7", "correct_answer"),
            row("train/2", "#### 5", "incorrect_answer"),
            row("train/3", "#### 5", "incorrect_answer"),
        ]
        paired, summary = summarize_paired(baseline, final, questions)
        self.assertEqual(summary["strict_change_counts"], {
            "correct_in_both": 1, "incorrect_in_both": 1,
            "newly_correct": 1, "lost_correct": 1,
        })
        self.assertEqual(len(paired), 4)
        bad = [dict(item) for item in final]
        bad[0]["reward_reason"] = "incorrect_answer"
        with self.assertRaises(ValueError):
            summarize_paired(baseline, bad, questions)


if __name__ == "__main__":
    unittest.main()
