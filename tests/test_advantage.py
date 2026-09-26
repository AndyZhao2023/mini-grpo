import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from scripts.normalize_advantages import main as normalize_main
from src.data.schema import Trajectory, read_jsonl, validate_group, write_jsonl
from src.grpo.advantage import normalize_group_advantages


ALGORITHM = {"advantage_std": "population", "advantage_epsilon": 1e-8}


def scored_group(rewards: list[float]) -> list[Trajectory]:
    return [
        Trajectory(
            schema_version=1,
            prompt_id="gsm8k/main/train/00000",
            group_id="step-000000/gsm8k/main/train/00000",
            sample_index=index,
            group_size=len(rewards),
            prompt_text="How many clips?",
            prompt_token_ids=[1, 2],
            response_text=f"#### {index}",
            response_token_ids=[3, 4],
            response_mask=[1, 1],
            finish_reason="eos",
            policy_version="step-000000",
            model_revision="model-sha",
            tokenizer_revision="tokenizer-sha",
            old_log_probs=[-1.0, -2.0],
            reward=reward,
            reward_reason="correct_answer" if reward == 1.0 else "incorrect_answer",
        )
        for index, reward in enumerate(rewards)
    ]


class AdvantageTests(unittest.TestCase):
    def test_mixed_rewards_use_population_std_and_preserve_input(self) -> None:
        group = scored_group([1.0, 0.0, 0.0, 0.0])
        result = normalize_group_advantages(group, ALGORITHM)
        self.assertEqual(result.reward_mean, 0.25)
        self.assertAlmostEqual(result.reward_std, math.sqrt(3) / 4)
        self.assertFalse(result.zero_variance)
        denominator = math.sqrt(3) / 4 + 1e-8
        self.assertAlmostEqual(result.group[0].advantage, 0.75 / denominator)
        for item in result.group[1:]:
            self.assertAlmostEqual(item.advantage, -0.25 / denominator)
        self.assertAlmostEqual(sum(item.advantage for item in result.group), 0.0)
        self.assertTrue(all(item.advantage is None for item in group))
        self.assertEqual([item.old_log_probs for item in result.group], [item.old_log_probs for item in group])
        validate_group(result.group, require_complete=True)

    def test_all_equal_rewards_have_exact_zero_advantages(self) -> None:
        for reward in (0.0, 1.0):
            with self.subTest(reward=reward):
                result = normalize_group_advantages(scored_group([reward] * 4), ALGORITHM)
                self.assertTrue(result.zero_variance)
                self.assertEqual(result.reward_mean, reward)
                self.assertEqual(result.reward_std, 0.0)
                self.assertEqual([item.advantage for item in result.group], [0.0] * 4)

    def test_rejects_unscored_reused_or_incomplete_groups(self) -> None:
        group = scored_group([1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "rewards and reward reasons"):
            normalize_group_advantages([replace(group[0], reward=None), group[1]], ALGORITHM)
        with self.assertRaisesRegex(ValueError, "rewards and reward reasons"):
            normalize_group_advantages([replace(group[0], reward_reason=None), group[1]], ALGORITHM)
        with self.assertRaisesRegex(ValueError, "already has advantages"):
            normalize_group_advantages([replace(group[0], advantage=0.0), group[1]], ALGORITHM)
        with self.assertRaisesRegex(ValueError, "complete group requires old_log_probs"):
            normalize_group_advantages([replace(group[0], old_log_probs=None), group[1]], ALGORITHM)
        with self.assertRaisesRegex(ValueError, "group_size"):
            normalize_group_advantages(group[:1], ALGORITHM)

    def test_rejects_unexpected_normalization_config(self) -> None:
        group = scored_group([1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "advantage_std"):
            normalize_group_advantages(group, {**ALGORITHM, "advantage_std": "sample"})
        for epsilon in (0, -1, float("nan"), True):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(ValueError, "advantage_epsilon"):
                    normalize_group_advantages(group, {**ALGORITHM, "advantage_epsilon": epsilon})

    def test_command_round_trip_preserves_scored_file(self) -> None:
        group = scored_group([1.0, 0.0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, input_path, output_path = (
                root / "config.yaml", root / "scored.jsonl", root / "normalized.jsonl"
            )
            config_path.write_text(yaml.safe_dump({
                "generation": {"group_size": 2}, "algorithm": ALGORITHM,
                "model": {"revision": "model-sha", "tokenizer_revision": "tokenizer-sha"},
            }))
            write_jsonl(input_path, group)
            normalize_main(config_path, input_path, output_path)
            self.assertEqual(read_jsonl(input_path), group)
            normalized = read_jsonl(output_path)
            validate_group(normalized, require_complete=True)
            self.assertAlmostEqual(normalized[0].advantage, 1.0, places=6)
            self.assertAlmostEqual(normalized[1].advantage, -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
