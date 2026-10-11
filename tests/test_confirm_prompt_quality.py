import unittest

from scripts.compare_prompt_quality import select_development_and_reserve
from scripts.confirm_prompt_quality import validate_reserved_selection
from src.data.dataset import MathProblem, TrainValidationSplit


class PromptConfirmationTests(unittest.TestCase):
    def test_reserved_ids_must_match_pre_evaluation_selection(self):
        training = tuple(MathProblem(f"train/{i}", f"q{i}", "#### 1") for i in range(10))
        split = TrainValidationSplit(
            training=training,
            validation=(MathProblem("train/10", "held out", "#### 1"),),
            source_sha256="source-hash",
        )
        spec = {"selection": {"salt": "fixed", "development_count": 2, "reserved_count": 3}}
        excluded = {"train/0"}
        development, reserved = select_development_and_reserve(training, excluded, "fixed", 2, 3)
        manifest = {
            "source_split": "train", "source_sha256": "source-hash",
            "development_prompt_ids": [item.prompt_id for item in development],
            "reserved_prompt_ids": [item.prompt_id for item in reserved],
            "excluded_prior_prompt_count": 1,
        }
        self.assertEqual(validate_reserved_selection(manifest, spec, split, excluded), reserved)
        tampered = {**manifest, "reserved_prompt_ids": list(reversed(manifest["reserved_prompt_ids"]))}
        with self.assertRaisesRegex(ValueError, "reserved questions differ"):
            validate_reserved_selection(tampered, spec, split, excluded)


if __name__ == "__main__":
    unittest.main()
