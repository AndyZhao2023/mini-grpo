import unittest

from scripts.run_warm_start import select_fresh_split
from src.data.dataset import MathProblem, TrainValidationSplit


class WarmStartSelectionTests(unittest.TestCase):
    def setUp(self):
        self.split = TrainValidationSplit(
            training=tuple(MathProblem(f"train/{i}", "word " * (i + 1), "#### 1")
                           for i in range(8)),
            validation=(MathProblem("validation/0", "held-out", "#### 1"),),
            source_sha256="source-hash",
        )

    def test_reserves_disjoint_unseen_questions_deterministically(self):
        first = select_fresh_split(
            self.split, {"train/0"}, training_pool_size=3,
            validation_size=2, validation_salt="fixed",
        )
        second = select_fresh_split(
            self.split, {"train/0"}, training_pool_size=3,
            validation_size=2, validation_salt="fixed",
        )
        self.assertEqual(first, second)
        self.assertEqual([item.prompt_id for item in first.training],
                         ["train/1", "train/2", "train/3"])
        self.assertEqual(len(first.validation), 2)
        self.assertFalse({item.prompt_id for item in first.training}
                         & {item.prompt_id for item in first.validation})
        self.assertNotIn("train/0", {item.prompt_id for item in first.validation})
        self.assertNotIn("validation/0", {item.prompt_id for item in first.validation})

    def test_rejects_unknown_exclusion_and_oversized_selection(self):
        with self.assertRaisesRegex(ValueError, "excluded IDs"):
            select_fresh_split(self.split, {"validation/0"}, training_pool_size=3,
                               validation_size=2, validation_salt="fixed")
        with self.assertRaisesRegex(ValueError, "not enough"):
            select_fresh_split(self.split, {"train/0"}, training_pool_size=6,
                               validation_size=2, validation_salt="fixed")


if __name__ == "__main__":
    unittest.main()
