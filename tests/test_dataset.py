import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.data.dataset import build_manifest, build_prompt_messages, load_pinned_train, split_train_rows


def fixture_rows(count: int = 10) -> list[dict[str, str]]:
    return [
        {"question": f"Question {index}?", "answer": f"PRIVATE_WORK_{index}\n#### {index}"}
        for index in range(count)
    ]


DATA_CONFIG = {
    "id": "openai/gsm8k",
    "name": "main",
    "revision": "a" * 40,
    "expected_train_size": 10,
    "expected_train_sha256": "8f0c0afca7b4b02ca77806f5fe818ead9f88575ed0c11b1cbe1c7319756f21fb",
    "validation_size": 3,
    "split_seed": 42,
}


class DatasetTests(unittest.TestCase):
    def test_fixed_disjoint_split_keeps_original_row_ids(self) -> None:
        first = split_train_rows(fixture_rows(), validation_size=3, seed=42)
        second = split_train_rows(fixture_rows(), validation_size=3, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(
            [row.prompt_id for row in first.validation],
            ["gsm8k/main/train/00000", "gsm8k/main/train/00001", "gsm8k/main/train/00006"],
        )
        train_ids = {row.prompt_id for row in first.training}
        validation_ids = {row.prompt_id for row in first.validation}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(len(train_ids | validation_ids), 10)

    def test_prompt_never_includes_reference_answer(self) -> None:
        problem = split_train_rows(fixture_rows(), validation_size=3, seed=42).training[0]
        messages = build_prompt_messages(problem, "Solve it")
        self.assertEqual(messages, [
            {"role": "system", "content": "Solve it"},
            {"role": "user", "content": problem.question},
        ])
        self.assertNotIn("PRIVATE_WORK", str(messages))

    def test_manifest_records_split_without_reference_answers(self) -> None:
        split = split_train_rows(fixture_rows(), validation_size=3, seed=42)
        manifest = build_manifest(split, DATA_CONFIG, "Solve it", preview_count=2)
        self.assertEqual(manifest["source_split"], "train")
        self.assertEqual(manifest["training_count"], 7)
        self.assertEqual(manifest["validation_count"], 3)
        self.assertEqual(len(manifest["validation_prompt_ids"]), 3)
        self.assertEqual(len(manifest["source_sha256"]), 64)
        self.assertNotIn("PRIVATE_WORK", str(manifest))

    def test_loader_requests_only_pinned_train_split(self) -> None:
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return fixture_rows()

        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=fake_load_dataset)}):
            split = load_pinned_train(DATA_CONFIG)
        self.assertEqual(len(split.training), 7)
        self.assertEqual(calls, [(('openai/gsm8k',), {
            "name": "main",
            "data_files": {"train": "main/train-00000-of-00001.parquet"},
            "split": "train", "revision": "a" * 40,
            "verification_mode": "no_checks",
        })])

    def test_loader_rejects_unexpected_train_size(self) -> None:
        def fake_load_dataset(*args, **kwargs):
            return fixture_rows(9)

        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=fake_load_dataset)}):
            with self.assertRaisesRegex(ValueError, "expected_train_size"):
                load_pinned_train(DATA_CONFIG)

    def test_loader_rejects_changed_source_content(self) -> None:
        changed = fixture_rows()
        changed[0] = {"question": "Different question?", "answer": changed[0]["answer"]}

        def fake_load_dataset(*args, **kwargs):
            return changed

        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=fake_load_dataset)}):
            with self.assertRaisesRegex(ValueError, "expected_train_sha256"):
                load_pinned_train(DATA_CONFIG)

    def test_invalid_rows_and_split_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation_size"):
            split_train_rows(fixture_rows(3), validation_size=3, seed=42)
        with self.assertRaisesRegex(ValueError, "question"):
            split_train_rows([{"question": "", "answer": "#### 1"}, *fixture_rows(3)], 1, 42)
        with self.assertRaisesRegex(ValueError, "answer"):
            split_train_rows([{"question": "Q?", "answer": ""}, *fixture_rows(3)], 1, 42)


if __name__ == "__main__":
    unittest.main()
