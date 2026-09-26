"""Load the pinned GSM8K train split and select a reproducible validation set."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MathProblem:
    prompt_id: str
    question: str
    reference_answer: str  # Held for verification; never passed to the prompt builder.


@dataclass(frozen=True)
class TrainValidationSplit:
    training: tuple[MathProblem, ...]
    validation: tuple[MathProblem, ...]
    source_sha256: str


def build_prompt_messages(problem: MathProblem, system_prompt: str) -> list[dict[str, str]]:
    """The model sees the question and instruction, never the reference answer."""
    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError("system_prompt must be nonempty")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem.question},
    ]


def split_train_rows(
    rows: Sequence[Mapping[str, str]], validation_size: int, seed: int,
) -> TrainValidationSplit:
    """Select validation by stable SHA-256 rank of original train row positions.

    Hash ranking avoids dependence on a library's shuffle implementation. Both
    returned partitions retain the original dataset order and row IDs.
    """
    if type(validation_size) is not int or not 0 < validation_size < len(rows):
        raise ValueError("validation_size must be positive and smaller than train size")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    digest = hashlib.sha256()
    problems: list[MathProblem] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"train row {index} must be an object")
        question, answer = row.get("question"), row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"train row {index} has no question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"train row {index} has no answer")
        digest.update(json.dumps(
            {"question": question, "answer": answer},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(b"\n")
        problems.append(MathProblem(f"gsm8k/main/train/{index:05d}", question, answer))

    ranked = sorted(
        range(len(problems)),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode("ascii")).digest(),
    )
    validation_indices = set(ranked[:validation_size])
    return TrainValidationSplit(
        training=tuple(problem for index, problem in enumerate(problems) if index not in validation_indices),
        validation=tuple(problem for index, problem in enumerate(problems) if index in validation_indices),
        source_sha256=digest.hexdigest(),
    )


def load_pinned_train(data_config: Mapping[str, object]) -> TrainValidationSplit:
    """Fetch only GSM8K `train` from the exact dataset repository commit."""
    revision = data_config["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("data.revision must be a full 40-character commit SHA")
    if data_config["id"] != "openai/gsm8k" or data_config["name"] != "main":
        raise ValueError("this loader requires openai/gsm8k main")
    from datasets import load_dataset

    rows = load_dataset(
        data_config["id"], name=data_config["name"],
        data_files={"train": "main/train-00000-of-00001.parquet"},
        split="train", revision=revision,
        # The repository metadata declares test as well; the explicit train-only
        # file subset would otherwise fail the library's split-set comparison.
        verification_mode="no_checks",
    )
    if len(rows) != data_config["expected_train_size"]:
        raise ValueError(
            f"expected_train_size={data_config['expected_train_size']}, got {len(rows)}"
        )
    split = split_train_rows(rows, data_config["validation_size"], data_config["split_seed"])
    if split.source_sha256 != data_config["expected_train_sha256"]:
        raise ValueError(
            f"expected_train_sha256={data_config['expected_train_sha256']}, "
            f"got {split.source_sha256}"
        )
    return split


def build_manifest(
    split: TrainValidationSplit,
    data_config: Mapping[str, object],
    system_prompt: str,
    preview_count: int = 2,
) -> dict:
    if type(preview_count) is not int or preview_count < 0:
        raise ValueError("preview_count must be nonnegative")
    return {
        "dataset_id": data_config["id"],
        "dataset_name": data_config["name"],
        "dataset_revision": data_config["revision"],
        "source_split": "train",
        "source_sha256": split.source_sha256,
        "split_seed": data_config["split_seed"],
        "selection_method": "lowest SHA256(seed:index), then preserve source order",
        "source_count": len(split.training) + len(split.validation),
        "training_count": len(split.training),
        "validation_count": len(split.validation),
        "validation_prompt_ids": [item.prompt_id for item in split.validation],
        "prompt_preview": [
            {"prompt_id": item.prompt_id, "messages": build_prompt_messages(item, system_prompt)}
            for item in split.training[:preview_count]
        ],
    }
