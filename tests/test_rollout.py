import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.data.dataset import MathProblem
from src.data.schema import read_jsonl, validate_group, write_jsonl
from src.rollout.hf_rollout import (
    compute_old_log_probs,
    gather_response_log_probs,
    sample_group,
    trim_generated_response,
)


class FakeTokenizer:
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [
            {"role": "system", "content": "Solve it"},
            {"role": "user", "content": "What is 2+2?"},
        ]
        assert kwargs["add_generation_prompt"] is True
        return torch.tensor([[1, 2]])

    def decode(self, token_ids, skip_special_tokens):
        return " ".join(str(token) for token in token_ids if token != self.eos_token_id)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_grad_modes = []
        self.generate_config = None
        self.use_model_defaults = None

    def generate(self, *, input_ids, attention_mask, generation_config,
                 use_model_defaults=None):
        self.generate_config = generation_config
        self.use_model_defaults = use_model_defaults
        assert input_ids.tolist() == [[1, 2]]
        assert attention_mask.tolist() == [[1, 1]]
        return torch.tensor([
            [1, 2, 3, 9, 9],
            [1, 2, 4, 5, 9],
            [1, 2, 6, 7, 8],
            [1, 2, 3, 4, 9],
        ])

    def forward(self, *, input_ids, attention_mask):
        self.forward_grad_modes.append(torch.is_grad_enabled())
        return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], 12))


class RolloutTests(unittest.TestCase):
    def test_eos_included_once_and_padding_removed(self) -> None:
        self.assertEqual(trim_generated_response([3, 9, 9], 9), ([3, 9], "eos"))
        self.assertEqual(trim_generated_response([3, 4, 5], 9), ([3, 4, 5], "length"))

    def test_empty_or_invalid_response_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "generated response"):
            trim_generated_response([], 9)
        with self.assertRaisesRegex(ValueError, "eos_token_id"):
            trim_generated_response([3], -1)

    def test_first_response_uses_prompt_last_logit(self) -> None:
        logits = torch.zeros(1, 4, 5)
        logits[0, 1, 3] = 2.0
        logits[0, 2, 4] = 3.0
        result = gather_response_log_probs(logits, torch.tensor([[3, 4]]), prompt_length=2)
        self.assertEqual(tuple(result.shape), (1, 2))
        self.assertAlmostEqual(result[0, 0].item(), 2 - math.log(math.exp(2) + 4), places=6)
        self.assertAlmostEqual(result[0, 1].item(), 3 - math.log(math.exp(3) + 4), places=6)

    def test_old_log_probs_forward_has_no_grad(self) -> None:
        model = FakeModel()
        values = compute_old_log_probs(model, [1, 2], [3, 9], device="cpu")
        self.assertEqual(model.forward_grad_modes, [False])
        self.assertEqual(len(values), 2)
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_group_uses_explicit_sampling_config_and_round_trips(self) -> None:
        model = FakeModel()
        config = {
            "model": {"device": "cpu", "revision": "model-sha", "tokenizer_revision": "tokenizer-sha"},
            "prompt": {"system": "Solve it"},
            "generation": {
                "group_size": 4, "do_sample": True, "temperature": 1.0,
                "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0,
                "max_new_tokens": 3,
                "extra_logits_processors": [],
            },
        }
        problem = MathProblem("gsm8k/main/train/00000", "What is 2+2?", "SECRET #### 4")
        group = sample_group(model, FakeTokenizer(), problem, config, "step-000000")
        validate_group(group)
        self.assertFalse(model.training)
        self.assertEqual(model.forward_grad_modes, [False] * 4)
        self.assertEqual(model.generate_config.repetition_penalty, 1.0)
        self.assertEqual(model.generate_config.top_k, 0)
        self.assertIs(model.use_model_defaults, False)
        self.assertEqual(model.generate_config.num_return_sequences, 4)
        self.assertEqual([item.finish_reason for item in group], ["eos", "eos", "length", "eos"])
        self.assertEqual(group[0].response_token_ids, [3, 9])
        self.assertEqual(group[0].response_mask, [1, 1])
        self.assertEqual({item.policy_version for item in group}, {"step-000000"})
        self.assertTrue(all(len(item.old_log_probs) == len(item.response_token_ids) for item in group))
        self.assertTrue(all(item.reward is None and item.advantage is None for item in group))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollout.jsonl"
            write_jsonl(path, group)
            self.assertEqual(read_jsonl(path), group)


if __name__ == "__main__":
    unittest.main()
