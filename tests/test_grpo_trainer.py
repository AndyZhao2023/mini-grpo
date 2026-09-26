import copy
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from src.data.schema import Trajectory
from src.grpo.trainer import run_one_grpo_step
from src.rollout.hf_rollout import gather_response_log_probs


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(10, 10)

    def forward(self, *, input_ids, attention_mask):
        return SimpleNamespace(logits=self.table(input_ids))


def make_group(model: TinyCausalModel) -> list[Trajectory]:
    prompt_ids = [1, 2]
    responses = [[3, 9], [4, 9]]
    with torch.no_grad():
        full = torch.tensor([prompt_ids + response for response in responses])
        old = gather_response_log_probs(model(input_ids=full, attention_mask=torch.ones_like(full)).logits,
                                        torch.tensor(responses), len(prompt_ids)).tolist()
    return [
        Trajectory(
            schema_version=1,
            prompt_id="gsm8k/main/train/00000",
            group_id="step-000000/gsm8k/main/train/00000",
            sample_index=index,
            group_size=2,
            prompt_text="Question?",
            prompt_token_ids=prompt_ids,
            response_text="answer",
            response_token_ids=response,
            response_mask=[1, 1],
            finish_reason="eos",
            policy_version="step-000000",
            model_revision="model-sha",
            tokenizer_revision="tokenizer-sha",
            old_log_probs=old[index],
            reward=float(1 - index),
            reward_reason="scored",
            advantage=1.0 if index == 0 else -1.0,
        )
        for index, response in enumerate(responses)
    ]


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyCausalModel()
        self.group = make_group(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def step(self, group=None):
        return run_one_grpo_step(
            self.model, self.optimizer, self.group if group is None else group,
            pad_token_id=9, device="cpu", clip_epsilon=0.2,
            loss_reduction="response_then_group_then_batch", max_grad_norm=1.0,
        )

    def test_one_step_changes_parameters_and_tracks_policy_drift(self) -> None:
        before = copy.deepcopy(self.model.state_dict())
        metrics = self.step()
        self.assertLessEqual(metrics.pre_max_ratio_deviation, 1e-3)
        self.assertEqual(metrics.pre_clip_fraction, 0.0)
        self.assertGreater(metrics.gradient_norm_before_clip, 0.0)
        self.assertGreater(metrics.post_max_ratio_deviation, 0.0)
        self.assertGreaterEqual(metrics.post_approx_kl_old, 0.0)
        self.assertTrue(math.isfinite(metrics.post_loss))
        self.assertEqual(metrics.valid_token_count, 4)
        self.assertTrue(any(not torch.equal(before[key], self.model.state_dict()[key]) for key in before))
        self.assertTrue(all(item.advantage is not None for item in self.group))

    def test_stale_old_log_probs_abort_before_optimizer_step(self) -> None:
        before = copy.deepcopy(self.model.state_dict())
        stale = [replace(self.group[0], old_log_probs=[self.group[0].old_log_probs[0] + 0.1,
                                                      self.group[0].old_log_probs[1]]), self.group[1]]
        with self.assertRaisesRegex(RuntimeError, "pre-update ratio"):
            self.step(stale)
        self.assertTrue(all(torch.equal(before[key], self.model.state_dict()[key]) for key in before))

    def test_all_equal_rewards_or_zero_advantages_abort(self) -> None:
        all_equal = [replace(item, reward=0.0, advantage=0.0) for item in self.group]
        with self.assertRaisesRegex(ValueError, "mixed-reward"):
            self.step(all_equal)
        zero_advantages = [replace(item, advantage=0.0) for item in self.group]
        with self.assertRaisesRegex(ValueError, "nonzero advantages"):
            self.step(zero_advantages)

    def test_gradient_cancellation_aborts_before_step(self) -> None:
        identical = [self.group[0], replace(
            self.group[1], response_token_ids=self.group[0].response_token_ids,
            old_log_probs=self.group[0].old_log_probs,
        )]
        before = copy.deepcopy(self.model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "no finite nonzero gradient"):
            self.step(identical)
        self.assertTrue(all(torch.equal(before[key], self.model.state_dict()[key]) for key in before))


if __name__ == "__main__":
    unittest.main()
