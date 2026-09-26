import math
import unittest

import torch

from src.data.schema import Trajectory
from src.grpo.loss import clipped_grpo_loss, pack_group


def make_group() -> list[Trajectory]:
    responses = [([7, 9, 0], [1, 1, 0], [-1.0, -2.0, 0.0]),
                 ([8, 6, 9], [1, 1, 1], [-3.0, -4.0, -5.0])]
    return [
        Trajectory(
            schema_version=1,
            prompt_id="gsm8k/main/train/00000",
            group_id="step-000000/gsm8k/main/train/00000",
            sample_index=index,
            group_size=2,
            prompt_text="Question?",
            prompt_token_ids=[1, 2],
            response_text="answer",
            response_token_ids=ids,
            response_mask=mask,
            finish_reason="eos",
            policy_version="step-000000",
            model_revision="model-sha",
            tokenizer_revision="tokenizer-sha",
            old_log_probs=old,
            reward=float(index),
            reward_reason="scored",
            advantage=1.0 if index == 0 else -1.0,
        )
        for index, (ids, mask, old) in enumerate(responses)
    ]


class GrpoLossTests(unittest.TestCase):
    def test_pack_group_excludes_prompt_and_padding_from_response_mask(self) -> None:
        packed = pack_group(make_group(), pad_token_id=9, device="cpu")
        self.assertEqual(packed.input_ids.tolist(), [[1, 2, 7, 9, 9], [1, 2, 8, 6, 9]])
        self.assertEqual(packed.attention_mask.tolist(), [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
        self.assertEqual(packed.response_mask.tolist(), [[[True, True, False], [True, True, True]]])
        self.assertEqual(packed.old_log_probs.tolist(), [[[-1.0, -2.0, 0.0], [-3.0, -4.0, -5.0]]])
        self.assertEqual(packed.advantages.tolist(), [[1.0, -1.0]])
        self.assertEqual(packed.prompt_length, 2)

    def test_pre_update_ratio_one_has_gradient_only_through_new_policy(self) -> None:
        new = torch.zeros((1, 2, 1), requires_grad=True)
        old = torch.zeros((1, 2, 1), requires_grad=True)
        advantages = torch.tensor([[1.0, -1.0]], requires_grad=True)
        mask = torch.ones((1, 2, 1), dtype=torch.bool)
        result = clipped_grpo_loss(new, old, advantages, mask, 0.2)
        self.assertEqual(result.loss.item(), 0.0)
        self.assertEqual(result.mean_ratio.item(), 1.0)
        self.assertEqual(result.max_ratio_deviation.item(), 0.0)
        self.assertEqual(result.clip_fraction.item(), 0.0)
        self.assertEqual(result.approx_kl_old.item(), 0.0)
        result.loss.backward()
        self.assertEqual(new.grad.tolist(), [[[-0.5], [0.5]]])
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)

    def test_clipping_both_sides_blocks_outward_gradients(self) -> None:
        new = torch.tensor([[[math.log(1.5)], [math.log(0.5)]]], requires_grad=True)
        old = torch.zeros_like(new)
        advantages = torch.tensor([[1.0, -1.0]])
        mask = torch.ones_like(new, dtype=torch.bool)
        result = clipped_grpo_loss(new, old, advantages, mask, 0.2)
        self.assertAlmostEqual(result.loss.item(), -0.2, places=6)
        self.assertEqual(result.clip_fraction.item(), 1.0)
        self.assertAlmostEqual(result.max_ratio_deviation.item(), 0.5, places=6)
        self.assertAlmostEqual(
            result.approx_kl_old.item(),
            ((1.5 - 1 - math.log(1.5)) + (0.5 - 1 - math.log(0.5))) / 2,
            places=6,
        )
        result.loss.backward()
        self.assertEqual(new.grad.tolist(), [[[0.0], [0.0]]])

    def test_clipping_does_not_block_inward_gradients(self) -> None:
        new = torch.tensor([[[math.log(0.5)], [math.log(1.5)]]], requires_grad=True)
        result = clipped_grpo_loss(
            new, torch.zeros_like(new), torch.tensor([[1.0, -1.0]]),
            torch.ones_like(new, dtype=torch.bool), 0.2,
        )
        self.assertAlmostEqual(result.loss.item(), 0.5, places=6)
        result.loss.backward()
        self.assertAlmostEqual(new.grad[0, 0, 0].item(), -0.25, places=6)
        self.assertAlmostEqual(new.grad[0, 1, 0].item(), 0.75, places=6)

    def test_response_mean_precedes_group_mean_and_padding_has_no_effect(self) -> None:
        new = torch.zeros((1, 2, 3), requires_grad=True)
        old = torch.zeros_like(new)
        mask = torch.tensor([[[1, 0, 0], [1, 1, 1]]], dtype=torch.bool)
        with torch.no_grad():
            new[0, 0, 1:] = 10.0
        result = clipped_grpo_loss(new, old, torch.tensor([[1.0, -1.0]]), mask, 0.2)
        self.assertEqual(result.valid_token_count, 4)
        self.assertEqual(result.loss.item(), 0.0)
        self.assertEqual(result.clip_fraction.item(), 0.0)
        self.assertEqual(result.mean_ratio.item(), 1.0)
        result.loss.backward()
        self.assertEqual(new.grad[0, 0, 1:].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(new.grad[0, 0, 0].item(), -0.5, places=6)
        self.assertAlmostEqual(new.grad[0, 1, 0].item(), 1 / 6, places=6)

    def test_all_zero_advantages_produce_exact_zero_loss_and_gradients(self) -> None:
        new = torch.tensor([[[math.log(1.5)], [math.log(0.5)]]], requires_grad=True)
        result = clipped_grpo_loss(
            new, torch.zeros_like(new), torch.zeros((1, 2)),
            torch.ones_like(new, dtype=torch.bool), 0.2,
        )
        self.assertEqual(result.loss.item(), 0.0)
        result.loss.backward()
        self.assertEqual(new.grad.tolist(), [[[0.0], [0.0]]])

    def test_invalid_shapes_masks_and_values_fail(self) -> None:
        new = torch.zeros((1, 2, 1))
        old = torch.zeros_like(new)
        advantages = torch.ones((1, 2))
        mask = torch.ones_like(new, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "expected log_probs"):
            clipped_grpo_loss(new, old, torch.ones((2,)), mask, 0.2)
        with self.assertRaisesRegex(ValueError, "at least one valid token"):
            clipped_grpo_loss(new, old, advantages, torch.tensor([[[1], [0]]], dtype=torch.bool), 0.2)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            clipped_grpo_loss(new + float("nan"), old, advantages, mask, 0.2)
        with self.assertRaisesRegex(ValueError, "ratio must be finite"):
            clipped_grpo_loss(new + 1000, old, advantages, mask, 0.2)
        with self.assertRaisesRegex(ValueError, "GRPO loss must be finite"):
            clipped_grpo_loss(new + 80, old, -advantages * 1e10, mask, 0.2)
        with self.assertRaisesRegex(ValueError, "clip_epsilon"):
            clipped_grpo_loss(new, old, advantages, mask, 1.0)
        with self.assertRaisesRegex(ValueError, "reduction"):
            clipped_grpo_loss(new, old, advantages, mask, 0.2, "token_mean")


if __name__ == "__main__":
    unittest.main()
