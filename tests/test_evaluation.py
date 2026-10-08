import unittest

import torch

from src.data.dataset import MathProblem
from src.evaluation import evaluate_model


class FakeTokenizer:
    eos_token_id = 9
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_tensors):
        self.messages.append(messages)
        return torch.tensor([[1, len(self.messages)]])

    def decode(self, token_ids, *, skip_special_tokens):
        return {3: "#### 5", 4: "5"}[token_ids[0]]


class FakeModel:
    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def generate(self, *, input_ids, attention_mask, generation_config,
                 use_model_defaults=None):
        self.calls.append((generation_config, use_model_defaults))
        response = [3, 9] if input_ids[0, -1].item() == 1 else [4, 9]
        return torch.cat((input_ids, torch.tensor([response])), dim=1)


CONFIG = {
    "model": {"device": "cpu"},
    "prompt": {"system": "End with #### <number>."},
    "evaluation": {"do_sample": False, "max_new_tokens": 8},
}


class EvaluationTests(unittest.TestCase):
    def test_greedy_evaluation_scores_fixed_questions_without_answer_leakage(self):
        problems = (
            MathProblem("train/1", "Two plus three?", "#### 5"),
            MathProblem("train/2", "Four plus one?", "#### 5"),
        )
        model, tokenizer = FakeModel(), FakeTokenizer()
        rows, summary = evaluate_model(model, tokenizer, problems, CONFIG)
        self.assertEqual([row["reward"] for row in rows], [1.0, 0.0])
        self.assertEqual(summary["correct_count"], 1)
        self.assertEqual(summary["question_count"], 2)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["reason_counts"], {
            "correct_answer": 1, "missing_final_answer": 1,
        })
        self.assertTrue(all(call.do_sample is False and use_defaults is False
                            for call, use_defaults in model.calls))
        self.assertTrue(all("#### 5" not in messages[-1]["content"] for messages in tokenizer.messages))
        self.assertTrue(all(row["finish_reason"] == "eos" for row in rows))

    def test_rejects_empty_or_sampling_evaluation(self):
        with self.assertRaisesRegex(ValueError, "validation problems"):
            evaluate_model(FakeModel(), FakeTokenizer(), (), CONFIG)
        bad = {**CONFIG, "evaluation": {"do_sample": True, "max_new_tokens": 8}}
        with self.assertRaisesRegex(ValueError, "greedy"):
            evaluate_model(FakeModel(), FakeTokenizer(), (MathProblem("x", "Q?", "#### 5"),), bad)

    def test_length_limited_response_is_counted(self):
        class LengthModel(FakeModel):
            def generate(self, *, input_ids, attention_mask, generation_config,
                         use_model_defaults=None):
                return torch.cat((input_ids, torch.tensor([[4, 4]])), dim=1)

        rows, summary = evaluate_model(
            LengthModel(), FakeTokenizer(), (MathProblem("x", "Q?", "#### 5"),), CONFIG,
        )
        self.assertEqual(rows[0]["finish_reason"], "length")
        self.assertEqual(summary["truncation_count"], 1)


if __name__ == "__main__":
    unittest.main()
