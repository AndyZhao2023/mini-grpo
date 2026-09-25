"""Day 1: generate two groups and verify a small backward pass.

The loss here is generated-token NLL for a resource check, not GRPO training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SMOKE_QUESTIONS = (
    "Mia has 3 apples and buys 4 more. How many apples does she have?",
    "A box has 12 pencils. Five are removed. How many remain?",
)


def run(config_path: Path, output_path: Path) -> int:
    record: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "stage": "dependencies",
        "config": str(config_path),
        "note": "Generated-token NLL backward is a resource check, not a GRPO update.",
    }
    start = time.perf_counter()
    try:
        import yaml
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model_config = config["model"]
        generation = config["generation"]
        smoke = config["smoke"]
        record.update({
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "model_id": model_config["id"],
            "model_revision": model_config["revision"],
            "tokenizer_revision": model_config["tokenizer_revision"],
            "device": model_config["device"],
            "dtype": model_config["dtype"],
            "group_size": smoke["group_size"],
            "max_new_tokens": smoke["max_new_tokens"],
        })
        if smoke["prompt_count"] != len(SMOKE_QUESTIONS):
            raise ValueError("smoke.prompt_count must match the fixed question count")
        if smoke["group_size"] != generation["group_size"]:
            raise ValueError("smoke.group_size must match generation.group_size")
        if smoke["backward_responses"] != len(SMOKE_QUESTIONS):
            raise ValueError("smoke.backward_responses must be one per prompt")
        if model_config["device"] != "mps" or not torch.backends.mps.is_available():
            raise RuntimeError("configured MPS device is unavailable")
        if model_config["dtype"] != "float32":
            raise ValueError("Day 1 smoke currently supports float32 only")

        torch.manual_seed(config["seed"])
        record["stage"] = "model_load"
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["id"], revision=model_config["tokenizer_revision"],
            trust_remote_code=False,
        )
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        model = AutoModelForCausalLM.from_pretrained(
            model_config["id"], revision=model_config["revision"],
            dtype=torch.float32, trust_remote_code=False,
        ).to("mps")
        model.eval()  # Dropout off; autograd remains enabled during the backward stage.
        record["load_seconds"] = round(time.perf_counter() - start, 3)

        record["stage"] = "generation"
        generated: list[dict] = []
        backward_inputs: list[list[int]] = []
        backward_prompt_lengths: list[int] = []
        generation_start = time.perf_counter()
        with torch.no_grad():
            for question in SMOKE_QUESTIONS:
                messages = [
                    {"role": "system", "content": config["prompt"]["system"]},
                    {"role": "user", "content": question},
                ]
                prompt_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt",
                ).to("mps")
                prompt_length = prompt_ids.shape[1]
                sequences = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    do_sample=generation["do_sample"],
                    num_return_sequences=smoke["group_size"],
                    temperature=generation["temperature"],
                    top_p=generation["top_p"],
                    top_k=generation["top_k"],
                    max_new_tokens=smoke["max_new_tokens"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
                for sample_index, sequence in enumerate(sequences):
                    response_ids = sequence[prompt_length:].tolist()
                    if tokenizer.eos_token_id in response_ids:
                        response_ids = response_ids[:response_ids.index(tokenizer.eos_token_id) + 1]
                        finish_reason = "eos"
                    else:
                        finish_reason = "length"
                    generated.append({
                        "question": question,
                        "sample_index": sample_index,
                        "response_tokens": len(response_ids),
                        "finish_reason": finish_reason,
                        "response_text": tokenizer.decode(response_ids, skip_special_tokens=True),
                    })
                    if sample_index == 0:
                        backward_inputs.append(prompt_ids[0].tolist() + response_ids)
                        backward_prompt_lengths.append(prompt_length)
        record["generation_seconds"] = round(time.perf_counter() - generation_start, 3)
        record["generated"] = generated
        if len(generated) != len(SMOKE_QUESTIONS) * smoke["group_size"]:
            raise RuntimeError("generation did not produce a complete two-prompt group")

        record["stage"] = "backward"
        backward_start = time.perf_counter()
        longest = max(map(len, backward_inputs))
        padded_ids = []
        attention_masks = []
        labels = []
        for ids, prompt_length in zip(backward_inputs, backward_prompt_lengths):
            padding = longest - len(ids)
            padded_ids.append(ids + [tokenizer.eos_token_id] * padding)
            attention_masks.append([1] * len(ids) + [0] * padding)
            labels.append([-100] * prompt_length + ids[prompt_length:] + [-100] * padding)
        input_ids = torch.tensor(padded_ids, device="mps")
        attention_mask = torch.tensor(attention_masks, device="mps")
        label_ids = torch.tensor(labels, device="mps")
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("backward loss is not finite")
        loss.backward()
        first_gradient = next((p.grad for p in model.parameters() if p.grad is not None), None)
        if first_gradient is None or not torch.isfinite(first_gradient).all().item():
            raise RuntimeError("no finite parameter gradient was produced")
        gradient_norm = first_gradient.float().norm().item()
        if gradient_norm <= 0:
            raise RuntimeError("the first parameter gradient is zero")
        record.update({
            "backward_seconds": round(time.perf_counter() - backward_start, 3),
            "backward_loss": round(loss.item(), 6),
            "first_gradient_norm": round(gradient_norm, 6),
            "mps_allocated_bytes": torch.mps.current_allocated_memory(),
            "status": "passed",
            "stage": "complete",
        })
        return_code = 0
    except Exception as error:
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        return_code = 1
    record["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Day 1 smoke: {record['status']} at {record['stage']} → {output_path}")
    if return_code:
        print(f"{record['error_type']}: {record['error']}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m0.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/m0/day1-smoke.json"))
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.output))
