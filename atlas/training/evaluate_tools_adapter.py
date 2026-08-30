from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


def load_jsonl(
    path: Path,
) -> list[
    dict[str, Any]
]:
    examples: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(
                    line
                )
            )

    return examples


def score_action(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> bool:
    if (
        expected.get("type")
        != actual.get("type")
    ):
        return False

    if expected.get(
        "type"
    ) == "tool_call":
        return (
            expected.get("name")
            == actual.get("name")
            and expected.get(
                "arguments"
            )
            == actual.get(
                "arguments"
            )
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
    )

    parser.add_argument(
        "--adapter",
        type=Path,
        default=(
            PROJECT_ROOT
            / "training_output"
            / "atlas-tools-v0.1"
            / "final"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "training"
            / "tools_eval.jsonl"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    try:
        import torch

        from peft import (
            PeftModel,
        )

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        from atlas.protocol import (
            parse_tool_specialist_action,
        )

    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. "
            'Install with: pip install -e ".[training]"'
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU was not detected."
        )

    if not args.adapter.exists():
        raise SystemExit(
            "Adapter directory does not exist: "
            f"{args.adapter}"
        )

    if not args.dataset.exists():
        raise SystemExit(
            "Evaluation dataset does not exist: "
            f"{args.dataset}"
        )

    use_bf16 = bool(
        torch.cuda.is_bf16_supported()
    )

    compute_dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                compute_dtype
            ),
        )
    )

    print(
        "Loading base model: "
        f"{args.model}"
    )

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=(
                quantization_config
            ),
            device_map="auto",
            dtype=compute_dtype,
        )
    )

    print(
        "Loading Atlas Tools adapter: "
        f"{args.adapter}"
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(
            args.adapter
        ),
    )

    model.eval()

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.adapter,
            use_fast=True,
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    examples = load_jsonl(
        args.dataset
    )[:args.limit]

    if not examples:
        raise SystemExit(
            "Evaluation dataset is empty."
        )

    correct = 0
    invalid = 0

    print()
    print(
        "Starting Atlas Tools adapter benchmark."
    )
    print(
        f"Examples: {len(examples)}"
    )
    print()

    for index, example in enumerate(
        examples,
        start=1,
    ):
        messages = list(
            example["messages"]
        )

        expected_message = (
            messages.pop()
        )

        expected = json.loads(
            expected_message[
                "content"
            ]
        )

        actual: dict[
            str,
            Any,
        ]

        try:
            try:
                prompt = (
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            except TypeError:
                prompt = (
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )

            encoded = {
                key: value.to(
                    model.device
                )
                for key, value
                in encoded.items()
            }

            input_length = (
                encoded[
                    "input_ids"
                ].shape[1]
            )

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                    do_sample=False,
                    pad_token_id=(
                        tokenizer.pad_token_id
                    ),
                    eos_token_id=(
                        tokenizer.eos_token_id
                    ),
                )

            new_tokens = generated[
                0,
                input_length:,
            ]

            raw_output = (
                tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
            ).strip()

            action = (
                parse_tool_specialist_action(
                    raw_output
                )
            )

            actual = action.model_dump(
                mode="json"
            )

        except Exception:
            invalid += 1
            actual = {
                "type": "invalid"
            }

        if score_action(
            expected,
            actual,
        ):
            correct += 1

        if (
            index % 25
            == 0
            or index
            == len(examples)
        ):
            print(
                f"{index}/{len(examples)} "
                f"correct={correct} "
                f"invalid={invalid}"
            )

    total = len(
        examples
    )

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    print()
    print(
        "Atlas Tools adapter benchmark"
    )

    print(
        f"Base model: {args.model}"
    )

    print(
        f"Adapter: {args.adapter}"
    )

    print(
        f"Examples: {total}"
    )

    print(
        f"Correct: {correct}"
    )

    print(
        f"Invalid: {invalid}"
    )

    print(
        f"Accuracy: {accuracy:.2%}"
    )

    print()

    baseline_correct = 213
    baseline_total = 250
    baseline_accuracy = (
        baseline_correct
        / baseline_total
    )

    if total == baseline_total:
        difference = (
            accuracy
            - baseline_accuracy
        )

        print(
            "Stock Qwen3-1.7B baseline: "
            f"{baseline_accuracy:.2%}"
        )

        print(
            "Accuracy change: "
            f"{difference:+.2%}"
        )


if __name__ == "__main__":
    main()
