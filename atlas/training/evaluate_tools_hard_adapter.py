from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    output = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            line = line.strip()

            if line:
                output.append(
                    json.loads(
                        line
                    )
                )

    return output


def score(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> bool:
    if (
        expected.get("type")
        != actual.get("type")
    ):
        return False

    if (
        expected.get("type")
        == "tool_call"
    ):
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

    # -------------------------------------------------
    # New flag: evaluate without loading the adapter
    # -------------------------------------------------
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help=(
            "Evaluate the untouched base model "
            "without loading the Atlas Tools adapter."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "training"
            / "tools_hard_eval.jsonl"
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--show-invalid",
        type=int,
        default=0,
        help=(
            "Print up to this many "
            "raw invalid model outputs."
        ),
    )

    args = parser.parse_args()

    try:
        import torch

        from peft import PeftModel

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

    examples = load_jsonl(
        args.dataset
    )

    if not examples:
        raise SystemExit(
            "Hard benchmark is empty."
        )

    use_bf16 = bool(
        torch.cuda.is_bf16_supported()
    )

    dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    )

    print(
        f"Loading base model: {args.model}"
    )

    base = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization,
            device_map="auto",
            dtype=dtype,
        )
    )

    # -------------------------------------------------
    # Adapter loading is now optional
    # -------------------------------------------------
    if args.no_adapter:
        print(
            "Adapter: disabled "
            "(evaluating stock base model)"
        )
        model = base
    else:
        if not args.adapter.exists():
            raise SystemExit(
                "Adapter directory does not exist: "
                f"{args.adapter}"
            )
        print(
            f"Loading adapter: {args.adapter}"
        )
        model = PeftModel.from_pretrained(
            base,
            str(
                args.adapter
            ),
        )

    model.eval()

    # -------------------------------------------------
    # Tokenizer source depends on whether adapter is used
    # -------------------------------------------------
    tokenizer_source = (
        args.model
        if args.no_adapter
        else args.adapter
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=True,
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    category_total = defaultdict(
        int
    )

    category_correct = defaultdict(
        int
    )

    category_invalid = defaultdict(
        int
    )

    total_correct = 0
    total_invalid = 0
    shown_invalid = 0

    print()
    print(
        "Starting Atlas Tools hard benchmark."
    )

    for index, item in enumerate(
        examples,
        start=1,
    ):
        category = item[
            "category"
        ]

        messages = list(
            item[
                "messages"
            ]
        )

        expected_message = (
            messages.pop()
        )

        expected = json.loads(
            expected_message[
                "content"
            ]
        )

        category_total[
            category
        ] += 1

        raw = "<no output>"
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

            raw = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            parsed = (
                parse_tool_specialist_action(
                    raw
                )
            )

            actual = parsed.model_dump(
                mode="json"
            )

        except Exception:
            total_invalid += 1

            category_invalid[
                category
            ] += 1

            if (
                args.show_invalid > 0
                and shown_invalid
                < args.show_invalid
            ):
                print()
                print(
                    "-" * 50
                )
                print(
                    "INVALID OUTPUT"
                )
                print(
                    f"Category: {category}"
                )
                print(
                    "Expected:"
                )
                print(
                    json.dumps(
                        expected,
                        ensure_ascii=False,
                    )
                )
                print()
                print(
                    "Raw model output:"
                )
                print(
                    raw
                )
                print(
                    "-" * 50
                )

                shown_invalid += 1

            actual = {
                "type": "invalid"
            }

        if score(
            expected,
            actual,
        ):
            total_correct += 1

            category_correct[
                category
            ] += 1

        if (
            index % 25
            == 0
        ):
            print(
                f"{index}/{len(examples)} "
                f"correct={total_correct} "
                f"invalid={total_invalid}"
            )

    total = len(
        examples
    )

    print()
    print(
        "Atlas Tools Hard Benchmark"
    )

    print(
        f"Base model: {args.model}"
    )

    # -------------------------------------------------
    # Report adapter usage based on flag
    # -------------------------------------------------
    if args.no_adapter:
        print(
            "Adapter: none "
            "(stock base model)"
        )
    else:
        print(
            f"Adapter: {args.adapter}"
        )

    print()

    order = [
        "unseen_phrasing",
        "ambiguity",
        "hard_negative",
        "failure_handling",
        "unseen_tool",
    ]

    for category in order:
        category_count = (
            category_total[
                category
            ]
        )

        correct = (
            category_correct[
                category
            ]
        )

        invalid = (
            category_invalid[
                category
            ]
        )

        accuracy = (
            correct
            / category_count
            if category_count
            else 0.0
        )

        print(
            f"{category:18} "
            f"{correct:3}/{category_count:3} "
            f"{accuracy:7.2%} "
            f"invalid={invalid}"
        )

    overall = (
        total_correct
        / total
        if total
        else 0.0
    )

    print()
    print(
        f"Overall: {total_correct}/{total}"
    )

    print(
        f"Accuracy: {overall:.2%}"
    )

    print(
        f"Invalid outputs: {total_invalid}"
    )


if __name__ == "__main__":
    main()
