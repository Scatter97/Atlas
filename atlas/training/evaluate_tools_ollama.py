from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from atlas.model import OllamaBackend
from atlas.protocol import (
    parse_tool_specialist_action,
    tool_specialist_action_json_schema,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


def load_jsonl(
    path: Path,
) -> list[
    dict[str, Any]
]:
    examples = []

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
        default="qwen3:1.7b",
    )

    parser.add_argument(
        "--base-url",
        default=(
            "http://127.0.0.1:11434"
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

    args = parser.parse_args()

    backend = OllamaBackend(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=120,
        keep_alive="30m",
        think=False,
        temperature=0.0,
    )

    examples = load_jsonl(
        args.dataset
    )[:args.limit]

    correct = 0
    invalid = 0

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

        try:
            response = (
                backend.generate_structured(
                    messages=messages,
                    response_schema=(
                        tool_specialist_action_json_schema()
                    ),
                )
            )

            action = (
                parse_tool_specialist_action(
                    response.content
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
        "Atlas Tools benchmark"
    )

    print(
        f"Model: {args.model}"
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


if __name__ == "__main__":
    main()
