from __future__ import annotations

import argparse
import json
import random
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


def print_example(
    example: dict[str, Any],
) -> None:
    print(
        "=" * 70
    )

    print(
        f"ID: {example['id']}"
    )

    print(
        f"Category: {example['category']}"
    )

    print()

    for message in example[
        "messages"
    ]:
        print(
            f"[{message['role'].upper()}]"
        )

        print(
            message[
                "content"
            ]
        )

        print()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "eval",
        ],
        default="train",
    )

    parser.add_argument(
        "--per-category",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260830,
    )

    args = parser.parse_args()

    if args.per_category <= 0:
        raise SystemExit(
            "--per-category must be positive."
        )

    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "training"
        / (
            "tools_train.jsonl"
            if args.split == "train"
            else "tools_eval.jsonl"
        )
    )

    examples = load_jsonl(
        dataset_path
    )

    grouped: dict[
        str,
        list[
            dict[str, Any]
        ],
    ] = defaultdict(
        list
    )

    for example in examples:
        grouped[
            example[
                "category"
            ]
        ].append(
            example
        )

    categories = [
        "positive_tool_call",
        "hard_negative",
        "ambiguity",
        "success_followup",
        "failure_followup",
    ]

    rng = random.Random(
        args.seed
    )

    print(
        "Atlas Tools Dataset Inspector"
    )

    print(
        f"Split: {args.split}"
    )

    print(
        f"Dataset: {dataset_path}"
    )

    print()

    for category in categories:
        candidates = grouped[
            category
        ]

        print(
            "# " + category
        )

        print(
            f"Available: {len(candidates)}"
        )

        print()

        sample_count = min(
            args.per_category,
            len(candidates),
        )

        for example in rng.sample(
            candidates,
            sample_count,
        ):
            print_example(
                example
            )


if __name__ == "__main__":
    main()
