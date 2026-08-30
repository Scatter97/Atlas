from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=(
            "Qwen/Qwen3-1.7B"
        ),
    )

    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "training"
            / "tools_train.jsonl"
        ),
    )

    parser.add_argument(
        "--eval-dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "training"
            / "tools_eval.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "training_output"
            / "atlas-tools-v0.1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
    )

    args = parser.parse_args()

    try:
        import torch

        from datasets import (
            load_dataset,
        )

        from peft import (
            LoraConfig,
        )

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        from trl import (
            SFTConfig,
            SFTTrainer,
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

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=(
                quantization_config
            ),
            device_map="auto",
            dtype=(
                compute_dtype
            ),
        )
    )

    model.config.use_cache = False

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(
                args.train_dataset
            ),
            "validation": str(
                args.eval_dataset
            ),
        },
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_config = SFTConfig(
        output_dir=str(
            args.output_dir
        ),
        num_train_epochs=(
            args.epochs
        ),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=(
            args.learning_rate
        ),
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=100,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        max_length=args.max_length,
        packing=False,
        padding_free=False,
        assistant_only_loss=False,
        report_to="none",
        seed=20260829,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=(
            dataset["train"]
        ),
        eval_dataset=(
            dataset["validation"]
        ),
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(
        "Starting Atlas Tools QLoRA training."
    )

    trainer.train()

    final_dir = (
        args.output_dir
        / "final"
    )

    trainer.save_model(
        str(
            final_dir
        )
    )

    tokenizer.save_pretrained(
        str(
            final_dir
        )
    )

    print(
        "Training complete."
    )

    print(
        "Adapter saved to: "
        f"{final_dir}"
    )


if __name__ == "__main__":
    main()
