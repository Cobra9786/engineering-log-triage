"""QLoRA smoke training for the engineering-log triage adapter.

This is intentionally small. It verifies that the local machine can:
- load Qwen in 4-bit NF4 form;
- attach LoRA adapters;
- run a short TRL SFTTrainer training loop;
- save the adapter;
- reload the adapter;
- generate one triage response.

It is not the final training run and should not be used for performance claims.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from log_triage.dataset import load_jsonl_dataset
from log_triage.model_loader import (
    BASE_MODEL_ID,
    build_4bit_quantization_config,
    load_qwen_for_inference,
)
from log_triage.prompting import SYSTEM_PROMPT, render_user_prompt
from log_triage.schema import TriageResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "lora_v1"
RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "seed_examples.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "training" / "lora_smoke"


def load_supervised_splits(dataset_dir: Path) -> DatasetDict:
    """Load exported chat-message JSONL splits for TRL SFT training."""

    data_files = {
        "train": str(dataset_dir / "train.jsonl"),
        "validation": str(dataset_dir / "validation.jsonl"),
    }

    dataset = load_dataset("json", data_files=data_files)

    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a DatasetDict from the supervised JSONL export.")

    return dataset


def load_trainable_qwen() -> tuple[Any, Any]:
    """Load Qwen in 4-bit form and prepare it for k-bit LoRA training."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the QLoRA smoke training script.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=build_4bit_quantization_config(),
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    return tokenizer, model


def build_lora_config() -> LoraConfig:
    """Create a conservative LoRA config for Qwen 1.5B smoke training."""

    return LoraConfig(
        r=8,
        lora_alpha=16,
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


def build_sft_config(output_dir: Path, max_steps: int) -> SFTConfig:
    """Build a tiny SFT configuration for a local smoke run."""

    return SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        max_length=1024,
        packing=False,
        fp16=False,
        bf16=False,
        max_grad_norm=0.0,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )



def prepare_output_dir(output_dir: Path, overwrite_output_dir: bool) -> Path:
    """Create a training output directory without accidental overwrite."""

    resolved_output_dir = output_dir.resolve()

    if resolved_output_dir.exists() and not resolved_output_dir.is_dir():
        raise NotADirectoryError(
            f"Training output path exists but is not a directory: {resolved_output_dir}",
        )

    if resolved_output_dir.exists() and any(resolved_output_dir.iterdir()):
        if not overwrite_output_dir:
            raise FileExistsError(
                "Training output directory already exists and is not empty: "
                f"{resolved_output_dir}. Use --overwrite-output-dir only when you "
                "intentionally want to delete and replace this run.",
            )

        shutil.rmtree(resolved_output_dir)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return resolved_output_dir


def train_smoke_adapter(
    dataset_dir: Path,
    output_dir: Path,
    max_steps: int,
    overwrite_output_dir: bool = False,
) -> Path:
    """Run a tiny QLoRA SFT smoke training loop and save the adapter."""

    dataset_dir = dataset_dir.resolve()
    output_dir = prepare_output_dir(output_dir, overwrite_output_dir)

    dataset = load_supervised_splits(dataset_dir)
    tokenizer, model = load_trainable_qwen()

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(output_dir, max_steps=max_steps),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=build_lora_config(),
    )

    started_at = time.perf_counter()
    trainer.train()
    elapsed_seconds = time.perf_counter() - started_at

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metadata = {
        "training_run_type": "qlora_smoke",
        "base_model": BASE_MODEL_ID,
        "adapter_dir": str(adapter_dir.relative_to(PROJECT_ROOT)),
        "dataset_dir": str(dataset_dir.relative_to(PROJECT_ROOT)),
        "max_steps": max_steps,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "torch_cuda_device": torch.cuda.get_device_name(0),
        "gpu_memory_allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "gpu_memory_reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    }

    (output_dir / "smoke_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("QLoRA smoke training completed.")
    print(json.dumps(metadata, indent=2, sort_keys=True))

    del trainer
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return adapter_dir


def generate_with_reloaded_adapter(adapter_dir: Path) -> None:
    """Reload the saved adapter and generate one response for a training example."""

    example = load_jsonl_dataset(RAW_DATASET_PATH)[0]
    loaded = load_qwen_for_inference()
    loaded.model = PeftModel.from_pretrained(loaded.model, adapter_dir)
    loaded.model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(example.report_text)},
    ]

    rendered_prompt = loaded.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = loaded.tokenizer(rendered_prompt, return_tensors="pt")
    model_inputs = {
        name: value.to(loaded.input_device)
        for name, value in model_inputs.items()
    }

    with torch.inference_mode():
        generated_ids = loaded.model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=256,
            pad_token_id=loaded.tokenizer.pad_token_id,
        )

    completion_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
    response_text = loaded.tokenizer.decode(
        completion_ids[0],
        skip_special_tokens=True,
    ).strip()

    print()
    print("Reloaded-adapter smoke generation:")
    print(response_text)

    try:
        payload = json.loads(response_text)
        validated = TriageResult.model_validate(payload)
    except Exception as error:  # noqa: BLE001 - smoke diagnostics should print exact failure.
        print(f"Reloaded-adapter strict schema validation: FAILED ({error})")
        return

    print("Reloaded-adapter strict schema validation: PASSED")
    print(json.dumps(validated.model_dump(mode="json"), indent=2))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run a tiny QLoRA smoke training loop for engineering-log triage.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROCESSED_DATASET_DIR,
        help="Directory containing exported train/validation JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the smoke adapter and metadata will be written.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Tiny number of training steps for smoke validation.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Delete and replace an existing non-empty output directory.",
    )
    parser.add_argument(
        "--skip-reload-check",
        action="store_true",
        help="Skip adapter reload and generation check after training.",
    )
    return parser.parse_args()


def main() -> None:
    """Run QLoRA smoke training."""

    args = parse_args()

    adapter_dir = train_smoke_adapter(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        overwrite_output_dir=args.overwrite_output_dir,
    )

    if not args.skip_reload_check:
        generate_with_reloaded_adapter(adapter_dir)


if __name__ == "__main__":
    main()
