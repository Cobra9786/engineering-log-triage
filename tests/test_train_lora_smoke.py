from pathlib import Path

from log_triage.train_lora_smoke import (
    DEFAULT_OUTPUT_DIR,
    PROCESSED_DATASET_DIR,
    build_lora_config,
    build_sft_config,
)


def test_lora_config_is_small_and_causal_lm() -> None:
    config = build_lora_config()

    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert config.task_type == "CAUSAL_LM"
    assert "q_proj" in config.target_modules
    assert "v_proj" in config.target_modules


def test_sft_smoke_config_uses_tiny_training_run(tmp_path: Path) -> None:
    config = build_sft_config(tmp_path, max_steps=4)

    assert config.max_steps == 4
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 2
    assert config.max_length == 1024
    assert config.packing is False
    assert config.fp16 is False
    assert config.bf16 is False
    assert config.max_grad_norm == 0.0
    assert config.output_dir == str(tmp_path)


def test_default_training_paths_are_project_local() -> None:
    assert PROCESSED_DATASET_DIR.parts[-3:] == ("data", "processed", "lora_v1")
    assert DEFAULT_OUTPUT_DIR.parts[-3:] == ("artifacts", "training", "lora_smoke")
