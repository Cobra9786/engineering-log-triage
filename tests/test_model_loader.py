import torch

from log_triage.model_loader import (
    BASE_MODEL_ID,
    build_4bit_quantization_config,
)


def test_base_model_identity_is_explicit() -> None:
    assert BASE_MODEL_ID == "Qwen/Qwen2.5-1.5B-Instruct"


def test_4bit_quantization_uses_nf4_for_local_inference() -> None:
    config = build_4bit_quantization_config()

    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype is torch.float16
