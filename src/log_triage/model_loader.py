"""Reusable local loader for the engineering-log-triage foundation model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_BASE_MODEL_REVISION = os.environ.get(
    "LOG_TRIAGE_BASE_MODEL_REVISION",
    "main",
)


@dataclass
class LoadedLanguageModel:
    """A tokenizer and quantized causal language model loaded for inference."""

    tokenizer: Any
    model: Any
    model_id: str
    requested_revision: str
    resolved_revision: str

    @property
    def input_device(self) -> torch.device:
        """Return the device holding the model input embeddings."""

        return self.model.get_input_embeddings().weight.device


def build_4bit_quantization_config() -> BitsAndBytesConfig:
    """Build the 4-bit NF4 configuration used for local Qwen inference."""

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_qwen_for_inference(
    model_id: str = BASE_MODEL_ID,
    revision: str = DEFAULT_BASE_MODEL_REVISION,
) -> LoadedLanguageModel:
    """Load Qwen once for a long-running local inference process.

    The first call downloads missing model files into the Hugging Face cache.
    Later calls reuse the cached files, but each new process loads its own
    quantized model instance into available GPU memory.
    """

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This loader is configured for local NVIDIA GPU inference."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=build_4bit_quantization_config(),
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    resolved_revision = str(
        getattr(model.config, "_commit_hash", revision)
    )

    return LoadedLanguageModel(
        tokenizer=tokenizer,
        model=model,
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
    )
