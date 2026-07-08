"""FastAPI service for LoRA-adapted engineering-log triage."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from peft import PeftModel
from pydantic import BaseModel, Field

from log_triage.model_loader import BASE_MODEL_ID, load_qwen_for_inference
from log_triage.output_parsing import parse_json_object_with_recovery
from log_triage.prompting import SYSTEM_PROMPT, render_user_prompt
from log_triage.schema import TriageResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "qwen_lora_r8_train48_steps72_v1"
    / "adapter"
)


class TriageRequest(BaseModel):
    """Request body for engineering-log triage."""

    report_text: str = Field(
        min_length=1,
        max_length=10_000,
        description="Unstructured engineering or sensor fault report.",
    )


class TriageApiResponse(BaseModel):
    """Validated triage response returned by the API."""

    result: TriageResult
    raw_response: str
    strict_json_schema_valid: bool
    recovered_json_schema_valid: bool
    markdown_fence_recovery_used: bool
    latency_seconds: float
    model_id: str
    adapter_dir: str


class TriageRuntime(Protocol):
    """Protocol for real and test triage runtimes."""

    def triage(self, report_text: str) -> TriageApiResponse:
        """Triage one engineering report."""


class ModelOutputError(RuntimeError):
    """Raised when model output cannot be parsed or schema-validated."""


class QwenLoRATriageRuntime:
    """Qwen base model plus a saved PEFT/LoRA adapter."""

    def __init__(
        self,
        adapter_dir: Path,
        max_new_tokens: int = 256,
    ) -> None:
        self.adapter_dir = adapter_dir.resolve()
        self.max_new_tokens = max_new_tokens

        if not self.adapter_dir.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {self.adapter_dir}")

        self.loaded = load_qwen_for_inference()
        self.loaded.model = PeftModel.from_pretrained(
            self.loaded.model,
            self.adapter_dir,
        )
        self.loaded.model.eval()

    @property
    def model_id(self) -> str:
        """Return the loaded base model identifier."""

        return self.loaded.model_id

    def generate_raw_response(self, report_text: str) -> str:
        """Generate a raw model response for one report."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(report_text)},
        ]

        rendered_prompt = self.loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.loaded.tokenizer(rendered_prompt, return_tensors="pt")
        model_inputs = {
            name: value.to(self.loaded.input_device)
            for name, value in model_inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.loaded.model.generate(
                **model_inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.loaded.tokenizer.pad_token_id,
            )

        completion_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
        return self.loaded.tokenizer.decode(
            completion_ids[0],
            skip_special_tokens=True,
        ).strip()

    def triage(self, report_text: str) -> TriageApiResponse:
        """Generate, parse, validate, and return one triage result."""

        started_at = time.perf_counter()
        raw_response = self.generate_raw_response(report_text)
        latency_seconds = time.perf_counter() - started_at

        try:
            parsed = parse_json_object_with_recovery(raw_response)
            result = TriageResult.model_validate(parsed.payload)
        except Exception as error:  # noqa: BLE001 - preserve exact model-output error.
            raise ModelOutputError(
                f"Model output did not validate against TriageResult: {error}",
            ) from error

        recovered_from_markdown = bool(parsed.recovered_from_markdown_fence)

        return TriageApiResponse(
            result=result,
            raw_response=raw_response,
            strict_json_schema_valid=not recovered_from_markdown,
            recovered_json_schema_valid=True,
            markdown_fence_recovery_used=recovered_from_markdown,
            latency_seconds=round(latency_seconds, 4),
            model_id=self.model_id,
            adapter_dir=str(self.adapter_dir),
        )


def resolve_adapter_dir() -> Path:
    """Resolve the adapter directory from environment or default local artifact path."""

    configured_path = os.environ.get("LOG_TRIAGE_ADAPTER_DIR")

    if configured_path:
        return Path(configured_path).expanduser().resolve()

    return DEFAULT_ADAPTER_DIR.resolve()


def get_runtime(request: Request) -> TriageRuntime:
    """Return the loaded triage runtime from application state."""

    runtime = getattr(request.app.state, "runtime", None)

    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="Triage model runtime is not loaded.",
        )

    return runtime


def create_app(load_runtime: bool = True) -> FastAPI:
    """Create the FastAPI app.

    Tests pass load_runtime=False and inject a fake runtime.
    Production/local serving uses load_runtime=True and loads Qwen+LoRA once at startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        if load_runtime:
            app.state.runtime = QwenLoRATriageRuntime(resolve_adapter_dir())

        yield

        if load_runtime:
            app.state.runtime = None

    app = FastAPI(
        title="Engineering Log Triage API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        runtime = getattr(request.app.state, "runtime", None)

        return {
            "status": "ok",
            "model_loaded": runtime is not None,
            "base_model": BASE_MODEL_ID,
            "adapter_dir": str(resolve_adapter_dir()),
        }

    @app.post("/triage", response_model=TriageApiResponse)
    def triage_report(
        request_body: TriageRequest,
        runtime: TriageRuntime = Depends(get_runtime),
    ) -> TriageApiResponse:
        try:
            return runtime.triage(request_body.report_text)
        except ModelOutputError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return app


app = create_app()
