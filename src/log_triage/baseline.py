"""Prompt-only baseline evaluation for engineering-log triage."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from log_triage.dataset import load_jsonl_dataset
from log_triage.model_loader import LoadedLanguageModel, load_qwen_for_inference
from log_triage.output_parsing import (
    parse_json_object_with_recovery,
    parse_strict_json_object,
)
from log_triage.prompting import SYSTEM_PROMPT, render_user_prompt
from log_triage.schema import EngineeringLogExample, TriageResult
from log_triage.splits import load_split_manifest, partition_examples


BASELINE_CONFIG_ID = "qwen2.5-1.5b-instruct-nf4-greedy-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "seed_examples.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.json"


def generate_response(
    loaded: LoadedLanguageModel,
    report_text: str,
) -> tuple[str, float]:
    """Generate one deterministic model response and measure GPU latency."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(report_text)},
    ]

    rendered_prompt = loaded.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = loaded.tokenizer(
        rendered_prompt,
        return_tensors="pt",
    )
    model_inputs = {
        name: value.to(loaded.input_device)
        for name, value in model_inputs.items()
    }

    torch.cuda.synchronize()
    started_at = time.perf_counter()

    with torch.inference_mode():
        generated_ids = loaded.model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=256,
            pad_token_id=loaded.tokenizer.pad_token_id,
        )

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started_at

    completion_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
    response_text = loaded.tokenizer.decode(
        completion_ids[0],
        skip_special_tokens=True,
    ).strip()

    return response_text, elapsed_seconds


def evaluate_example(
    example: EngineeringLogExample,
    loaded: LoadedLanguageModel,
) -> dict[str, Any]:
    """Evaluate one example against strict and recoverable output contracts."""

    response_text, latency_seconds = generate_response(loaded, example.report_text)

    strict_result: TriageResult | None = None
    strict_error: str | None = None

    try:
        strict_payload = parse_strict_json_object(response_text)
        strict_result = TriageResult.model_validate(strict_payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        strict_error = str(error)

    recovered_result: TriageResult | None = None
    recovered_error: str | None = None
    markdown_fence_recovery_used: bool | None = None

    try:
        parsed_output = parse_json_object_with_recovery(response_text)
        recovered_result = TriageResult.model_validate(parsed_output.payload)
        markdown_fence_recovery_used = parsed_output.recovered_from_markdown_fence
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        recovered_error = str(error)

    field_correctness = {
        "category": False,
        "severity": False,
        "requires_human_review": False,
    }

    if recovered_result is not None:
        field_correctness = {
            "category": recovered_result.category == example.target.category,
            "severity": recovered_result.severity == example.target.severity,
            "requires_human_review": (
                recovered_result.requires_human_review
                == example.target.requires_human_review
            ),
        }

    return {
        "id": example.id,
        "report_text": example.report_text,
        "expected": example.target.model_dump(mode="json"),
        "raw_response": response_text,
        "latency_seconds": round(latency_seconds, 4),
        "strict_json_schema_valid": strict_result is not None,
        "strict_error": strict_error,
        "recovered_json_schema_valid": recovered_result is not None,
        "recovered_error": recovered_error,
        "markdown_fence_recovery_used": markdown_fence_recovery_used,
        "prediction": (
            recovered_result.model_dump(mode="json")
            if recovered_result is not None
            else None
        ),
        "field_correctness": field_correctness,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate strict validity, recoverable validity, and label accuracy."""

    total_examples = len(records)

    if total_examples == 0:
        raise ValueError("Cannot summarize an empty evaluation result set.")

    def count_true(key: str) -> int:
        return sum(record[key] is True for record in records)

    def rate(count: int) -> float:
        return round(count / total_examples, 4)

    strict_valid_count = count_true("strict_json_schema_valid")
    recovered_valid_count = count_true("recovered_json_schema_valid")
    recovery_used_count = count_true("markdown_fence_recovery_used")

    field_accuracy: dict[str, dict[str, float | int]] = {}

    for field_name in ("category", "severity", "requires_human_review"):
        correct_count = sum(
            record["field_correctness"][field_name] is True
            for record in records
        )
        field_accuracy[field_name] = {
            "correct_count": correct_count,
            "total_examples": total_examples,
            "accuracy": rate(correct_count),
        }

    strict_all_measured_fields_count = sum(
        record["strict_json_schema_valid"] is True
        and all(record["field_correctness"].values())
        for record in records
    )

    recovered_all_measured_fields_count = sum(
        record["recovered_json_schema_valid"] is True
        and all(record["field_correctness"].values())
        for record in records
    )

    latency_values = [record["latency_seconds"] for record in records]

    return {
        "total_examples": total_examples,
        "strict_json_schema_valid": {
            "count": strict_valid_count,
            "rate": rate(strict_valid_count),
        },
        "recovered_json_schema_valid": {
            "count": recovered_valid_count,
            "rate": rate(recovered_valid_count),
        },
        "markdown_fence_recovery_used": {
            "count": recovery_used_count,
            "rate": rate(recovery_used_count),
        },
        "field_accuracy": field_accuracy,
        "strict_all_measured_fields_correct": {
            "count": strict_all_measured_fields_count,
            "rate": rate(strict_all_measured_fields_count),
        },
        "recovered_all_measured_fields_correct": {
            "count": recovered_all_measured_fields_count,
            "rate": rate(recovered_all_measured_fields_count),
        },
        "latency_seconds": {
            "min": round(min(latency_values), 4),
            "max": round(max(latency_values), 4),
            "mean": round(sum(latency_values) / total_examples, 4),
        },
    }


def evaluate_split(
    split_name: str,
    output_path: Path,
) -> dict[str, Any]:
    """Load Qwen once, evaluate one protected split, and persist the report."""

    examples = load_jsonl_dataset(DATASET_PATH)
    manifest = load_split_manifest(MANIFEST_PATH)
    partitions = partition_examples(examples, manifest)

    if split_name not in ("validation", "test"):
        raise ValueError("Baseline evaluation is restricted to validation or test.")

    split_examples = partitions[split_name]
    loaded = load_qwen_for_inference()

    print(f"Loaded model: {loaded.model_id}")
    print(f"Resolved revision: {loaded.resolved_revision}")
    print(f"Evaluating split: {split_name} ({len(split_examples)} examples)")

    records: list[dict[str, Any]] = []

    for index, example in enumerate(split_examples, start=1):
        print(f"[{index}/{len(split_examples)}] Evaluating {example.id}...")
        records.append(evaluate_example(example, loaded))

    report = {
        "evaluation_type": "prompt_only_baseline",
        "baseline_config_id": BASELINE_CONFIG_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": {
            "id": loaded.model_id,
            "requested_revision": loaded.requested_revision,
            "resolved_revision": loaded.resolved_revision,
            "quantization": "4-bit NF4 with double quantization",
            "decoding": {
                "do_sample": False,
                "max_new_tokens": 256,
            },
        },
        "split": split_name,
        "metrics": summarize_records(records),
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return report


def parse_args() -> argparse.Namespace:
    """Parse evaluator command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Qwen prompt-only baseline on a protected split.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Protected dataset split to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional report path. Defaults under artifacts/evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one baseline evaluation and print its aggregate metrics."""

    args = parse_args()

    output_path = args.output or (
        PROJECT_ROOT
        / "artifacts"
        / "evaluation"
        / "prompt_baseline"
        / BASELINE_CONFIG_ID
        / f"{args.split}.json"
    )

    report = evaluate_split(
        split_name=args.split,
        output_path=output_path,
    )

    print()
    print("Baseline metrics:")
    print(json.dumps(report["metrics"], indent=2))
    print()
    print(f"Wrote evaluation artifact: {output_path}")


if __name__ == "__main__":
    main()
