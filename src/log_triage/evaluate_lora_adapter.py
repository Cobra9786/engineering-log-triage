"""Evaluate a saved LoRA adapter against the active engineering-log triage split."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from peft import PeftModel

from log_triage.baseline import evaluate_example, summarize_records
from log_triage.dataset import load_jsonl_dataset
from log_triage.model_loader import BASE_MODEL_ID, load_qwen_for_inference
from log_triage.splits import load_split_manifest, partition_examples


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "seed_examples.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.json"


def load_qwen_with_adapter(adapter_dir: Path) -> Any:
    """Load base Qwen in 4-bit form and attach a saved LoRA adapter."""

    if not adapter_dir.is_dir():
        raise ValueError(f"Adapter directory does not exist: {adapter_dir}")

    loaded = load_qwen_for_inference()
    loaded.model = PeftModel.from_pretrained(loaded.model, adapter_dir)
    loaded.model.eval()

    return loaded


def evaluate_lora_adapter(
    adapter_dir: Path,
    split_name: str,
    output_path: Path,
    run_id: str,
) -> dict[str, Any]:
    """Evaluate a saved adapter against validation or test."""

    if split_name not in ("validation", "test"):
        raise ValueError("LoRA adapter evaluation is restricted to validation or test.")

    examples = load_jsonl_dataset(DATASET_PATH)
    manifest = load_split_manifest(MANIFEST_PATH)
    partitions = partition_examples(examples, manifest)

    split_examples = partitions[split_name]
    loaded = load_qwen_with_adapter(adapter_dir)

    print(f"Loaded base model: {loaded.model_id}")
    print(f"Resolved revision: {loaded.resolved_revision}")
    print(f"Loaded adapter:    {adapter_dir}")
    print(f"Evaluating split:  {split_name} ({len(split_examples)} examples)")

    records: list[dict[str, Any]] = []

    for index, example in enumerate(split_examples, start=1):
        print(f"[{index}/{len(split_examples)}] Evaluating {example.id}...")
        records.append(evaluate_example(example, loaded))

    report = {
        "evaluation_type": "lora_adapter",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_model": {
            "id": BASE_MODEL_ID,
            "loaded_id": loaded.model_id,
            "requested_revision": loaded.requested_revision,
            "resolved_revision": loaded.resolved_revision,
            "quantization": "4-bit NF4 with double quantization",
        },
        "adapter": {
            "path": str(adapter_dir),
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
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a saved LoRA adapter for engineering-log triage.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Directory containing the saved PEFT/LoRA adapter.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Dataset split to evaluate. Use test only after adapter selection is frozen.",
    )
    parser.add_argument(
        "--run-id",
        default="local_lora_adapter",
        help="Run identifier recorded in the evaluation report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the evaluation JSON report will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate one LoRA adapter and print aggregate metrics."""

    args = parse_args()

    report = evaluate_lora_adapter(
        adapter_dir=args.adapter_dir,
        split_name=args.split,
        output_path=args.output,
        run_id=args.run_id,
    )

    print()
    print("LoRA adapter metrics:")
    print(json.dumps(report["metrics"], indent=2))
    print()
    print(f"Wrote evaluation artifact: {args.output}")


if __name__ == "__main__":
    main()
