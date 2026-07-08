"""Export curated engineering-log examples into supervised chat-format JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from log_triage.dataset import load_jsonl_dataset
from log_triage.prompting import format_supervised_messages
from log_triage.schema import TriageResult
from log_triage.splits import SPLIT_NAMES, load_split_manifest, partition_examples


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "seed_examples.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "lora_v1"


def build_supervised_record(example_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Build and validate one model-ready supervised training record."""

    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"Unexpected message roles for {example_id}.")

    assistant_payload = json.loads(messages[-1]["content"])
    TriageResult.model_validate(assistant_payload)

    return {
        "id": example_id,
        "messages": messages,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as deterministic JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def export_supervised_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Export active train, validation, and test splits to supervised JSONL."""

    examples = load_jsonl_dataset(DATASET_PATH)
    manifest_payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    split_manifest = load_split_manifest(MANIFEST_PATH)
    partitions = partition_examples(examples, split_manifest)

    split_counts: dict[str, int] = {}

    for split_name in SPLIT_NAMES:
        split_examples = partitions[split_name]
        split_records = [
            build_supervised_record(
                example_id=example.id,
                messages=format_supervised_messages(example),
            )
            for example in split_examples
        ]

        write_jsonl(output_dir / f"{split_name}.jsonl", split_records)
        split_counts[split_name] = len(split_records)

    export_manifest = {
        "dataset_version": manifest_payload["dataset_version"],
        "split_policy": manifest_payload.get("split_policy", "unspecified"),
        "source_dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "format": "chat_messages_jsonl",
        "splits": split_counts,
        "total_records": sum(split_counts.values()),
    }

    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(export_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return export_manifest


def parse_args() -> argparse.Namespace:
    """Parse export command arguments."""

    parser = argparse.ArgumentParser(
        description="Export active engineering-log triage splits for LoRA training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where train/validation/test JSONL files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Export the active supervised dataset and print a compact summary."""

    args = parse_args()
    manifest = export_supervised_dataset(args.output_dir)

    print(f"Wrote supervised dataset to: {args.output_dir}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
