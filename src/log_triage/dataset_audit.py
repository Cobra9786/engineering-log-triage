"""Dataset split and category-coverage audit for engineering-log triage."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from log_triage.dataset import load_jsonl_dataset
from log_triage.splits import load_split_manifest, partition_examples


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "seed_examples.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.json"


def main() -> None:
    """Print split membership and category coverage."""

    examples = load_jsonl_dataset(DATASET_PATH)
    manifest = load_split_manifest(MANIFEST_PATH)
    partitions = partition_examples(examples, manifest)

    print(f"Dataset records: {len(examples)}")
    print()

    for split_name in ("train", "validation", "test"):
        split_examples = partitions[split_name]
        counts = Counter(
            example.target.category.value
            for example in split_examples
        )

        print(f"{split_name.upper()} ({len(split_examples)} records)")
        for category, count in sorted(counts.items()):
            print(f"  {category:32} {count}")
        print()


if __name__ == "__main__":
    main()
