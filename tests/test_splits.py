from pathlib import Path

from log_triage.dataset import load_jsonl_dataset
from log_triage.splits import load_split_manifest, partition_examples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "seed_examples.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.json"


def test_split_manifest_partitions_every_seed_example_once() -> None:
    examples = load_jsonl_dataset(DATASET_PATH)
    manifest = load_split_manifest(MANIFEST_PATH)
    partitions = partition_examples(examples, manifest)

    assert len(partitions["train"]) == 6
    assert len(partitions["validation"]) == 2
    assert len(partitions["test"]) == 2

    assigned_ids = {
        example.id
        for split_examples in partitions.values()
        for example in split_examples
    }

    assert assigned_ids == {example.id for example in examples}


def test_test_examples_are_not_in_train_or_validation() -> None:
    manifest = load_split_manifest(MANIFEST_PATH)

    test_ids = set(manifest["test"])

    assert test_ids.isdisjoint(manifest["train"])
    assert test_ids.isdisjoint(manifest["validation"])
