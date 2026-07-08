from pathlib import Path

from log_triage.dataset import load_jsonl_dataset
from log_triage.schema import IncidentCategory
from log_triage.splits import load_split_manifest, partition_examples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "seed_examples.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.json"
HISTORICAL_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "manifests" / "prompt_baseline_v1_manifest.json"
)


def test_active_manifest_partitions_every_dataset_example_once() -> None:
    examples = load_jsonl_dataset(DATASET_PATH)
    manifest = load_split_manifest(MANIFEST_PATH)
    partitions = partition_examples(examples, manifest)

    assigned_ids = {
        example.id
        for split_examples in partitions.values()
        for example in split_examples
    }

    assert assigned_ids == {example.id for example in examples}
    assert len(partitions["train"]) == 48
    assert len(partitions["validation"]) == 7
    assert len(partitions["test"]) == 3


def test_active_test_ids_are_not_in_train_or_validation() -> None:
    manifest = load_split_manifest(MANIFEST_PATH)

    test_ids = set(manifest["test"])

    assert test_ids.isdisjoint(manifest["train"])
    assert test_ids.isdisjoint(manifest["validation"])


def test_active_training_split_covers_every_incident_category() -> None:
    examples = load_jsonl_dataset(DATASET_PATH)
    manifest = load_split_manifest(MANIFEST_PATH)
    partitions = partition_examples(examples, manifest)

    training_categories = {
        example.target.category.value
        for example in partitions["train"]
    }

    expected_categories = {category.value for category in IncidentCategory}

    assert training_categories == expected_categories


def test_historical_prompt_baseline_manifest_is_preserved() -> None:
    historical_manifest = load_split_manifest(HISTORICAL_MANIFEST_PATH)

    assert historical_manifest["test"] == ["ENG-0007", "ENG-0008"]
    assert historical_manifest["validation"] == ["ENG-0009", "ENG-0010"]
