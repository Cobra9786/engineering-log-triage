from pathlib import Path

from log_triage.dataset import load_jsonl_dataset
from log_triage.schema import IncidentCategory


DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "seed_examples.jsonl"
)


def test_seed_dataset_loads_and_has_unique_ids() -> None:
    examples = load_jsonl_dataset(DATASET_PATH)

    assert len(examples) >= 10
    assert len({example.id for example in examples}) == len(examples)
    assert all(example.source_type == "synthetic_sanitized" for example in examples)


def test_seed_dataset_covers_required_incident_categories() -> None:
    examples = load_jsonl_dataset(DATASET_PATH)
    categories = {example.target.category.value for example in examples}

    expected_categories = {category.value for category in IncidentCategory}

    assert expected_categories.issubset(categories)
