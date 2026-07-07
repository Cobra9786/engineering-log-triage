from pathlib import Path

from log_triage.dataset import load_jsonl_dataset


DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "seed_examples.jsonl"
)


def test_seed_dataset_loads_and_has_unique_ids() -> None:
    examples = load_jsonl_dataset(DATASET_PATH)

    assert len(examples) == 10
    assert len({example.id for example in examples}) == 10
    assert all(example.source_type == "synthetic_sanitized" for example in examples)


def test_seed_dataset_covers_required_incident_categories() -> None:
    examples = load_jsonl_dataset(DATASET_PATH)
    categories = {example.target.category.value for example in examples}

    assert {
        "sensor_or_signal_path",
        "communications",
        "power_or_battery",
        "firmware_or_software",
        "calibration_or_configuration",
        "mechanical_or_environmental",
        "data_pipeline_or_api",
        "unknown",
    }.issubset(categories)
