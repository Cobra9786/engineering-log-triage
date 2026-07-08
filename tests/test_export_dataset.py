import json

from log_triage.export_dataset import export_supervised_dataset
from log_triage.schema import TriageResult


def test_export_supervised_dataset_writes_expected_splits(tmp_path) -> None:
    manifest = export_supervised_dataset(tmp_path)

    assert manifest["split_policy"] == "lora_v1"
    assert manifest["splits"] == {
        "train": 48,
        "validation": 7,
        "test": 3,
    }
    assert manifest["total_records"] == 58

    for split_name, expected_count in manifest["splits"].items():
        path = tmp_path / f"{split_name}.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert len(records) == expected_count

        for record in records:
            assert set(record) == {"id", "messages"}
            assert [message["role"] for message in record["messages"]] == [
                "system",
                "user",
                "assistant",
            ]

            assistant_payload = json.loads(record["messages"][-1]["content"])
            TriageResult.model_validate(assistant_payload)


def test_export_manifest_is_written(tmp_path) -> None:
    export_supervised_dataset(tmp_path)

    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["format"] == "chat_messages_jsonl"
    assert manifest["source_dataset"] == "data/raw/seed_examples.jsonl"
    assert manifest["source_manifest"] == "data/manifests/split_manifest.json"
