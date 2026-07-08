from pathlib import Path

import pytest

from log_triage.evaluate_lora_adapter import evaluate_lora_adapter


def test_lora_evaluator_rejects_missing_adapter_dir(tmp_path: Path) -> None:
    missing_adapter = tmp_path / "missing-adapter"
    output_path = tmp_path / "report.json"

    with pytest.raises(ValueError, match="Adapter directory does not exist"):
        evaluate_lora_adapter(
            adapter_dir=missing_adapter,
            split_name="validation",
            output_path=output_path,
            run_id="missing-adapter-test",
        )


def test_lora_evaluator_rejects_training_split(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with pytest.raises(ValueError, match="restricted to validation or test"):
        evaluate_lora_adapter(
            adapter_dir=adapter_dir,
            split_name="train",
            output_path=tmp_path / "report.json",
            run_id="invalid-split-test",
        )
