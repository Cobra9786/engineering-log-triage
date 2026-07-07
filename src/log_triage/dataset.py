"""Dataset loading and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from log_triage.schema import EngineeringLogExample


def load_jsonl_dataset(path: Path) -> list[EngineeringLogExample]:
    """Load a non-empty JSONL dataset and validate every record."""

    examples: list[EngineeringLogExample] = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {path} at line {line_number}: {error.msg}"
            ) from error

        try:
            examples.append(EngineeringLogExample.model_validate(payload))
        except ValidationError as error:
            raise ValueError(
                f"Invalid dataset record in {path} at line {line_number}: {error}"
            ) from error

    if not examples:
        raise ValueError(f"Dataset is empty: {path}")

    return examples
