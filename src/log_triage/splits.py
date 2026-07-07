"""Dataset split-manifest loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from log_triage.schema import EngineeringLogExample


SPLIT_NAMES: Final = ("train", "validation", "test")


def load_split_manifest(path: Path) -> dict[str, list[str]]:
    """Load a split manifest and reject duplicate or malformed assignments."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in split manifest {path}: {error.msg}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"Split manifest must be a JSON object: {path}")

    dataset_version = payload.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("Split manifest must include a non-empty dataset_version.")

    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Split manifest must include a splits object.")

    expected_names = set(SPLIT_NAMES)
    actual_names = set(splits)

    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            f"Split names must be {sorted(expected_names)}; "
            f"missing={missing}, unexpected={unexpected}."
        )

    normalized: dict[str, list[str]] = {}
    seen_ids: set[str] = set()

    for split_name in SPLIT_NAMES:
        example_ids = splits[split_name]

        if not isinstance(example_ids, list) or not example_ids:
            raise ValueError(
                f"Split '{split_name}' must be a non-empty list of example identifiers."
            )

        if not all(isinstance(example_id, str) for example_id in example_ids):
            raise ValueError(f"Split '{split_name}' contains a non-string example identifier.")

        duplicate_ids = {
            example_id
            for example_id in example_ids
            if example_ids.count(example_id) > 1
        }
        if duplicate_ids:
            raise ValueError(
                f"Split '{split_name}' contains duplicate identifiers: {sorted(duplicate_ids)}."
            )

        overlap = seen_ids.intersection(example_ids)
        if overlap:
            raise ValueError(
                f"Split '{split_name}' overlaps another split: {sorted(overlap)}."
            )

        seen_ids.update(example_ids)
        normalized[split_name] = example_ids

    return normalized


def partition_examples(
    examples: list[EngineeringLogExample],
    split_manifest: dict[str, list[str]],
) -> dict[str, list[EngineeringLogExample]]:
    """Partition examples according to a validated split manifest."""

    examples_by_id = {example.id: example for example in examples}
    source_ids = set(examples_by_id)
    manifest_ids = {
        example_id
        for split_ids in split_manifest.values()
        for example_id in split_ids
    }

    missing_from_source = manifest_ids - source_ids
    unassigned_source_ids = source_ids - manifest_ids

    if missing_from_source or unassigned_source_ids:
        raise ValueError(
            "Split manifest does not match source dataset; "
            f"missing_from_source={sorted(missing_from_source)}, "
            f"unassigned_source_ids={sorted(unassigned_source_ids)}."
        )

    return {
        split_name: [examples_by_id[example_id] for example_id in example_ids]
        for split_name, example_ids in split_manifest.items()
    }
