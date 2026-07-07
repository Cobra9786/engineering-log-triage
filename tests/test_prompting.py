import json
from pathlib import Path

import pytest

from log_triage.dataset import load_jsonl_dataset
from log_triage.prompting import (
    SYSTEM_PROMPT,
    format_supervised_messages,
    render_user_prompt,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "seed_examples.jsonl"
)


def test_supervised_messages_preserve_the_curated_target() -> None:
    example = load_jsonl_dataset(DATASET_PATH)[0]

    messages = format_supervised_messages(example)

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert example.report_text in messages[1]["content"]

    assistant_payload = json.loads(messages[2]["content"])

    assert assistant_payload == example.target.model_dump(mode="json")


def test_empty_report_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        render_user_prompt("   ")
