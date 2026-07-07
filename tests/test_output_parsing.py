import json

import pytest

from log_triage.output_parsing import (
    parse_json_object_with_recovery,
    parse_strict_json_object,
)


def test_strict_parser_accepts_raw_json_object() -> None:
    payload = parse_strict_json_object('{"category": "communications"}')

    assert payload == {"category": "communications"}


def test_strict_parser_rejects_fenced_json() -> None:
    response = """```json
{"category": "communications"}
```"""

    with pytest.raises(json.JSONDecodeError):
        parse_strict_json_object(response)


def test_recovery_parser_accepts_one_complete_fenced_json_object() -> None:
    response = """```json
{"category": "communications"}
```"""

    parsed = parse_json_object_with_recovery(response)

    assert parsed.payload == {"category": "communications"}
    assert parsed.recovered_from_markdown_fence is True


def test_recovery_parser_rejects_prose_outside_the_json_fence() -> None:
    response = """Here is the result:

```json
{"category": "communications"}
```"""

    with pytest.raises(json.JSONDecodeError):
        parse_json_object_with_recovery(response)
