"""Strict and recoverable parsing for model-generated JSON output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


FENCED_JSON_PATTERN = re.compile(
    r"^```(?:json)?\s*\n(?P<payload>\{.*\})\s*```$",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedModelOutput:
    """A parsed JSON payload plus whether fenced-JSON recovery was required."""

    payload: dict[str, Any]
    recovered_from_markdown_fence: bool


def parse_strict_json_object(response_text: str) -> dict[str, Any]:
    """Parse a response only when it is raw JSON with a top-level object."""

    payload = json.loads(response_text.strip())

    if not isinstance(payload, dict):
        raise ValueError("Model response must be a top-level JSON object.")

    return payload


def parse_json_object_with_recovery(response_text: str) -> ParsedModelOutput:
    """Parse raw JSON, or recover exactly one complete fenced JSON object.

    Recovery is deliberately narrow: it accepts only a response consisting
    entirely of one Markdown code fence containing one JSON object.
    """

    try:
        return ParsedModelOutput(
            payload=parse_strict_json_object(response_text),
            recovered_from_markdown_fence=False,
        )
    except json.JSONDecodeError as strict_error:
        match = FENCED_JSON_PATTERN.fullmatch(response_text.strip())

        if match is None:
            raise strict_error

        payload = json.loads(match.group("payload"))

        if not isinstance(payload, dict):
            raise ValueError("Recovered model response must be a JSON object.")

        return ParsedModelOutput(
            payload=payload,
            recovered_from_markdown_fence=True,
        )
