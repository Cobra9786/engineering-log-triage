"""Model-neutral prompting and supervised-training formatting."""

from __future__ import annotations

from log_triage.schema import EngineeringLogExample, IncidentCategory, Severity


CATEGORY_VALUES = ", ".join(f'"{category.value}"' for category in IncidentCategory)
SEVERITY_VALUES = ", ".join(f'"{severity.value}"' for severity in Severity)


SYSTEM_PROMPT = f"""You are an engineering incident triage assistant.

Convert one unstructured engineering report into exactly one JSON object.
Return JSON only: no Markdown, explanation, prefix, or suffix.

The required JSON fields are:
- category: one of {CATEGORY_VALUES}
- severity: one of {SEVERITY_VALUES}
- subsystem: short snake_case subsystem identifier, or "unknown"
- symptoms: a non-empty array of observed symptoms grounded in the report
- suspected_cause: a likely cause, or "insufficient evidence" when unsupported
- recommended_action: a concrete next diagnostic or mitigation action
- requires_human_review: true or false

Do not invent observed facts. Route ambiguous, safety-relevant, or evidence-poor
reports to human review.
"""


def render_user_prompt(report_text: str) -> str:
    """Wrap a raw engineering report in the stable user-message format."""

    normalized_report = report_text.strip()

    if not normalized_report:
        raise ValueError("Engineering report text must not be empty.")

    return f"""Engineering report:

{normalized_report}
"""


def format_supervised_messages(example: EngineeringLogExample) -> list[dict[str, str]]:
    """Format one curated example as generic chat messages for later SFT use."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(example.report_text)},
        {
            "role": "assistant",
            "content": example.target.model_dump_json(),
        },
    ]
