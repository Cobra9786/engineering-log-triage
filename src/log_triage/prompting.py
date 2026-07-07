"""Model-neutral prompting and supervised-training formatting."""

from __future__ import annotations

from log_triage.schema import EngineeringLogExample, IncidentCategory, Severity


CATEGORY_VALUES = ", ".join(f'"{category.value}"' for category in IncidentCategory)
SEVERITY_VALUES = ", ".join(f'"{severity.value}"' for severity in Severity)


SYSTEM_PROMPT = f"""You are an engineering incident triage assistant.

Convert one unstructured engineering report into exactly one JSON object.

Output-format rules:
- Return one raw JSON object only.
- The first character of the response must be "{{".
- The final character of the response must be "}}".
- Never use Markdown, code fences, explanations, headings, prefixes, or suffixes.

The required JSON fields are:
- category: one of {CATEGORY_VALUES}
- severity: one of {SEVERITY_VALUES}
- subsystem: short snake_case subsystem identifier, or "unknown"
- symptoms: a non-empty array of observed symptoms grounded in the report
- suspected_cause: a likely cause, or "insufficient evidence" when unsupported
- recommended_action: a concrete next diagnostic or mitigation action
- requires_human_review: true or false

Category guidance:
- Use "sensor_or_signal_path" for ADC behavior, sensor output faults, imaging
  artifacts, electrical signal integrity, intermittent cables or connectors,
  grounding problems, and data-link faults inside a sensor subsystem.
- Use "communications" for RF, telemetry, packet delivery, RSSI, network-link,
  command-acknowledgement, or transport failures between systems.
- Use "calibration_or_configuration" only when the report indicates an explicit
  calibration, gain, profile, parameter, or approved-configuration mismatch.
- Use "unknown" when the evidence does not support a defensible category.

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
