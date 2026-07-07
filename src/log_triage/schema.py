"""Typed structured-output contract for engineering log triage."""

from enum import StrEnum

from pydantic import BaseModel, Field


class IncidentCategory(StrEnum):
    SENSOR_OR_SIGNAL_PATH = "sensor_or_signal_path"
    COMMUNICATIONS = "communications"
    POWER_OR_BATTERY = "power_or_battery"
    FIRMWARE_OR_SOFTWARE = "firmware_or_software"
    CALIBRATION_OR_CONFIGURATION = "calibration_or_configuration"
    MECHANICAL_OR_ENVIRONMENTAL = "mechanical_or_environmental"
    DATA_PIPELINE_OR_API = "data_pipeline_or_api"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriageResult(BaseModel):
    """Normalized engineering-triage result produced by a model or reviewer."""

    category: IncidentCategory
    severity: Severity
    subsystem: str = Field(
        min_length=2,
        max_length=120,
        description="Primary affected subsystem.",
    )
    symptoms: list[str] = Field(
        min_length=1,
        max_length=8,
        description="Observed symptoms grounded in the report.",
    )
    suspected_cause: str = Field(
        min_length=3,
        max_length=500,
        description="Likely cause, or 'insufficient evidence' when unknown.",
    )
    recommended_action: str = Field(
        min_length=3,
        max_length=500,
        description="Concrete next diagnostic or mitigation action.",
    )
    requires_human_review: bool
