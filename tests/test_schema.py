from log_triage.schema import IncidentCategory, Severity, TriageResult


def test_valid_triage_result_serializes_to_expected_json() -> None:
    result = TriageResult(
        category=IncidentCategory.SENSOR_OR_SIGNAL_PATH,
        severity=Severity.HIGH,
        subsystem="acquisition_controller",
        symptoms=[
            "intermittent channel 3 dropout",
            "ADC saturation during thermal soak",
        ],
        suspected_cause="Possible thermal instability in the channel 3 analog path.",
        recommended_action=(
            "Inspect the channel 3 analog path, verify supply rails, "
            "and repeat the thermal soak test."
        ),
        requires_human_review=True,
    )

    payload = result.model_dump(mode="json")

    assert payload["category"] == "sensor_or_signal_path"
    assert payload["severity"] == "high"
    assert payload["requires_human_review"] is True


def test_empty_symptoms_are_rejected() -> None:
    invalid_payload = {
        "category": "communications",
        "severity": "medium",
        "subsystem": "telemetry_link",
        "symptoms": [],
        "suspected_cause": "insufficient evidence",
        "recommended_action": "Collect link-quality and packet-loss telemetry.",
        "requires_human_review": True,
    }

    try:
        TriageResult.model_validate(invalid_payload)
    except ValueError:
        return

    raise AssertionError("Expected empty symptoms to be rejected.")
