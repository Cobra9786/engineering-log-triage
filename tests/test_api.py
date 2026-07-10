from fastapi.testclient import TestClient

from log_triage.api import TriageApiResponse, create_app, resolve_adapter_source
from log_triage.schema import TriageResult


class FakeTriageRuntime:
    def triage(self, report_text: str) -> TriageApiResponse:
        assert report_text

        result = TriageResult(
            category="sensor_or_signal_path",
            severity="medium",
            subsystem="sonar_channel_3_adc",
            symptoms=[
                "channel 3 ADC dropout",
                "intermittent saturation",
            ],
            suspected_cause="Possible intermittent signal-path issue.",
            recommended_action="Inspect cable routing and repeat acquisition with a reference unit.",
            requires_human_review=True,
        )

        return TriageApiResponse(
            result=result,
            raw_response=result.model_dump_json(),
            strict_json_schema_valid=True,
            recovered_json_schema_valid=True,
            markdown_fence_recovery_used=False,
            latency_seconds=0.001,
            model_id="fake-model",
            adapter_dir="/tmp/fake-adapter",
        )


def test_health_reports_runtime_not_loaded_when_disabled() -> None:
    app = create_app(load_runtime=False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False


def test_adapter_repo_id_is_used_when_configured(monkeypatch) -> None:
    monkeypatch.delenv("LOG_TRIAGE_ADAPTER_DIR", raising=False)
    monkeypatch.setenv(
        "LOG_TRIAGE_ADAPTER_REPO_ID",
        "cobra9786/engineering-log-triage-qwen-lora",
    )

    assert resolve_adapter_source() == "cobra9786/engineering-log-triage-qwen-lora"


def test_adapter_dir_takes_precedence_over_repo_id(monkeypatch, tmp_path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    monkeypatch.setenv("LOG_TRIAGE_ADAPTER_DIR", str(adapter_dir))
    monkeypatch.setenv(
        "LOG_TRIAGE_ADAPTER_REPO_ID",
        "cobra9786/engineering-log-triage-qwen-lora",
    )

    assert resolve_adapter_source() == str(adapter_dir.resolve())


def test_triage_returns_valid_response_with_injected_runtime() -> None:
    app = create_app(load_runtime=False)
    app.state.runtime = FakeTriageRuntime()

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={
                "report_text": (
                    "Channel 3 ADC drops out after warmup and saturation appears "
                    "intermittently during the bench run."
                ),
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["strict_json_schema_valid"] is True
    assert payload["recovered_json_schema_valid"] is True
    assert payload["markdown_fence_recovery_used"] is False
    assert payload["result"]["category"] == "sensor_or_signal_path"
    assert payload["result"]["severity"] == "medium"
    assert payload["result"]["requires_human_review"] is True


def test_triage_rejects_empty_report_text() -> None:
    app = create_app(load_runtime=False)
    app.state.runtime = FakeTriageRuntime()

    with TestClient(app) as client:
        response = client.post("/triage", json={"report_text": ""})

    assert response.status_code == 422


def test_triage_returns_503_without_runtime() -> None:
    app = create_app(load_runtime=False)

    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={"report_text": "The forward camera shows colored streaks."},
        )

    assert response.status_code == 503
