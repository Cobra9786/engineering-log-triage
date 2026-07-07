from log_triage.baseline import summarize_records


def test_summary_distinguishes_strict_and_recovered_validity() -> None:
    records = [
        {
            "latency_seconds": 1.0,
            "strict_json_schema_valid": False,
            "recovered_json_schema_valid": True,
            "markdown_fence_recovery_used": True,
            "field_correctness": {
                "category": False,
                "severity": True,
                "requires_human_review": True,
            },
        },
        {
            "latency_seconds": 3.0,
            "strict_json_schema_valid": True,
            "recovered_json_schema_valid": True,
            "markdown_fence_recovery_used": False,
            "field_correctness": {
                "category": True,
                "severity": True,
                "requires_human_review": False,
            },
        },
    ]

    summary = summarize_records(records)

    assert summary["total_examples"] == 2
    assert summary["strict_json_schema_valid"]["rate"] == 0.5
    assert summary["recovered_json_schema_valid"]["rate"] == 1.0
    assert summary["markdown_fence_recovery_used"]["rate"] == 0.5
    assert summary["field_accuracy"]["category"]["accuracy"] == 0.5
    assert summary["field_accuracy"]["severity"]["accuracy"] == 1.0
    assert summary["field_accuracy"]["requires_human_review"]["accuracy"] == 0.5
    assert summary["latency_seconds"]["mean"] == 2.0
