# LoRA-Adapted Engineering Log Triage Service

A reproducible applied-AI project that converts unstructured engineering
and sensor-fault reports into typed triage records.

## Project goals

- Establish a prompt-only structured-output baseline.
- Curate a synthetic and sanitized engineering-log dataset.
- Fine-tune a compact open-weight model with PEFT/LoRA.
- Compare baseline and adapter performance on held-out test cases.
- Measure JSON validity, schema conformance, field-level accuracy,
  latency, and human-review routing.
- Package local inference behind a typed API.
- Publish the dataset, model adapter, evaluation artifacts, and demo.

## Core contract

Every prediction must produce:

- category
- severity
- subsystem
- symptoms
- suspected cause
- recommended action
- requires_human_review

## Current status

Phase 1: typed output contract and test foundation.
