# Training Data Expansion

## Purpose

The initial ten examples established the typed contract, fixed split membership,
and prompt-only baseline. They were not sufficient to train a credible LoRA
adapter.

Training Expansion A adds synthetic and sanitized examples designed to improve
coverage of category boundaries and safety conventions.

## Important boundaries

- Sensor or signal-path faults versus external communications failures.
- Explicit calibration or configuration mismatches versus unknown incidents.
- Safety-critical water ingress versus lower-severity mechanical observations.
- Firmware behavior versus configuration-state behavior.
- API/data-ordering failures versus device-side faults.

## Split policy

Validation and held-out test examples remain fixed.

New curated records are added only to the training split until the first LoRA
adapter experiment is complete. The model is never trained on validation or
held-out test text.
