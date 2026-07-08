# LoRA-Adapted Engineering Log Triage Service — Model Card v1

## Summary

This project adapts an open-weight instruction model for structured classification of engineering and sensor fault reports.

The model takes an unstructured engineering report and returns a strict JSON object matching the project schema:

- `category`
- `severity`
- `subsystem`
- `symptoms`
- `suspected_cause`
- `recommended_action`
- `requires_human_review`

The goal is not to claim production-grade triage accuracy from a small dataset. The goal is to demonstrate a reproducible foundation-model adaptation workflow:

1. curated dataset creation;
2. train/validation/test split discipline;
3. prompt-only baseline evaluation;
4. QLoRA adapter training;
5. strict JSON/schema validation;
6. held-out evaluation;
7. preparation for API deployment.

## Base Model

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Loading mode: 4-bit NF4 quantization with double quantization
- Runtime stack:
  - PyTorch
  - Transformers
  - Datasets
  - PEFT/LoRA
  - TRL `SFTTrainer`
  - bitsandbytes

The base model is loaded locally and adapted with LoRA adapters rather than full-parameter fine-tuning.

## Dataset

Dataset type: synthetic/sanitized engineering-log examples.

Current dataset version: `0.3.0`

Split policy: `lora_v1`

| Split | Records |
|---|---:|
| Train | 48 |
| Validation | 7 |
| Test | 3 |
| Total | 58 |

The dataset includes fault reports across these categories:

- `sensor_or_signal_path`
- `communications`
- `power_or_battery`
- `firmware_or_software`
- `calibration_or_configuration`
- `mechanical_or_environmental`
- `data_pipeline_or_api`
- `unknown`

The validation set was used to select the adapter candidate. The test split was held out until after adapter selection.

## Output Schema

The model is evaluated against strict JSON output requirements. A response is considered strictly valid only when:

- the response is raw JSON;
- the first character is `{`;
- the final character is `}`;
- there are no Markdown fences;
- the JSON validates against the `TriageResult` schema;
- enum values match the allowed schema values.

The evaluator also records a recovered JSON validity metric for Markdown-fenced JSON, but strict JSON validity is the preferred API-readiness metric.

## Prompt-Only Baseline

The prompt-only baseline used the base Qwen model with the same task prompt and greedy decoding.

On the active 7-record validation split:

| Metric | Qwen Prompt Baseline |
|---|---:|
| Strict JSON/schema validity | 0.0000 |
| Recovered JSON/schema validity | 0.8571 |
| Markdown fence recovery used | 0.8571 |
| Category accuracy | 0.4286 |
| Severity accuracy | 0.2857 |
| Requires-human-review accuracy | 0.8571 |
| All measured fields correct | 0.1429 |
| Mean latency seconds | 2.0247 |

The baseline frequently produced Markdown-fenced JSON. That was recoverable for evaluation, but not ideal for an API boundary.

## Selected LoRA Adapter

Selected adapter run:

`qwen_lora_r8_train48_steps72_v1`

Training setup:

| Setting | Value |
|---|---:|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Training records | 48 |
| Max steps | 72 |
| Batch size per device | 1 |
| Gradient accumulation | 2 |
| Max sequence length | 1024 |
| Precision flags | `fp16=False`, `bf16=False` |
| Output overwrite policy | training output directories are protected from accidental overwrite |

LoRA target modules:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

Adapter weights are local training artifacts and are not committed to Git. Evaluation reports are committed separately under `artifacts/evaluation/`.

## Validation Results

Validation split: 7 examples.

| Metric | Qwen Baseline | LoRA Adapter |
|---|---:|---:|
| Strict JSON/schema validity | 0.0000 | 1.0000 |
| Recovered JSON/schema validity | 0.8571 | 1.0000 |
| Markdown fence recovery used | 0.8571 | 0.0000 |
| Category accuracy | 0.4286 | 1.0000 |
| Severity accuracy | 0.2857 | 0.7143 |
| Requires-human-review accuracy | 0.8571 | 1.0000 |
| All measured fields correct | 0.1429 | 0.7143 |
| Mean latency seconds | 2.0247 | 3.1859 |

The selected adapter significantly improved strict JSON compliance and validation classification quality over the prompt-only baseline.

## Held-Out Test Results

Test split: 3 examples.

| Metric | LoRA Adapter |
|---|---:|
| Strict JSON/schema validity | 1.0000 |
| Recovered JSON/schema validity | 1.0000 |
| Markdown fence recovery used | 0.0000 |
| Category accuracy | 0.6667 |
| Severity accuracy | 0.3333 |
| Requires-human-review accuracy | 0.6667 |
| All measured fields correct | 0.3333 |
| Mean latency seconds | 3.0377 |

The held-out test result shows that the adapter learned the structured output contract reliably. Semantic classification, especially severity and human-review policy, still needs a larger and more diverse dataset.

Because the test split contains only 3 examples, each incorrect field changes the metric by 33.3 percentage points. These test metrics should be treated as directional evidence, not production accuracy estimates.

## Interpretation

The strongest result is strict JSON/schema compliance.

The LoRA adapter moved from a prompt-only baseline that usually required Markdown-fence recovery to a selected adapter that produced raw schema-valid JSON on both validation and test splits.

The adapter also improved validation category accuracy and complete measured-field correctness. However, held-out semantic generalization remains limited by dataset size.

A fair project claim is:

> Built a reproducible Hugging Face QLoRA training and evaluation pipeline that improved strict JSON compliance and validation triage accuracy over a prompt-only Qwen baseline on a compact engineering-log dataset.

A fair limitation is:

> The current dataset is too small for production accuracy claims. Additional examples are needed, especially for severity calibration, human-review policy, water-ingress cases, ambiguous low-evidence reports, and configuration-versus-software distinctions.

## Known Limitations

- The dataset is compact and synthetic/sanitized.
- The held-out test split has only 3 examples.
- Severity accuracy remains weak on held-out test.
- The model is not production-ready.
- Adapter weights are not currently published to Hugging Face Hub.
- No FastAPI inference service is included yet.
- No Docker deployment image is included yet.
- No constrained decoding or JSON repair layer is implemented yet.

## Recommended Next Improvements

1. Expand the dataset to at least:
   - 80 to 120 training examples;
   - 12 to 20 validation examples;
   - 12 to 20 held-out test examples.

2. Add more hard examples for:
   - critical water ingress;
   - battery compartment moisture;
   - ambiguous low-evidence warnings;
   - configuration/profile/table mismatch;
   - signal-path faults caused by cable movement;
   - communications versus sensor-link distinction.

3. Add a FastAPI inference service:
   - `POST /triage`;
   - typed request/response schemas;
   - load base model and LoRA adapter once at startup;
   - validate every response before returning it.

4. Add Docker-based local deployment.

5. Publish the dataset and adapter to Hugging Face Hub once the project boundary is stable.

## Project Status

Current status: adapter training and evaluation pipeline complete.

Next engineering milestone: FastAPI inference service with typed request/response validation.
