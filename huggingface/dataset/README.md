---
license: apache-2.0
language:
- en
task_categories:
- text-classification
- text-generation
pretty_name: Engineering Log Triage Dataset
size_categories:
- n<1K
tags:
- engineering
- fault-triage
- structured-output
- json-schema
- synthetic-data
- lora
- qlora
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# Engineering Log Triage Dataset

## Summary

This dataset contains synthetic/sanitized engineering-log examples for structured fault triage.

Each example is formatted as a chat-style supervised fine-tuning record. The model input is an unstructured engineering report. The target assistant message is a strict JSON object containing a structured triage result.

This dataset was created for the **LoRA-Adapted Engineering Log Triage Service** project.

## Intended Use

The dataset is intended for small-scale supervised fine-tuning and evaluation of instruction models on engineering fault-report triage.

The expected model behavior is to convert an unstructured report into validated JSON with these fields:

- `category`
- `severity`
- `subsystem`
- `symptoms`
- `suspected_cause`
- `recommended_action`
- `requires_human_review`

## Dataset Structure

Files:

| File | Split | Records |
|---|---:|---:|
| `train.jsonl` | train | 48 |
| `validation.jsonl` | validation | 7 |
| `test.jsonl` | test | 3 |
| `dataset_manifest.json` | metadata | n/a |

Total records: 58

Each JSONL row has this shape:

```json
{
  "id": "ENG-0001",
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "Engineering report:\n\n..."
    },
    {
      "role": "assistant",
      "content": "{\"category\":\"sensor_or_signal_path\",...}"
    }
  ]
}
```

## Label Schema

### Categories

Allowed `category` values:

- `sensor_or_signal_path`
- `communications`
- `power_or_battery`
- `firmware_or_software`
- `calibration_or_configuration`
- `mechanical_or_environmental`
- `data_pipeline_or_api`
- `unknown`

### Severity

Allowed `severity` values:

- `low`
- `medium`
- `high`
- `critical`

### Human Review

`requires_human_review` is a boolean field indicating whether the generated triage result should be escalated for human inspection.

## Data Creation

The examples are synthetic/sanitized and designed to mimic engineering, robotics, sensor, telemetry, firmware, configuration, mechanical, environmental, and data-pipeline fault reports.

The dataset was curated to support:

- prompt-only baseline evaluation
- LoRA/QLoRA supervised fine-tuning
- strict JSON/schema validation
- train/validation/test split discipline
- evaluation of output contract adherence

## Split Policy

Current dataset version: `0.3.0`

Split policy: `lora_v1`

The validation split was used for adapter selection. The test split was held out until after the selected adapter was chosen.

## Limitations

This is a compact portfolio-scale dataset.

Known limitations:

- small total record count
- synthetic/sanitized examples
- small held-out test split
- not representative of all engineering domains
- not production validated
- severity classification needs more examples
- human-review policy needs more examples

The dataset should not be used to claim production-grade fault triage accuracy.

## Related Project

This dataset supports a LoRA-adapted engineering-log triage service using:

- `Qwen/Qwen2.5-1.5B-Instruct`
- Hugging Face Transformers
- Datasets
- PEFT/LoRA
- TRL `SFTTrainer`
- bitsandbytes 4-bit quantization
- FastAPI
- Docker GPU serving

## Evaluation Context

The dataset was used to compare a prompt-only Qwen baseline against a LoRA-adapted model.

The selected LoRA adapter improved strict JSON/schema compliance on validation and produced strict schema-valid JSON on the held-out test split.

Because the dataset is small, these results should be treated as directional evidence for a portfolio project rather than production accuracy estimates.

## Citation

No formal citation is available. For portfolio or demonstration use, cite this dataset by its Hugging Face repository name.
