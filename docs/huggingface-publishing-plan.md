# Hugging Face Publishing Plan

## Purpose

This document defines the planned Hugging Face Hub publishing layout for the LoRA-Adapted Engineering Log Triage Service.

The goal is to separate source code, dataset artifacts, model artifacts, and demo deployment into clean, versioned assets.

## Planned Hugging Face Assets

### 1. Dataset Repository

Proposed repo:

```text
cobra9786/engineering-log-triage-dataset
```

Repository type:

```text
dataset
```

Purpose:

Publish the curated and processed engineering-log triage dataset used for supervised fine-tuning and evaluation.

Planned contents:

```text
train.jsonl
validation.jsonl
test.jsonl
dataset_manifest.json
README.md
```

Source files in this GitHub repo:

```text
data/processed/lora_v1/train.jsonl
data/processed/lora_v1/validation.jsonl
data/processed/lora_v1/test.jsonl
data/processed/lora_v1/dataset_manifest.json
```

The dataset card should describe:

* dataset purpose;
* synthetic/sanitized nature of the examples;
* split policy;
* schema fields;
* category labels;
* limitations;
* non-production status.

### 2. LoRA Adapter Model Repository

Proposed repo:

```text
cobra9786/engineering-log-triage-qwen-lora
```

Repository type:

```text
model
```

Purpose:

Publish the trained PEFT/LoRA adapter selected from validation evaluation.

Base model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Selected local adapter run:

```text
qwen_lora_r8_train48_steps72_v1
```

Local source directory:

```text
artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter
```

Planned contents:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
README.md
```

The model card should describe:

* base model;
* LoRA/QLoRA setup;
* training dataset version;
* validation metrics;
* held-out test metrics;
* known limitations;
* intended use;
* non-production status.

Adapter weights should remain out of the GitHub source repo and be published only as a Hugging Face model artifact.

### 3. Docker Space

Proposed repo:

```text
cobra9786/engineering-log-triage-api
```

Repository type:

```text
space
```

SDK:

```text
docker
```

Purpose:

Provide a public demo surface for the FastAPI triage service.

Planned Space behavior:

* container listens on port `7860`;
* app exposes `/health`;
* app exposes `POST /triage`;
* Space loads the base Qwen model;
* Space loads the LoRA adapter from the Hugging Face model repo;
* Space returns strict schema-validated JSON.

Current local Docker behavior:

```text
LOG_TRIAGE_ADAPTER_DIR=/app/adapter
```

Future Space behavior:

```text
LOG_TRIAGE_ADAPTER_REPO_ID=cobra9786/engineering-log-triage-qwen-lora
```

The Space should eventually download the adapter from the model repo at startup rather than relying on a local bind mount.

## GitHub Repository Role

This GitHub repo remains the source-of-truth for:

```text
source code
tests
dataset construction
training scripts
evaluation scripts
FastAPI service
Dockerfile
documentation
evaluation reports
```

GitHub should not contain local adapter weights under:

```text
artifacts/training/
```

Those stay ignored locally and are published separately to the Hugging Face model repo.

## Publishing Order

Recommended order:

1. Publish dataset repo.
2. Publish LoRA adapter model repo.
3. Update FastAPI runtime to optionally load adapter from Hugging Face Hub.
4. Create Docker Space.
5. Test public `/health` and `/triage`.
6. Update GitHub README with Hugging Face links.

## Initial Dataset Publishing Commands

After Hugging Face login is configured:

```bash
hf auth login

hf repo create cobra9786/engineering-log-triage-dataset \
  --type dataset \
  --public

hf upload cobra9786/engineering-log-triage-dataset \
  data/processed/lora_v1 \
  . \
  --repo-type dataset
```

## Initial Adapter Publishing Commands

After the selected adapter directory is confirmed locally:

```bash
hf repo create cobra9786/engineering-log-triage-qwen-lora \
  --type model \
  --public

hf upload cobra9786/engineering-log-triage-qwen-lora \
  artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter \
  . \
  --repo-type model
```

## Initial Docker Space Publishing Direction

Create a Hugging Face Space repo:

```bash
hf repo create cobra9786/engineering-log-triage-api \
  --type space \
  --space_sdk docker \
  --public
```

The Space repo should include:

```text
README.md
Dockerfile
requirements-docker.txt
src/
```

The Space README should include YAML metadata similar to:

```yaml
---
title: Engineering Log Triage API
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
```

## Before Publishing Checklist

* GitHub repo is clean.
* Dataset version is confirmed.
* Evaluation artifacts are committed.
* Adapter directory exists locally.
* Adapter is not committed to GitHub.
* Docker local deployment works.
* `/health` works in Docker.
* `/triage` works in Docker.
* Model card and dataset card text are reviewed.
* No private credentials or local notes are included.

## Current Status

Local FastAPI deployment: complete.

Local Docker GPU deployment: complete.

Next publishing step: prepare the dataset card and publish the dataset repo.
