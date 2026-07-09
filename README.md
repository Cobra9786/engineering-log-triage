# LoRA-Adapted Engineering Log Triage Service

A reproducible applied-AI project that converts unstructured engineering and sensor-fault reports into typed triage records.

The project demonstrates a full foundation-model adaptation workflow:

* curated engineering-log dataset
* prompt-only baseline
* QLoRA fine-tuning with PEFT/LoRA
* validation and held-out test evaluation
* strict JSON/schema validation
* FastAPI inference service
* GPU-backed Docker deployment
* Hugging Face dataset publishing

## Published Assets

Dataset:

https://huggingface.co/datasets/cobra9786/engineering-log-triage-dataset

Planned Hugging Face assets:

* Dataset repo: `cobra9786/engineering-log-triage-dataset`
* LoRA adapter/model repo: `cobra9786/engineering-log-triage-qwen-lora`
* Docker Space: `cobra9786/engineering-log-triage-api`

## Project Goal

The service reads an unstructured engineering report like:

> The forward camera stream stays connected, but colored streaks appear whenever the cable bundle touches the frame.

It returns a structured triage record:

```json
{
  "category": "sensor_or_signal_path",
  "severity": "medium",
  "subsystem": "forward_camera_data_link",
  "symptoms": [
    "colored streaks during cable contact",
    "signal remains connected",
    "streaks disappear with cable separation"
  ],
  "suspected_cause": "Signal integrity issue downstream from the front camera connector or harness.",
  "recommended_action": "Inspect the connector interface, cable routing, and surrounding enclosure for damage or strain; replace the harness if damaged.",
  "requires_human_review": true
}
```

## Core Output Contract

Every prediction must produce:

* `category`
* `severity`
* `subsystem`
* `symptoms`
* `suspected_cause`
* `recommended_action`
* `requires_human_review`

The output must validate against the project `TriageResult` schema.

Strict JSON is preferred. Markdown-fenced JSON is tracked separately as a recoverable but less API-safe output format.

## Model Stack

Base model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Main tooling:

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT/LoRA
* TRL `SFTTrainer`
* bitsandbytes 4-bit quantization
* FastAPI
* Docker with NVIDIA GPU access

## Dataset

Current dataset version:

```text
0.3.0
```

Split policy:

```text
lora_v1
```

Dataset size:

| Split      | Records |
| ---------- | ------: |
| Train      |      48 |
| Validation |       7 |
| Test       |       3 |
| Total      |      58 |

The dataset is synthetic/sanitized and intended for portfolio-scale model adaptation and evaluation.

The published dataset is available here:

https://huggingface.co/datasets/cobra9786/engineering-log-triage-dataset

## Training

Selected adapter run:

```text
qwen_lora_r8_train48_steps72_v1
```

Local adapter path:

```text
artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter
```

Training setup:

| Setting               | Value |
| --------------------- | ----: |
| LoRA rank             |     8 |
| LoRA alpha            |    16 |
| LoRA dropout          |  0.05 |
| Training records      |    48 |
| Max steps             |    72 |
| Batch size per device |     1 |
| Gradient accumulation |     2 |
| Max sequence length   |  1024 |

LoRA target modules:

* `q_proj`
* `k_proj`
* `v_proj`
* `o_proj`
* `gate_proj`
* `up_proj`
* `down_proj`

Adapter weights are not committed to this GitHub repo. They remain local under `artifacts/training/` until published to a Hugging Face model repo.

## Evaluation Summary

### Validation

| Metric                         | Qwen Baseline | LoRA Adapter |
| ------------------------------ | ------------: | -----------: |
| Strict JSON/schema validity    |        0.0000 |       1.0000 |
| Recovered JSON/schema validity |        0.8571 |       1.0000 |
| Markdown fence recovery used   |        0.8571 |       0.0000 |
| Category accuracy              |        0.4286 |       1.0000 |
| Severity accuracy              |        0.2857 |       0.7143 |
| Requires-human-review accuracy |        0.8571 |       1.0000 |
| All measured fields correct    |        0.1429 |       0.7143 |

### Held-Out Test

| Metric                         | LoRA Adapter |
| ------------------------------ | -----------: |
| Strict JSON/schema validity    |       1.0000 |
| Recovered JSON/schema validity |       1.0000 |
| Markdown fence recovery used   |       0.0000 |
| Category accuracy              |       0.6667 |
| Severity accuracy              |       0.3333 |
| Requires-human-review accuracy |       0.6667 |
| All measured fields correct    |       0.3333 |

The strongest result is reliable strict JSON/schema output. Semantic classification still needs a larger dataset before production-style accuracy claims would be appropriate.

## Local FastAPI Demo

Start the API locally:

```bash
LOG_TRIAGE_ADAPTER_DIR=artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter \
uvicorn log_triage.api:app --host 127.0.0.1 --port 8000
```

Check health:

```bash
curl -sS http://127.0.0.1:8000/health | python -m json.tool
```

Run one triage request:

```bash
curl -sS \
  -X POST http://127.0.0.1:8000/triage \
  -H 'Content-Type: application/json' \
  -d '{
    "report_text": "The forward camera stream stays connected, but colored streaks appear whenever the cable bundle touches the frame. The streaks disappear after the cable is separated and strain-relieved."
  }' | python -m json.tool
```

Expected response fields:

* `result`
* `raw_response`
* `strict_json_schema_valid`
* `recovered_json_schema_valid`
* `markdown_fence_recovery_used`
* `latency_seconds`
* `model_id`
* `adapter_dir`

## Docker GPU Deployment

The FastAPI service can also run inside a GPU-enabled Docker container.

This Docker image is for inference/serving only. Training remains a local workflow. The LoRA adapter is mounted into the container read-only at runtime.

### Verify Docker GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

Expected result: `nvidia-smi` prints the host GPU inside the container.

### Build the API Image

```bash
docker build -t engineering-log-triage-api:local .
```

### Run the API Container

```bash
docker run --rm --name triage-api --gpus all \
  -p 7860:7860 \
  -e LOG_TRIAGE_ADAPTER_DIR=/app/adapter \
  -v "$PWD/artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter:/app/adapter:ro" \
  engineering-log-triage-api:local
```

The container listens on port `7860`, matching the default port used by Hugging Face Docker Spaces.

### Check Docker Health

In another terminal:

```bash
curl -sS http://127.0.0.1:7860/health | python -m json.tool
```

Expected shape:

```json
{
  "status": "ok",
  "model_loaded": true,
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "adapter_dir": "/app/adapter"
}
```

### Run Docker Triage

```bash
curl -sS \
  -X POST http://127.0.0.1:7860/triage \
  -H 'Content-Type: application/json' \
  -d '{
    "report_text": "The forward camera stream stays connected, but colored streaks appear whenever the cable bundle touches the frame. The streaks disappear after the cable is separated and strain-relieved."
  }' | python -m json.tool
```

Expected success indicators:

```json
{
  "strict_json_schema_valid": true,
  "recovered_json_schema_valid": true,
  "markdown_fence_recovery_used": false
}
```

### Stop the Container

If running in the foreground, press:

```text
Ctrl+C
```

Or from another terminal:

```bash
docker rm -f triage-api 2>/dev/null || true
```

On a 6 GB GPU, run only one model server at a time. Do not run the local venv FastAPI server and the Docker FastAPI server simultaneously, because both load Qwen and the LoRA adapter into GPU memory.

## Hugging Face Publishing Plan

Current status:

* Dataset repo: published
* Adapter/model repo: planned
* Docker Space: planned

Publishing order:

1. Publish dataset repo.
2. Publish LoRA adapter model repo.
3. Update runtime to optionally load adapter from Hugging Face Hub.
4. Create Docker Space.
5. Test public `/health` and `/triage`.
6. Update GitHub README with final Hub links.

## Limitations

This is a compact portfolio-scale project.

Known limitations:

* dataset is small
* examples are synthetic/sanitized
* held-out test split has only 3 examples
* severity classification needs more examples
* human-review policy needs more examples
* adapter is not yet published to Hugging Face Hub
* public Docker Space is not yet deployed

The current project should not be presented as production-ready. It should be presented as a reproducible applied-AI pipeline with clear evaluation and deployment boundaries.

## Repository Role

This GitHub repo contains:

* source code
* tests
* dataset construction and export scripts
* training scripts
* evaluation scripts
* FastAPI service
* Dockerfile
* evaluation artifacts
* documentation

The local adapter weights under `artifacts/training/` are intentionally ignored and not committed.
