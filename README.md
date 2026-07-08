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

cat >> README.md <<'EOF'

## Demo: LoRA-Adapted Engineering Log Triage API

This project adapts `Qwen/Qwen2.5-1.5B-Instruct` with a PEFT/LoRA adapter for structured engineering-log triage.

The API accepts an unstructured engineering or sensor fault report and returns a validated JSON response matching the project `TriageResult` schema.

### Current local adapter

Selected local adapter run:

```text
artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter

## Docker GPU Deployment

The FastAPI service can also run inside a GPU-enabled Docker container.

This Docker image is for inference/serving only. Training remains a local workflow. The LoRA adapter is mounted into the container read-only at runtime.

### Verify Docker GPU access

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi


Expected result: nvidia-smi prints the host GPU inside the container.

### Build the API image

docker build -t engineering-log-triage-api:local .
Run the API container
docker run --rm --name triage-api --gpus all \
  -p 7860:7860 \
  -e LOG_TRIAGE_ADAPTER_DIR=/app/adapter \
  -v "$PWD/artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter:/app/adapter:ro" \
  engineering-log-triage-api:local

The container listens on port 7860, matching the default port used by Hugging Face Docker Spaces.

The adapter is mounted from the local training artifact directory:

artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter

Adapter weights are not committed to Git.

### Check health

In another terminal:

curl -sS http://127.0.0.1:7860/health | python -m json.tool

Expected shape:

{
  "status": "ok",
  "model_loaded": true,
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "adapter_dir": "/app/adapter"
}
### Run a triage request
curl -sS \
  -X POST http://127.0.0.1:7860/triage \
  -H 'Content-Type: application/json' \
  -d '{
    "report_text": "The forward camera stream stays connected, but colored streaks appear whenever the cable bundle touches the frame. The streaks disappear after the cable is separated and strain-relieved."
  }' | python -m json.tool

## Expected response shape:

{
  "result": {
    "category": "sensor_or_signal_path",
    "severity": "medium",
    "subsystem": "forward_camera_data_link",
    "symptoms": [
      "colored streaks during cable contact",
      "signal remains connected",
      "streaks disappear with cable separation"
    ],
    "suspected_cause": "...",
    "recommended_action": "...",
    "requires_human_review": true
  },
  "raw_response": "{...}",
  "strict_json_schema_valid": true,
  "recovered_json_schema_valid": true,
  "markdown_fence_recovery_used": false,
  "latency_seconds": 4.3,
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "adapter_dir": "/app/adapter"
}
## Stop the container

If running in the foreground, press:

Ctrl+C

Or from another terminal:

docker rm -f triage-api 2>/dev/null || true

## Notes

On a 6 GB GPU, run only one model server at a time. 
Do not run the local venv 
FastAPI server and the Docker FastAPI server 
simultaneously, because both will load Qwen 
and the LoRA adapter into GPU memory.




1. Verify Docker GPU access

# 2. Build image
docker build --no-cache -t engineering-log-triage-api:local .
# 3. Run container with mounted LoRA adapter
docker run --rm --name triage-api --gpus all \
  -p 7860:7860 \
  -e LOG_TRIAGE_ADAPTER_DIR=/app/adapter \
  -v "$PWD/artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter:/app/adapter:ro" \
  engineering-log-triage-api:local
# 4. Call /health
curl -sS http://127.0.0.1:7860/health | python -m json.tool

# 5. Call /triage
docker run --rm --name triage-api --gpus all \
  -p 7860:7860 \
  -e LOG_TRIAGE_ADAPTER_DIR=/app/adapter \
  -v "$PWD/artifacts/training/qwen_lora_r8_train48_steps72_v1/adapter:/app/adapter:ro" \
  engineering-log-triage-api:local

curl -sS \
  -X POST http://127.0.0.1:7860/triage \
  -H 'Content-Type: application/json' \
  -d '{
    "report_text": "The forward camera stream stays connected, but colored streaks appear whenever the cable bundle touches the frame. The streaks disappear after the cable is separated and strain-relieved."
  }' | python -m json.tool


