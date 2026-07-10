# Hugging Face Space Deployment Note

The Engineering Log Triage API was staged and locally tested as a Docker Space package.

Local Space-style Docker validation passed using this image:

    engineering-log-triage-space:local

The container successfully loaded the published LoRA adapter from Hugging Face Hub using:

    LOG_TRIAGE_ADAPTER_REPO_ID=cobra9786/engineering-log-triage-qwen-lora

Public Hugging Face Space creation was not completed because Hugging Face returned:

    402 Payment Required

The returned message said:

    Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription.

Current published Hugging Face assets:

- Dataset: https://huggingface.co/datasets/cobra9786/engineering-log-triage-dataset
- Adapter: https://huggingface.co/cobra9786/engineering-log-triage-qwen-lora

Current deployment status:

- FastAPI local inference: validated
- Docker local inference: validated
- Docker image loading adapter from Hugging Face Hub: validated
- Public Hugging Face Docker Space: blocked by subscription requirement

Resume-safe wording:

Packaged inference behind a FastAPI service with Docker-based local deployment and Hugging Face Hub adapter loading.
