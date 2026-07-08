# Dataset Split Policy

## Record storage

Each curated engineering incident exists once in:

`data/raw/seed_examples.jsonl`

A record's split role is assigned by identifier in the active manifest:

`data/manifests/split_manifest.json`

## Split roles

### Train

Training records are converted into chat-format prompt and target pairs.
The LoRA adapter receives gradient updates from these records.

### Validation

Validation records are never used for gradient updates. They are used to:

- compare candidate LoRA configurations;
- choose an epoch or checkpoint;
- inspect structured-output compliance;
- identify systematic model errors.

### Test

Test records are held out from training and validation-based model selection.
They are used only after the adapter configuration is frozen.

## Historical baseline split

`prompt_baseline_v1_manifest.json` preserves the split membership used for
the original Qwen prompt-only baseline.

The initial baseline test records were inspected during project development.
They are retained as historical regression cases, not as the final LoRA test
benchmark.

## Active LoRA split policy

The active manifest assigns newly curated records to train, validation, or
test before any LoRA training begins.

Each dataset identifier must appear exactly once across the three splits.
