"""One controlled local inference smoke test for the Qwen foundation model."""

from __future__ import annotations

import json
import time
from pathlib import Path
from log_triage.output_parsing import (
    parse_json_object_with_recovery,
    parse_strict_json_object,
)
import torch
from pydantic import ValidationError

from log_triage.dataset import load_jsonl_dataset
from log_triage.model_loader import load_qwen_for_inference
from log_triage.prompting import SYSTEM_PROMPT, render_user_prompt
from log_triage.schema import TriageResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "seed_examples.jsonl"


def main() -> None:
    """Load Qwen and run one deterministic engineering-log triage prompt."""

    example = load_jsonl_dataset(DATASET_PATH)[0]
    loaded = load_qwen_for_inference()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(example.report_text)},
    ]

    rendered_prompt = loaded.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = loaded.tokenizer(
        rendered_prompt,
        return_tensors="pt",
    )
    model_inputs = {
        name: value.to(loaded.input_device)
        for name, value in model_inputs.items()
    }

    torch.cuda.synchronize()
    started_at = time.perf_counter()

    with torch.inference_mode():
        generated_ids = loaded.model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=256,
            pad_token_id=loaded.tokenizer.pad_token_id,
        )

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started_at

    completion_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
    response_text = loaded.tokenizer.decode(
        completion_ids[0],
        skip_special_tokens=True,
    ).strip()

    print(f"Model:             {loaded.model_id}")
    print(f"Requested revision:{loaded.requested_revision}")
    print(f"Resolved revision: {loaded.resolved_revision}")
    print(f"Input record:      {example.id}")
    print(f"Latency:           {elapsed_seconds:.2f} seconds")
    print(
        "GPU allocated:     "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
    )
    print()
    print("Raw model response:")
    print(response_text)
    print()


    try:
        strict_payload = parse_strict_json_object(response_text)
        TriageResult.model_validate(strict_payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        print(f"Strict JSON/schema validity: FAILED ({error})")
    else:
        print("Strict JSON/schema validity: PASSED")

    try:
        parsed_output = parse_json_object_with_recovery(response_text)
        validated_result = TriageResult.model_validate(parsed_output.payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        print(f"Recovered JSON/schema validity: FAILED ({error})")
        return

    print(
        "Recovered JSON/schema validity: PASSED "
        f"(markdown-fence recovery used: "
        f"{parsed_output.recovered_from_markdown_fence})"
    )
    print(
        json.dumps(
            validated_result.model_dump(mode="json"),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
