from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_CHOICES, load_config, results_root
from src.data import load_samples
from src.prompts import build_prompt


def _model_context_limit(model_id: str, config: dict[str, Any]) -> int:
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        model_id,
        revision=config["models"]["revision"],
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
    )
    candidates = [
        getattr(model_config, name, None)
        for name in (
            "max_position_embeddings",
            "model_max_length",
            "max_sequence_length",
        )
    ]
    valid = [int(value) for value in candidates if value is not None]
    if not valid:
        raise ValueError(f"Cannot determine context limit for {model_id}")
    return max(valid)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate prompt/output context budgets before a long run"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="gsm8k")
    args = parser.parse_args()
    config = load_config(args.config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["base"],
        revision=config["models"]["revision"],
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
    )
    samples = load_samples(config, args.dataset)
    prompt_lengths = [
        len(
            build_prompt(
                tokenizer,
                sample["question"],
                bool(config["generation"]["enable_thinking"]),
                sample.get("system_message"),
            )["input_token_ids"]
        )
        for sample in samples
    ]
    max_input_tokens = max(prompt_lengths, default=0)
    max_new_tokens = int(config["generation"]["max_new_tokens"])
    required_context = max_input_tokens + max_new_tokens
    base_context = _model_context_limit(config["models"]["base"], config)
    real_context = _model_context_limit(config["models"]["real_gptq"], config)
    configured_real_context = int(
        config["quantization"]["real"]["max_model_len"]
    )

    errors = []
    if required_context > base_context:
        errors.append(
            f"Required context {required_context} exceeds BF16 model limit "
            f"{base_context}"
        )
    if required_context > real_context:
        errors.append(
            f"Required context {required_context} exceeds GPTQ model limit "
            f"{real_context}"
        )
    if required_context > configured_real_context:
        errors.append(
            f"Required context {required_context} exceeds configured Real "
            f"max_model_len {configured_real_context}"
        )
    if configured_real_context > real_context:
        errors.append(
            f"Configured Real max_model_len {configured_real_context} exceeds "
            f"checkpoint limit {real_context}"
        )
    if int(config["generation"]["batch_size"]) != 1:
        errors.append("The validated 39K protocol requires generation.batch_size=1")
    stop_token_ids = [
        int(item) for item in config["generation"].get("stop_token_ids", [])
    ]
    expected_qwen_stops = [151645, 151643]
    if stop_token_ids != expected_qwen_stops:
        errors.append(
            "The validated Qwen3 protocol requires ordered stop_token_ids "
            f"{expected_qwen_stops}, found {stop_token_ids}"
        )
    report = {
        "valid": not errors,
        "dataset": args.dataset,
        "samples": len(samples),
        "min_input_tokens": min(prompt_lengths, default=0),
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "required_context": required_context,
        "bf16_model_context_limit": base_context,
        "gptq_model_context_limit": real_context,
        "configured_real_max_model_len": configured_real_context,
        "stop_token_ids": stop_token_ids,
        "real_enforce_eager": bool(
            config["quantization"]["real"].get("enforce_eager", False)
        ),
        "errors": errors,
    }
    target = results_root(config, args.dataset) / "preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
