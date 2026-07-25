from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from .answer_parser import score_record
from .logging_utils import append_jsonl, batched, completed_ids, read_jsonl
from .metrics import summarize_predictions
from .prompts import build_prompt, split_reasoning_and_answer


def generation_kwargs(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    generation = config["generation"]
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "do_sample": not bool(generation["deterministic"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    additional_stops = [int(item) for item in generation.get("stop_token_ids", [])]
    if additional_stops:
        base_eos = tokenizer.eos_token_id
        if base_eos is None:
            base_eos = []
        elif not isinstance(base_eos, list):
            base_eos = [base_eos]
        kwargs["eos_token_id"] = list(dict.fromkeys([*base_eos, *additional_stops]))
    if not generation["deterministic"]:
        kwargs.update(
            {
                "temperature": float(generation["temperature"]),
                "top_p": float(generation["top_p"]),
                "top_k": int(generation["top_k"]),
            }
        )
    return kwargs


def _input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _trim_after_first_stop(token_ids: list[int], stop_ids: set[int]) -> list[int]:
    for index, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            return token_ids[: index + 1]
    return token_ids


@torch.inference_mode()
def warmup_hf(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    count = min(int(config["evaluation"].get("warmup_samples", 0)), len(samples))
    if count <= 0:
        return
    device = _input_device(model)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    for sample in samples[:count]:
        prompt = build_prompt(
            tokenizer,
            sample["question"],
            bool(config["generation"]["enable_thinking"]),
        )
        encoded = torch.tensor([prompt["input_token_ids"]], dtype=torch.long, device=device)
        model.generate(
            input_ids=encoded,
            attention_mask=torch.ones_like(encoded),
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def run_hf_generation(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    condition: str,
    output_path: str | Path,
    batch_size: int,
    overwrite: bool = False,
    on_exception: Callable[[dict[str, Any]], None] | None = None,
    condition_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    if overwrite and output.exists():
        output.unlink()
    done = completed_ids(output)
    pending = [sample for sample in samples if str(sample["sample_id"]) not in done]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = generation_kwargs(config, tokenizer)
    device = _input_device(model)
    configured_stops = {int(item) for item in config["generation"].get("stop_token_ids", [])}
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        configured_stops.update(int(item) for item in eos)
    elif eos is not None:
        configured_stops.add(int(eos))

    prepared = []
    for sample in pending:
        tokenization_started = time.perf_counter()
        prompt = build_prompt(
            tokenizer,
            sample["question"],
            bool(config["generation"]["enable_thinking"]),
        )
        prepared.append(
            {
                **sample,
                **prompt,
                "prompt_tokenization_latency_seconds": (
                    time.perf_counter() - tokenization_started
                ),
            }
        )

    for group in batched(prepared, batch_size):
        try:
            encoded = tokenizer.pad(
                [{"input_ids": row["input_token_ids"]} for row in group],
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            generated = model.generate(**encoded, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            prompt_width = encoded["input_ids"].shape[1]

            amortized_latency = elapsed / len(group)
            for row, output_ids in zip(group, generated):
                new_ids = [int(item) for item in output_ids[prompt_width:].tolist()]
                new_ids = _trim_after_first_stop(new_ids, configured_stops)
                reasoning, final_text, reasoning_count = split_reasoning_and_answer(
                    tokenizer, new_ids
                )
                generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                record = {
                    **row,
                    **(condition_metadata or {}),
                    "condition": condition,
                    "generated_text": generated_text,
                    "reasoning_text": reasoning,
                    "final_text": final_text,
                    "generated_token_ids": new_ids,
                    "generated_token_count": len(new_ids),
                    # L_i,m^gen: model m's actually generated reasoning-token count.
                    "generated_reasoning_token_count": reasoning_count,
                    "reasoning_token_count": reasoning_count,
                    "total_sequence_length": row["input_token_count"] + len(new_ids),
                    "batch_size_used": len(group),
                    "prefill_latency_seconds": None,
                    "time_to_first_token_seconds": None,
                    "decode_latency_seconds": None,
                    "batch_generation_latency_seconds": elapsed,
                    "total_generation_latency_seconds": amortized_latency,
                    "generated_tokens_per_second": (
                        len(new_ids) / amortized_latency if amortized_latency > 0 else None
                    ),
                }
                append_jsonl(output, score_record(record, row["dataset"]))
        except Exception as exc:
            failure = {
                "condition": condition,
                "sample_ids": [row["sample_id"] for row in group],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if on_exception:
                on_exception(failure)
            raise

    records = read_jsonl(output)
    summary = summarize_predictions(records)
    summary_path = output.parent / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
