from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.answer_parser import score_record
from src.config import add_common_args, load_config, results_root, save_effective_config
from src.data import load_samples
from src.logging_utils import (
    append_jsonl,
    batched,
    completed_ids,
    read_jsonl,
    seed_everything,
    write_environment,
)
from src.metrics import summarize_predictions
from src.inference import (
    generation_protocol,
    resolved_stop_token_ids,
    validate_resume_protocol,
)
from src.prompts import build_prompt, split_reasoning_and_answer
from src.quant_utils import fingerprint_gptq_checkpoint, validate_shared_quant_config


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _quant_config(model_id: str, trust_remote_code: bool, revision: str) -> dict[str, Any]:
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, revision=revision
    )
    quant = getattr(model_config, "quantization_config", None)
    if quant is None:
        quant = model_config.to_dict().get("quantization_config")
    if hasattr(quant, "to_dict"):
        quant = quant.to_dict()
    if isinstance(quant, dict):
        return dict(quant)
    if quant is not None and hasattr(quant, "__dict__"):
        return {
            key: value
            for key, value in vars(quant).items()
            if not key.startswith("_")
        }
    return {}


def validate_real_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    validate_shared_quant_config(config)
    model_id = config["models"]["real_gptq"]
    if not model_id or model_id.startswith("REPLACE_WITH"):
        raise ValueError(
            "Set models.real_gptq in configs/experiment.yaml to a real GPTQ checkpoint"
        )
    quant = _quant_config(
        model_id,
        bool(config["models"]["trust_remote_code"]),
        config["models"]["revision"],
    )
    expected = config["quantization"]["real"]
    checks = {
        "bits": (quant.get("bits"), int(expected["bits"])),
        "group_size": (quant.get("group_size"), int(expected["group_size"])),
        "sym": (quant.get("sym", quant.get("symmetric")), bool(expected["symmetric"])),
        "desc_act": (quant.get("desc_act"), bool(expected["desc_act"])),
    }
    mismatches = {
        key: {"checkpoint": actual, "expected": target}
        for key, (actual, target) in checks.items()
        if actual != target
    }
    if mismatches:
        raise ValueError(f"GPTQ checkpoint quantization mismatch: {mismatches}")
    checkpoint_format = str(
        quant.get("format", quant.get("checkpoint_format", "gptq"))
    ).lower()
    if "marlin" in checkpoint_format:
        raise ValueError(
            "A pre-Marlin checkpoint cannot expose a backend-independent shared "
            "(q,s,z,g) boundary. Use a standard GPTQ or GPTQ-v2 checkpoint and let "
            "vLLM repack it to Marlin at runtime."
        )
    return quant


def _metric(metrics: Any, name: str) -> float | None:
    value = getattr(metrics, name, None) if metrics is not None else None
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPTQ INT4 with vLLM Marlin")
    add_common_args(parser)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["generation"]["seed"]))
    quant_config = validate_real_checkpoint(config)
    real_config = config["quantization"]["real"]
    manifest = fingerprint_gptq_checkpoint(
        config["models"]["real_gptq"], revision=config["models"]["revision"]
    )
    samples = load_samples(config, args.dataset, max_samples=args.max_samples)
    output_dir = results_root(config, args.dataset) / "real_quant_marlin"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, args.dataset, condition="real_quant_marlin")
    predictions = output_dir / "predictions.jsonl"
    if args.overwrite and predictions.exists():
        predictions.unlink()
    (output_dir / "quantization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["base"],
        revision=config["models"]["revision"],
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generation = config["generation"]
    stop_token_ids = resolved_stop_token_ids(config, tokenizer)
    protocol = generation_protocol(config, tokenizer)
    validate_resume_protocol(read_jsonl(predictions), protocol, predictions)

    startup_log = output_dir / "startup.log"
    startup_log_mode = "a" if predictions.exists() and not args.overwrite else "w"
    with startup_log.open(startup_log_mode, encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_handle)):
            with contextlib.redirect_stderr(Tee(sys.stderr, log_handle)):
                llm = LLM(
                    model=config["models"]["real_gptq"],
                    tokenizer=config["models"]["base"],
                    quantization="gptq_marlin",
                    dtype="bfloat16",
                    seed=int(config["generation"]["seed"]),
                    revision=config["models"]["revision"],
                    trust_remote_code=bool(config["models"]["trust_remote_code"]),
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=int(real_config["max_model_len"]),
                    enable_chunked_prefill=bool(
                        real_config.get("enable_chunked_prefill", False)
                    ),
                    enforce_eager=bool(real_config.get("enforce_eager", True)),
                )

    selected_backend = None
    try:
        selected_backend = llm.llm_engine.model_config.quantization
    except Exception:
        selected_backend = "gptq_marlin (requested; inspect startup.log for resolved kernel)"
    write_environment(
        output_dir / "environment.json",
        {
            "real_model": config["models"]["real_gptq"],
            "real_quantization_config": quant_config,
            "real_quantization_requested": "gptq_marlin",
            "real_quantization_selected": selected_backend,
            "real_quant_tuple_sha256": manifest["tuple_sha256"],
            "enforce_eager": bool(real_config.get("enforce_eager", True)),
            "startup_log": str(startup_log),
        },
    )

    sampling = SamplingParams(
        temperature=0.0 if generation["deterministic"] else float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
        seed=int(generation["seed"]),
        max_tokens=int(generation["max_new_tokens"]),
        stop_token_ids=stop_token_ids,
    )
    done = completed_ids(predictions)
    prepared: list[dict[str, Any]] = []
    for sample in samples:
        if str(sample["sample_id"]) in done:
            continue
        started = time.perf_counter()
        prompt = build_prompt(
            tokenizer, sample["question"], bool(generation["enable_thinking"])
        )
        tokenization_latency = time.perf_counter() - started
        prepared.append(
            {**sample, **prompt, "prompt_tokenization_latency_seconds": tokenization_latency}
        )

    warmups = min(int(config["evaluation"].get("warmup_samples", 0)), len(prepared))
    if warmups:
        warmup_params = SamplingParams(temperature=0.0, max_tokens=1)
        llm.generate(
            [{"prompt_token_ids": row["input_token_ids"]} for row in prepared[:warmups]],
            warmup_params,
            use_tqdm=False,
        )

    batch_size = args.batch_size or int(generation["batch_size"])
    exception_path = output_dir / "exceptions.jsonl"
    for group in batched(prepared, batch_size):
        try:
            requests = [{"prompt_token_ids": row["input_token_ids"]} for row in group]
            started = time.perf_counter()
            outputs = llm.generate(requests, sampling, use_tqdm=False)
            batch_elapsed = time.perf_counter() - started
            for row, request_output in zip(group, outputs):
                completion = request_output.outputs[0]
                output_ids = [int(item) for item in completion.token_ids]
                (
                    reasoning,
                    final_text,
                    reasoning_count,
                    reasoning_complete,
                ) = split_reasoning_and_answer(tokenizer, output_ids)
                finish_reason = getattr(completion, "finish_reason", None)
                hit_max_new_tokens = finish_reason == "length"
                metrics = getattr(request_output, "metrics", None)
                arrival = _metric(metrics, "arrival_time")
                first = _metric(metrics, "first_token_time")
                finished = _metric(metrics, "finished_time")
                total_latency = (
                    finished - arrival
                    if finished is not None and arrival is not None
                    else batch_elapsed / len(group)
                )
                record = {
                    **row,
                    "condition": "real_quant_marlin",
                    "quantization_source_checkpoint": config["models"]["real_gptq"],
                    "quantization_tuple_sha256": manifest["tuple_sha256"],
                    "quantization_tuple": "(q,s,z,g)",
                    "generated_text": completion.text.strip(),
                    "reasoning_text": reasoning,
                    "final_text": final_text,
                    "generated_token_ids": output_ids,
                    "generated_token_count": len(output_ids),
                    # L_i,m^gen: model m's actually generated reasoning-token count.
                    "generated_reasoning_token_count": reasoning_count,
                    "reasoning_token_count": reasoning_count,
                    "reasoning_complete": reasoning_complete,
                    "reasoning_incomplete": (
                        bool(generation["enable_thinking"])
                        and not reasoning_complete
                    ),
                    "reasoning_length_censored": (
                        hit_max_new_tokens and not reasoning_complete
                    ),
                    "hit_max_new_tokens": hit_max_new_tokens,
                    "finish_reason": finish_reason,
                    **protocol,
                    "total_sequence_length": row["input_token_count"] + len(output_ids),
                    "batch_size_used": len(group),
                    "prefill_latency_seconds": (
                        first - arrival
                        if first is not None and arrival is not None
                        else None
                    ),
                    "time_to_first_token_seconds": (
                        first - arrival
                        if first is not None and arrival is not None
                        else None
                    ),
                    "decode_latency_seconds": (
                        finished - first
                        if finished is not None and first is not None
                        else None
                    ),
                    "total_generation_latency_seconds": total_latency,
                    "batch_generation_latency_seconds": batch_elapsed,
                    "generated_tokens_per_second": (
                        len(output_ids) / total_latency if total_latency > 0 else None
                    ),
                    "requested_quantization_backend": "gptq_marlin",
                    "selected_quantization_backend": selected_backend,
                }
                append_jsonl(predictions, score_record(record, args.dataset))
        except Exception as exc:
            append_jsonl(
                exception_path,
                {
                    "sample_ids": [row["sample_id"] for row in group],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    summary = summarize_predictions(read_jsonl(predictions))
    summary.update(
        {
            "quantization_source_checkpoint": config["models"]["real_gptq"],
            "quantization_tuple_sha256": manifest["tuple_sha256"],
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
