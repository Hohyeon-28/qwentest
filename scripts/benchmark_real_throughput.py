from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_CHOICES, load_config, results_root
from src.data import load_samples
from src.logging_utils import batched, seed_everything, write_environment
from src.prompts import build_prompt
from scripts.run_vllm_marlin import validate_real_checkpoint


def _parse_batch_sizes(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure GPTQ-Marlin offline aggregate throughput separately from the "
            "batch-1 quality experiment"
        )
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="gsm8k")
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Opt-in only: may fail on the legacy CUDA 11.8 stack",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    validate_real_checkpoint(config)
    seed_everything(int(config["generation"]["seed"]))
    benchmark = config["throughput_benchmark"]
    batch_sizes = args.batch_sizes or [int(x) for x in benchmark["batch_sizes"]]
    sample_limit = args.max_samples or int(benchmark["max_samples"])
    max_new_tokens = args.max_new_tokens or int(benchmark["max_new_tokens"])
    max_num_seqs = max(max(batch_sizes), int(benchmark["max_num_seqs"]))
    samples = load_samples(config, args.dataset, max_samples=sample_limit)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["base"],
        revision=config["models"]["revision"],
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
    )
    prepared = [
        build_prompt(
            tokenizer,
            sample["question"],
            bool(config["generation"]["enable_thinking"]),
            sample.get("system_message"),
        )["input_token_ids"]
        for sample in samples
    ]
    real = config["quantization"]["real"]
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
        max_model_len=int(real["max_model_len"]),
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=bool(real.get("enable_chunked_prefill", False)),
        enforce_eager=not args.enable_cuda_graph,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop_token_ids=[int(x) for x in config["generation"]["stop_token_ids"]],
    )
    if prepared:
        llm.generate(
            [{"prompt_token_ids": prepared[0]}],
            SamplingParams(temperature=0.0, max_tokens=1),
            use_tqdm=False,
        )

    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        total_tokens = 0
        total_seconds = 0.0
        request_rates: list[float] = []
        for group in batched(prepared, batch_size):
            requests = [{"prompt_token_ids": token_ids} for token_ids in group]
            started = time.perf_counter()
            outputs = llm.generate(requests, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            total_tokens += tokens
            total_seconds += elapsed
            request_rates.extend(
                len(output.outputs[0].token_ids) / elapsed for output in outputs
            )
        rows.append(
            {
                "batch_size": batch_size,
                "requests": len(prepared),
                "generated_tokens": total_tokens,
                "generation_wall_seconds": total_seconds,
                "aggregate_tokens_per_second": (
                    total_tokens / total_seconds if total_seconds > 0 else None
                ),
                "mean_per_request_share_tokens_per_second": (
                    mean(request_rates) if request_rates else None
                ),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False))

    output_dir = results_root(config, args.dataset) / "throughput_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": config["models"]["real_gptq"],
        "backend": "gptq_marlin",
        "dataset": args.dataset,
        "max_new_tokens": max_new_tokens,
        "max_num_seqs": max_num_seqs,
        "enforce_eager": not args.enable_cuda_graph,
        "metric": "total generated tokens / measured offline batch wall time",
        "quality_results_comparable": False,
        "reason": (
            "This capped, concurrent run is a performance benchmark, not the "
            "38,912-token batch-1 quality protocol."
        ),
        "results": rows,
    }
    target = output_dir / "real_marlin.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_environment(output_dir / "environment.json")
    print(f"report: {target}")


if __name__ == "__main__":
    main()
