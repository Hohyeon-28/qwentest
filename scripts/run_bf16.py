from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_common_args, load_config, results_root, save_effective_config
from src.data import load_samples
from src.inference import run_hf_generation, warmup_hf
from src.logging_utils import append_jsonl, seed_everything, write_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-8B BF16 baseline with HF")
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["generation"]["seed"]))
    samples = load_samples(config, args.dataset, max_samples=args.max_samples)
    output_dir = results_root(config, args.dataset) / "bf16"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, args.dataset, condition="bf16")
    write_environment(
        output_dir / "environment.json",
        {"bf16_model": config["models"]["base"], "bf16_backend": "transformers"},
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["base"],
        revision=config["models"]["revision"],
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["base"],
        revision=config["models"]["revision"],
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    warmup_hf(model, tokenizer, samples, config)

    errors = output_dir / "exceptions.jsonl"
    summary = run_hf_generation(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        config=config,
        condition="bf16",
        output_path=output_dir / "predictions.jsonl",
        batch_size=args.batch_size or int(config["generation"]["batch_size"]),
        overwrite=args.overwrite,
        on_exception=lambda row: append_jsonl(errors, row),
    )
    print(summary)


if __name__ == "__main__":
    main()
