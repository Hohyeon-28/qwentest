from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vllm_marlin import validate_real_checkpoint
from src.config import add_common_args, load_config, results_root, save_effective_config
from src.data import load_samples
from src.inference import run_hf_generation, warmup_hf
from src.logging_utils import append_jsonl, seed_everything, write_environment
from src.quant_utils import (
    fingerprint_gptq_checkpoint,
    load_shared_gptq_fake_model,
    validate_shared_quant_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dequantize the exact GPTQ (q,s,z,g) tuple used by Real, then run dense GEMM"
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device receiving the dequantized dense model, e.g. cuda or cuda:0",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    validate_shared_quant_config(config)
    validate_real_checkpoint(config)
    seed_everything(int(config["generation"]["seed"]))
    samples = load_samples(config, args.dataset, max_samples=args.max_samples)
    output_dir = results_root(config, args.dataset) / "fake_quant"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, args.dataset)

    manifest = fingerprint_gptq_checkpoint(
        config["models"]["real_gptq"], revision=config["models"]["revision"]
    )
    (output_dir / "quantization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["base"],
        revision=config["models"]["revision"],
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
    )
    model, reports = load_shared_gptq_fake_model(config, device=args.device)

    report_path = output_dir / "dequantization_report.jsonl"
    if report_path.exists():
        report_path.unlink()
    for report in reports:
        append_jsonl(report_path, report)
    write_environment(
        results_root(config, args.dataset) / "environment.json",
        {
            "fake_quant_source": config["models"]["real_gptq"],
            "fake_quant_tuple_sha256": manifest["tuple_sha256"],
            "fake_quantized_linear_layers": len(reports),
            "fake_operation": "W_fake = s[g] * (q - z[g])",
            "fake_execution": (
                f"dequantized_{config['quantization']['fake']['compute_dtype']}_torch_linear"
            ),
        },
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    warmup_hf(model, tokenizer, samples, config)

    errors = output_dir / "exceptions.jsonl"
    summary = run_hf_generation(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        config=config,
        condition="fake_quant",
        output_path=output_dir / "predictions.jsonl",
        batch_size=args.batch_size or int(config["generation"]["batch_size"]),
        overwrite=args.overwrite,
        on_exception=lambda row: append_jsonl(errors, row),
        condition_metadata={
            "quantization_source_checkpoint": config["models"]["real_gptq"],
            "quantization_tuple_sha256": manifest["tuple_sha256"],
            "quantization_tuple": "(q,s,z,g)",
        },
    )
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
