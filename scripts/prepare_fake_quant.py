from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_vllm_marlin import validate_real_checkpoint
from src.config import load_config
from src.quant_utils import (
    fingerprint_gptq_checkpoint,
    fingerprint_quant_tensor_dict,
    load_shared_gptq_fake_model,
    validate_shared_quant_config,
)


def self_test() -> dict:
    generator = torch.Generator().manual_seed(7)
    tensors = {
        "layer.qweight": torch.randint(
            -(2**31), 2**31 - 1, (16, 32), dtype=torch.int32, generator=generator
        ),
        "layer.scales": torch.rand(1, 32, generator=generator, dtype=torch.float16),
        "layer.qzeros": torch.randint(
            -(2**31), 2**31 - 1, (1, 4), dtype=torch.int32, generator=generator
        ),
        "layer.g_idx": torch.zeros(128, dtype=torch.int32),
    }
    fake = fingerprint_quant_tensor_dict(tensors, checkpoint="synthetic")
    real = fingerprint_quant_tensor_dict(
        {name: tensor.clone() for name, tensor in tensors.items()},
        checkpoint="synthetic",
    )
    assert fake["tuple_sha256"] == real["tuple_sha256"]
    changed = {name: tensor.clone() for name, tensor in tensors.items()}
    changed["layer.qweight"][0, 0] ^= 1
    changed_manifest = fingerprint_quant_tensor_dict(changed, checkpoint="synthetic")
    assert fake["tuple_sha256"] != changed_manifest["tuple_sha256"]
    return {
        "identical_tuple_hashes_match": True,
        "one_bit_change_is_detected": True,
        "tuple_sha256": fake["tuple_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.self_test and not args.verify_checkpoint and args.save_dir is None:
        parser.error("Choose --self-test, --verify-checkpoint, or --save-dir")

    config = load_config(args.config)
    validate_shared_quant_config(config)
    if args.self_test:
        print(json.dumps(self_test(), indent=2))

    manifest = None
    if args.verify_checkpoint or args.save_dir is not None:
        validate_real_checkpoint(config)
        manifest = fingerprint_gptq_checkpoint(
            config["models"]["real_gptq"], revision=config["models"]["revision"]
        )
        print(
            json.dumps(
                {
                    "checkpoint": manifest["checkpoint"],
                    "quantized_layers": manifest["quantized_layers"],
                    "tuple_sha256": manifest["tuple_sha256"],
                    "components": manifest["component_sha256"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.save_dir is not None:
        from transformers import AutoTokenizer

        model, reports = load_shared_gptq_fake_model(config, device="cpu")
        tokenizer = AutoTokenizer.from_pretrained(config["models"]["base"])
        args.save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(
            args.save_dir, safe_serialization=True, max_shard_size="4GB"
        )
        tokenizer.save_pretrained(args.save_dir)
        (args.save_dir / "shared_quantization_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.save_dir / "dequantization_report.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
