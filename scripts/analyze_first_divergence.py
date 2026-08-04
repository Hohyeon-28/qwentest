from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, results_root
from src.first_divergence import (
    build_divergence_rows,
    compare_logit_pair,
    numeric_summary,
    select_flip_and_matched_controls,
)
from src.logging_utils import read_jsonl, seed_everything, token_ids_sha256
from src.quant_utils import (
    load_shared_gptq_fake_model,
    load_shared_gptq_marlin_model,
    validate_shared_quant_config,
)


SCHEMA_VERSION = 1


def _analysis_root(
    config: dict[str, Any], dataset: str, analysis_tag: str
) -> Path:
    if not analysis_tag or any(character in analysis_tag for character in ("/", "\\")):
        raise ValueError("--analysis-tag must be a non-empty directory name")
    return results_root(config, dataset) / "first_divergence" / analysis_tag


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_candidates(args: argparse.Namespace, config: dict[str, Any]) -> None:
    root = results_root(config, args.dataset)
    fake_path = root / "fake_quant" / "predictions.jsonl"
    real_path = root / "real_quant_marlin" / "predictions.jsonl"
    fake = read_jsonl(fake_path)
    real = read_jsonl(real_path)
    if not fake or not real:
        raise RuntimeError(f"Missing prediction files: {fake_path}, {real_path}")
    rows = build_divergence_rows(fake, real)
    selected = select_flip_and_matched_controls(rows, max_flips=args.max_flips)
    output = _analysis_root(config, args.dataset, args.analysis_tag)
    selection_paths = [
        output / "all_divergences.jsonl",
        output / "candidates.jsonl",
        output / "selection_summary.json",
    ]
    if any(path.exists() for path in selection_paths) and not args.overwrite:
        raise FileExistsError(
            f"Selection already exists under {output}; use a new --analysis-tag "
            "or pass --overwrite"
        )
    _write_jsonl(output / "all_divergences.jsonl", rows)
    _write_jsonl(output / "candidates.jsonl", selected)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_tag": args.analysis_tag,
        "samples_compared": len(rows),
        "divergence_groups": dict(Counter(str(row["group"]) for row in rows)),
        "selected_samples": len(selected),
        "selection_roles": dict(
            Counter(str(row["selection_role"]) for row in selected)
        ),
        "selected_groups": dict(Counter(str(row["group"]) for row in selected)),
        "outcome_label_source": "stored_predictions_jsonl_is_correct",
        "outcome_label_guardrail": (
            "Recorded correctness is used only to select sensitive cases. Any known "
            "answer-parser corrections must be applied before claiming a mathematical "
            "correctness transition."
        ),
        "important_limitation": (
            "Candidates come from the original cross-runtime free generations. "
            "The controlled capture tests whether their old divergence prefix "
            "reproduces when only the linear backend changes."
        ),
    }
    _write_json(output / "selection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _prediction_maps(
    config: dict[str, Any], dataset: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    root = results_root(config, dataset)
    fake = {
        str(row["sample_id"]): row
        for row in read_jsonl(root / "fake_quant" / "predictions.jsonl")
    }
    real = {
        str(row["sample_id"]): row
        for row in read_jsonl(root / "real_quant_marlin" / "predictions.jsonl")
    }
    return fake, real


def _load_controlled_model(
    condition: str, config: dict[str, Any], device: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if condition == "fake_dense":
        model, reports = load_shared_gptq_fake_model(config, device=device)
        return model, {
            "linear_backend": "dequantized_bfloat16_torch_linear",
            "converted_linear_layers": len(reports),
        }
    model, backend_name, marlin_layers = load_shared_gptq_marlin_model(
        config, device=device
    )
    return model, {
        "linear_backend": backend_name,
        "marlin_linear_layers": marlin_layers,
    }


@torch.inference_mode()
def capture_logits(args: argparse.Namespace, config: dict[str, Any]) -> None:
    validate_shared_quant_config(config)
    seed_everything(int(config["generation"]["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    output = _analysis_root(config, args.dataset, args.analysis_tag)
    candidates = read_jsonl(output / "candidates.jsonl")
    if not candidates:
        raise RuntimeError("Run the select command first")
    fake_predictions, real_predictions = _prediction_maps(config, args.dataset)
    model, backend_metadata = _load_controlled_model(
        args.condition, config, args.device
    )
    model.eval()
    device = model.get_input_embeddings().weight.device

    logits_path = output / f"{args.condition}_logits.pt"
    metadata_path = output / f"{args.condition}_capture.json"
    if (logits_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Capture already exists for {args.condition}; pass --overwrite to replace it"
        )

    tensors: list[torch.Tensor] = []
    capture_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        sample_id = str(candidate["sample_id"])
        print(
            f"[{args.condition}] {candidate_index}/{len(candidates)} {sample_id}",
            flush=True,
        )
        fake_record = fake_predictions[sample_id]
        real_record = real_predictions[sample_id]
        position = int(candidate["first_divergence_index"])
        fake_prefix = [int(token) for token in fake_record["generated_token_ids"][:position]]
        real_prefix = [int(token) for token in real_record["generated_token_ids"][:position]]
        if fake_prefix != real_prefix:
            raise ValueError(f"Free-generation common prefix mismatch for {sample_id}")
        prefix_ids = [int(token) for token in fake_record["input_token_ids"]] + fake_prefix
        prefix_hash = token_ids_sha256(prefix_ids)
        if prefix_hash != candidate["forced_prefix_sha256"]:
            raise ValueError(f"Forced prefix hash mismatch for {sample_id}")
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        repeats: list[torch.Tensor] = []
        for _ in range(args.repeats):
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            repeats.append(result.logits[0, -1].detach().float().cpu())
        tensors.append(torch.stack(repeats))
        capture_rows.append(
            {
                "sample_id": sample_id,
                "forced_prefix_sha256": prefix_hash,
                "forced_prefix_token_count": len(prefix_ids),
                "first_divergence_index": position,
                "selection_role": candidate["selection_role"],
                "group": candidate["group"],
                "fake_free_token_id": int(candidate["fake_free_token_id"]),
                "real_free_token_id": int(candidate["real_free_token_id"]),
            }
        )

    stacked = torch.stack(tensors)
    torch.save(stacked, logits_path)
    repeat_max_abs = 0.0
    repeat_top1_agreement = 1.0
    if args.repeats > 1:
        baseline = stacked[:, :1, :]
        repeat_max_abs = float((stacked[:, 1:, :] - baseline).abs().max().item())
        repeat_top1_agreement = float(
            (
                stacked[:, 1:, :].argmax(dim=-1)
                == baseline.argmax(dim=-1)
            )
            .float()
            .mean()
            .item()
        )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "analysis_tag": args.analysis_tag,
        "condition": args.condition,
        "comparison_scope": "same_transformers_graph_linear_backend_control",
        "use_cache": False,
        "device_argument": args.device,
        "model_class": type(model).__name__,
        "attention_implementation": getattr(
            getattr(model, "config", None), "_attn_implementation", None
        ),
        "repeats": args.repeats,
        "samples": len(capture_rows),
        "vocab_size": int(stacked.shape[-1]),
        "dtype_saved": str(stacked.dtype),
        "same_backend_repeat_max_abs_logit_delta": repeat_max_abs,
        "same_backend_repeat_top1_agreement": repeat_top1_agreement,
        "quantization_tuple_sha256": candidates[0].get(
            "quantization_tuple_sha256"
        ),
        "gptqmodel_marlin_fp32_accumulation_env": os.environ.get(
            "GPTQMODEL_MARLIN_USE_FP32", "default"
        ),
        **backend_metadata,
        "rows": capture_rows,
    }
    _write_json(metadata_path, metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key != "rows"}, indent=2))


def _load_capture(root: Path, condition: str) -> tuple[dict[str, Any], torch.Tensor]:
    metadata = json.loads(
        (root / f"{condition}_capture.json").read_text(encoding="utf-8")
    )
    try:
        logits = torch.load(
            root / f"{condition}_logits.pt",
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        logits = torch.load(root / f"{condition}_logits.pt", map_location="cpu")
    if logits.ndim != 3:
        raise ValueError(f"Expected [sample, repeat, vocab] logits, got {logits.shape}")
    return metadata, logits.float()


def compare_captures(args: argparse.Namespace, config: dict[str, Any]) -> None:
    root = _analysis_root(config, args.dataset, args.analysis_tag)
    fake_meta, fake_logits = _load_capture(root, "fake_dense")
    real_meta, real_logits = _load_capture(root, "controlled_marlin")
    fake_rows = fake_meta["rows"]
    real_rows = real_meta["rows"]
    if len(fake_rows) != len(real_rows) or fake_logits.shape != real_logits.shape:
        raise ValueError("Fake and controlled-Marlin capture shapes differ")
    if fake_meta.get("attention_implementation") != real_meta.get(
        "attention_implementation"
    ):
        raise ValueError(
            "Controlled runs selected different attention implementations: "
            f"fake={fake_meta.get('attention_implementation')}, "
            f"real={real_meta.get('attention_implementation')}"
        )

    eos_ids = [int(token) for token in config["generation"]["stop_token_ids"]]
    comparisons: list[dict[str, Any]] = []
    for index, (fake_row, real_row) in enumerate(zip(fake_rows, real_rows)):
        if fake_row["sample_id"] != real_row["sample_id"]:
            raise ValueError(f"Sample ordering mismatch at row {index}")
        if fake_row["forced_prefix_sha256"] != real_row["forced_prefix_sha256"]:
            raise ValueError(f"Prefix mismatch for {fake_row['sample_id']}")
        metrics = compare_logit_pair(
            fake_logits[index, 0],
            real_logits[index, 0],
            eos_token_ids=eos_ids,
        )
        fake_free_token = int(fake_row["fake_free_token_id"])
        real_free_token = int(fake_row["real_free_token_id"])
        old_fake_gap = float(
            (
                fake_logits[index, 0, fake_free_token]
                - fake_logits[index, 0, real_free_token]
            ).item()
        )
        old_real_gap = float(
            (
                real_logits[index, 0, fake_free_token]
                - real_logits[index, 0, real_free_token]
            ).item()
        )
        comparisons.append(
            {
                **fake_row,
                **metrics,
                "old_free_candidate_fake_gap": old_fake_gap,
                "old_free_candidate_real_gap": old_real_gap,
                "old_free_candidate_gap_crossed_zero": (
                    old_fake_gap > 0.0 and old_real_gap < 0.0
                ),
                "old_free_divergence_reproduced_exactly": (
                    metrics["fake_top1_token_id"] == fake_free_token
                    and metrics["real_top1_token_id"] == real_free_token
                ),
            }
        )

    _write_jsonl(root / "controlled_comparison.jsonl", comparisons)

    def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": len(rows),
            "controlled_top1_flips": sum(
                bool(row["controlled_top1_flip"]) for row in rows
            ),
            "controlled_top1_flip_rate": (
                sum(bool(row["controlled_top1_flip"]) for row in rows) / len(rows)
                if rows
                else None
            ),
            "old_free_divergences_reproduced_exactly": sum(
                bool(row["old_free_divergence_reproduced_exactly"]) for row in rows
            ),
            "old_candidate_gap_zero_crossings": sum(
                bool(row["old_free_candidate_gap_crossed_zero"]) for row in rows
            ),
            "fake_top1_margin": numeric_summary(
                row["fake_top1_margin"] for row in rows
            ),
            "real_top1_margin": numeric_summary(
                row["real_top1_margin"] for row in rows
            ),
            "logit_delta_max_abs": numeric_summary(
                row["logit_delta_max_abs"] for row in rows
            ),
            "logit_delta_rms": numeric_summary(
                row["logit_delta_rms"] for row in rows
            ),
        }

    groups: dict[str, list[dict[str, Any]]] = {"all": comparisons}
    for row in comparisons:
        groups.setdefault(str(row["selection_role"]), []).append(row)
        groups.setdefault(str(row["group"]), []).append(row)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_tag": args.analysis_tag,
        "comparison_scope": "same_transformers_graph_linear_backend_control",
        "fake_backend": fake_meta.get("linear_backend"),
        "real_backend": real_meta.get("linear_backend"),
        "fake_repeat_max_abs_logit_delta": fake_meta.get(
            "same_backend_repeat_max_abs_logit_delta"
        ),
        "real_repeat_max_abs_logit_delta": real_meta.get(
            "same_backend_repeat_max_abs_logit_delta"
        ),
        "same_backend_top1_reproducibility_pass": (
            fake_meta.get("same_backend_repeat_top1_agreement") == 1.0
            and real_meta.get("same_backend_repeat_top1_agreement") == 1.0
        ),
        "attention_implementation": fake_meta.get("attention_implementation"),
        "groups": {name: group_summary(rows) for name, rows in groups.items()},
        "interpretation_guardrail": (
            "This removes the original HF-vs-vLLM runtime confound, but uses "
            "GPTQModel's Marlin integration rather than vLLM's deployment wrapper. "
            "A positive result motivates exact vLLM operator replay; a negative result "
            "means the old divergence cannot be attributed to Marlin from current data."
        ),
    }
    _write_json(root / "controlled_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Fake/Real first-divergence prefixes under a controlled runtime"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--config", default="configs/experiment.yaml")
    select.add_argument("--dataset", choices=("gsm8k", "math500"), default="math500")
    select.add_argument("--analysis-tag", default="controlled_v1")
    select.add_argument("--max-flips", type=int, default=None)
    select.add_argument("--overwrite", action="store_true")

    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", default="configs/experiment.yaml")
    capture.add_argument("--dataset", choices=("gsm8k", "math500"), default="math500")
    capture.add_argument("--analysis-tag", default="controlled_v1")
    capture.add_argument(
        "--condition",
        choices=("fake_dense", "controlled_marlin"),
        required=True,
    )
    capture.add_argument("--device", default="cuda")
    capture.add_argument("--repeats", type=int, default=None)
    capture.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--config", default="configs/experiment.yaml")
    compare.add_argument("--dataset", choices=("gsm8k", "math500"), default="math500")
    compare.add_argument("--analysis-tag", default="controlled_v1")

    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "select":
        select_candidates(args, config)
    elif args.command == "capture":
        if args.repeats is None:
            args.repeats = int(
                config["evaluation"].get("first_divergence_repeats", 2)
            )
        if args.repeats < 1:
            raise ValueError("--repeats must be positive")
        capture_logits(args, config)
    else:
        compare_captures(args, config)


if __name__ == "__main__":
    main()
