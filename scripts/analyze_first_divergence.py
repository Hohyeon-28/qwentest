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
from src.answer_parser import score_record
from src.first_divergence import (
    build_divergence_rows,
    compare_logit_pair,
    gap_relation,
    matched_continuation_budget,
    numeric_summary,
    select_flip_and_matched_controls,
    top_two,
)
from src.inference import resolved_stop_token_ids
from src.logging_utils import read_jsonl, seed_everything, token_ids_sha256
from src.prompts import split_reasoning_and_answer
from src.quant_utils import (
    fingerprint_gptq_checkpoint,
    load_shared_gptq_fake_model,
    load_shared_gptq_marlin_model,
    validate_shared_quant_config,
)
from scripts.run_vllm_marlin import validate_real_checkpoint


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


def _forced_prefix_for_candidate(
    candidate: dict[str, Any],
    fake_predictions: dict[str, dict[str, Any]],
    real_predictions: dict[str, dict[str, Any]],
) -> list[int]:
    sample_id = str(candidate["sample_id"])
    fake_record = fake_predictions[sample_id]
    real_record = real_predictions[sample_id]
    position = int(candidate["first_divergence_index"])
    fake_prefix = [
        int(token) for token in fake_record["generated_token_ids"][:position]
    ]
    real_prefix = [
        int(token) for token in real_record["generated_token_ids"][:position]
    ]
    if fake_prefix != real_prefix:
        raise ValueError(f"Free-generation common prefix mismatch for {sample_id}")
    prefix_ids = [int(token) for token in fake_record["input_token_ids"]] + fake_prefix
    if token_ids_sha256(prefix_ids) != candidate["forced_prefix_sha256"]:
        raise ValueError(f"Forced prefix hash mismatch for {sample_id}")
    return prefix_ids


def _build_vllm_engine(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any], str]:
    from vllm import LLM

    validate_real_checkpoint(config)
    manifest = fingerprint_gptq_checkpoint(
        config["models"]["real_gptq"],
        revision=config["models"]["revision"],
    )
    real_config = config["quantization"]["real"]
    llm = LLM(
        model=config["models"]["real_gptq"],
        tokenizer=config["models"]["base"],
        quantization="gptq_marlin",
        dtype="bfloat16",
        seed=int(config["generation"]["seed"]),
        revision=config["models"]["revision"],
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(real_config["max_model_len"]),
        enable_chunked_prefill=bool(
            real_config.get("enable_chunked_prefill", False)
        ),
        enforce_eager=bool(real_config.get("enforce_eager", True)),
    )
    try:
        selected_backend = str(llm.llm_engine.model_config.quantization)
    except Exception:
        selected_backend = "gptq_marlin (requested)"
    return llm, manifest, selected_backend


def capture_vllm_prefix_replay(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    from vllm import SamplingParams

    output = _analysis_root(config, args.dataset, args.analysis_tag)
    candidates = read_jsonl(output / "candidates.jsonl")
    if not candidates:
        raise RuntimeError("Run the select command first")
    rows_path = output / "vllm_deployment_replay.jsonl"
    metadata_path = output / "vllm_deployment_capture.json"
    if (rows_path.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError(
            "vLLM deployment replay already exists; use a new --analysis-tag "
            "or pass --overwrite"
        )
    fake_predictions, real_predictions = _prediction_maps(config, args.dataset)
    llm, manifest, selected_backend = _build_vllm_engine(args, config)
    expected_tuple = candidates[0].get("quantization_tuple_sha256")
    if expected_tuple and manifest["tuple_sha256"] != expected_tuple:
        raise ValueError(
            "vLLM checkpoint tuple differs from the selected candidates: "
            f"selected={expected_tuple}, vllm={manifest['tuple_sha256']}"
        )

    full_vocab_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=int(config["generation"]["seed"]),
    )
    replay_rows: list[dict[str, Any]] = []
    repeat_full_top1_agreements: list[bool] = []
    repeat_pair_preference_agreements: list[bool] = []
    for index, candidate in enumerate(candidates, start=1):
        sample_id = str(candidate["sample_id"])
        print(f"[vllm-prefix-replay] {index}/{len(candidates)} {sample_id}", flush=True)
        prefix_ids = _forced_prefix_for_candidate(
            candidate, fake_predictions, real_predictions
        )
        fake_token = int(candidate["fake_free_token_id"])
        real_token = int(candidate["real_free_token_id"])
        pair_sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            seed=int(config["generation"]["seed"]),
            allowed_token_ids=[fake_token, real_token],
        )
        full_top1_tokens: list[int] = []
        pair_preferred_tokens: list[int] = []
        for _ in range(args.repeats):
            full_result = llm.generate(
                [{"prompt_token_ids": prefix_ids}],
                full_vocab_sampling,
                use_tqdm=False,
            )[0]
            pair_result = llm.generate(
                [{"prompt_token_ids": prefix_ids}],
                pair_sampling,
                use_tqdm=False,
            )[0]
            full_ids = [int(token) for token in full_result.outputs[0].token_ids]
            pair_ids = [int(token) for token in pair_result.outputs[0].token_ids]
            if len(full_ids) != 1 or len(pair_ids) != 1:
                raise RuntimeError(
                    f"Expected one replay token for {sample_id}, got "
                    f"full={full_ids}, pair={pair_ids}"
                )
            if pair_ids[0] not in (fake_token, real_token):
                raise RuntimeError(
                    f"allowed_token_ids was not enforced for {sample_id}: "
                    f"generated={pair_ids[0]}"
                )
            full_top1_tokens.append(full_ids[0])
            pair_preferred_tokens.append(pair_ids[0])
        for token in full_top1_tokens[1:]:
            repeat_full_top1_agreements.append(token == full_top1_tokens[0])
        for token in pair_preferred_tokens[1:]:
            repeat_pair_preference_agreements.append(
                token == pair_preferred_tokens[0]
            )
        pair_relation = (
            "fake_token_preferred"
            if pair_preferred_tokens[0] == fake_token
            else "real_token_preferred"
        )
        replay_rows.append(
            {
                "sample_id": sample_id,
                "selection_role": candidate["selection_role"],
                "group": candidate["group"],
                "forced_prefix_sha256": candidate["forced_prefix_sha256"],
                "forced_prefix_token_count": len(prefix_ids),
                "first_divergence_index": int(candidate["first_divergence_index"]),
                "fake_free_token_id": fake_token,
                "real_free_token_id": real_token,
                "prefix_top1_token_id": full_top1_tokens[0],
                "vllm_pair_preferred_token_id": pair_preferred_tokens[0],
                "vllm_pair_preference_relation": pair_relation,
                "full_top1_is_one_of_old_candidates": full_top1_tokens[0]
                in (fake_token, real_token),
            }
        )

    _write_jsonl(rows_path, replay_rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "analysis_tag": args.analysis_tag,
        "condition": "vllm_deployment_marlin",
        "comparison_scope": "actual_vllm_deployment_prefix_replay",
        "samples": len(replay_rows),
        "repeats": args.repeats,
        "prompt_logprobs": None,
        "full_vocab_logits_saved": False,
        "full_vocab_top1_exact": True,
        "candidate_pair_preference_exact": True,
        "candidate_pair_gap_magnitude_available": False,
        "requested_quantization_backend": "gptq_marlin",
        "selected_quantization_backend": selected_backend,
        "quantization_tuple_sha256": manifest["tuple_sha256"],
        "repeat_full_top1_agreement": (
            sum(repeat_full_top1_agreements) / len(repeat_full_top1_agreements)
            if repeat_full_top1_agreements
            else 1.0
        ),
        "repeat_pair_preference_agreement": (
            sum(repeat_pair_preference_agreements)
            / len(repeat_pair_preference_agreements)
            if repeat_pair_preference_agreements
            else 1.0
        ),
        "interpretation_guardrail": (
            "The legacy CUDA 11.8/vLLM stack cannot safely compute prompt-logprob "
            "ranks for these prefixes. This replay records the exact full-vocabulary "
            "greedy top-1 and an exact two-candidate preference using vLLM's "
            "allowed_token_ids processor, without requesting any logprobs. It does "
            "not report the numerical magnitude of the candidate gap."
        ),
    }
    _write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def compare_vllm_prefix_replay(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    root = _analysis_root(config, args.dataset, args.analysis_tag)
    fake_meta, fake_logits = _load_capture(root, "fake_dense")
    replay_rows = read_jsonl(root / "vllm_deployment_replay.jsonl")
    if not replay_rows:
        raise RuntimeError("Run capture-vllm first")
    replay = {str(row["sample_id"]): row for row in replay_rows}
    comparisons: list[dict[str, Any]] = []
    for index, fake_row in enumerate(fake_meta["rows"]):
        sample_id = str(fake_row["sample_id"])
        real_row = replay.get(sample_id)
        if real_row is None:
            raise ValueError(f"Missing vLLM replay row for {sample_id}")
        if fake_row["forced_prefix_sha256"] != real_row["forced_prefix_sha256"]:
            raise ValueError(f"Prefix mismatch for {sample_id}")
        fake_top1, fake_top1_logit, fake_top2, fake_top2_logit = top_two(
            fake_logits[index, 0]
        )
        fake_token = int(fake_row["fake_free_token_id"])
        real_token = int(fake_row["real_free_token_id"])
        fake_gap = float(
            (fake_logits[index, 0, fake_token] - fake_logits[index, 0, real_token]).item()
        )
        vllm_top1 = int(real_row["prefix_top1_token_id"])
        vllm_pair_relation = str(
            real_row["vllm_pair_preference_relation"]
        )
        comparisons.append(
            {
                **fake_row,
                "fake_dense_top1_token_id": fake_top1,
                "fake_dense_top2_token_id": fake_top2,
                "fake_dense_top1_margin": fake_top1_logit - fake_top2_logit,
                "vllm_top1_token_id": vllm_top1,
                "fake_vllm_top1_flip": fake_top1 != vllm_top1,
                "old_free_candidate_fake_dense_gap": fake_gap,
                "fake_dense_gap_relation": gap_relation(fake_gap),
                "vllm_pair_preferred_token_id": int(
                    real_row["vllm_pair_preferred_token_id"]
                ),
                "vllm_pair_preference_relation": vllm_pair_relation,
                "candidate_relation_changed_including_fake_ties": (
                    gap_relation(fake_gap) != vllm_pair_relation
                ),
                "old_free_divergence_reproduced_exactly": (
                    fake_top1 == fake_token and vllm_top1 == real_token
                ),
                "full_top1_is_one_of_old_candidates": bool(
                    real_row["full_top1_is_one_of_old_candidates"]
                ),
            }
        )
    _write_jsonl(root / "vllm_deployment_comparison.jsonl", comparisons)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": len(rows),
            "fake_vllm_top1_flips": sum(
                bool(row["fake_vllm_top1_flip"]) for row in rows
            ),
            "old_free_divergences_reproduced_exactly": sum(
                bool(row["old_free_divergence_reproduced_exactly"])
                for row in rows
            ),
            "candidate_relations_changed_including_fake_ties": sum(
                bool(row["candidate_relation_changed_including_fake_ties"])
                for row in rows
            ),
            "full_top1_outside_old_candidate_pair": sum(
                not bool(row["full_top1_is_one_of_old_candidates"])
                for row in rows
            ),
            "fake_dense_top1_margin": numeric_summary(
                row["fake_dense_top1_margin"] for row in rows
            ),
        }

    groups: dict[str, list[dict[str, Any]]] = {"all": comparisons}
    for row in comparisons:
        groups.setdefault(str(row["selection_role"]), []).append(row)
        groups.setdefault(str(row["group"]), []).append(row)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_tag": args.analysis_tag,
        "comparison_scope": "fake_dense_transformers_vs_actual_vllm_marlin_at_fixed_prefix",
        "groups": {name: summarize(rows) for name, rows in groups.items()},
        "interpretation_guardrail": (
            "This tests whether the old split reappears at an identical forced prefix "
            "through the actual vLLM deployment path. It intentionally includes the "
            "vLLM runtime plus Marlin and therefore does not isolate Marlin alone."
        ),
    }
    _write_json(root / "vllm_deployment_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _branch_record(
    *,
    tokenizer: Any,
    source: dict[str, Any],
    candidate: dict[str, Any],
    branch_name: str,
    branch_token_id: int,
    common_generated_prefix: list[int],
    continuation_ids: list[int],
    finish_reason: str | None,
    remaining_budget: int,
    repeat: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    full_generated_ids = [
        *common_generated_prefix,
        int(branch_token_id),
        *continuation_ids,
    ]
    reasoning, final_text, reasoning_count, reasoning_complete = (
        split_reasoning_and_answer(tokenizer, full_generated_ids)
    )
    record = {
        "sample_id": str(candidate["sample_id"]),
        "dataset_index": source.get("dataset_index"),
        "question": source.get("question"),
        "ground_truth": source.get("ground_truth"),
        "branch": branch_name,
        "branch_repeat": repeat,
        "branch_token_id": int(branch_token_id),
        "branch_token_text": tokenizer.decode([int(branch_token_id)]),
        "forced_prefix_sha256": candidate["forced_prefix_sha256"],
        "first_divergence_index": int(candidate["first_divergence_index"]),
        "common_generated_prefix_count": len(common_generated_prefix),
        "continuation_budget": remaining_budget,
        "continuation_token_count": len(continuation_ids),
        "generated_token_count": len(full_generated_ids),
        "generated_token_ids": full_generated_ids,
        "generated_token_ids_sha256": token_ids_sha256(full_generated_ids),
        "generated_text": tokenizer.decode(full_generated_ids),
        "reasoning": reasoning,
        "final_text": final_text,
        "generated_reasoning_token_count": reasoning_count,
        "reasoning_complete": reasoning_complete,
        "finish_reason": finish_reason,
        "hit_max_new_tokens": finish_reason == "length",
        "generation_enable_thinking": bool(
            config["generation"]["enable_thinking"]
        ),
        "generation_max_new_tokens": int(
            config["generation"]["max_new_tokens"]
        ),
        "continuation_backend": "vllm_gptq_marlin_for_both_branches",
    }
    return score_record(record, "math500")


def run_vllm_forced_branches(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    if args.dataset != "math500":
        raise ValueError("The causal branch scorer is currently defined for math500")
    analysis_root = _analysis_root(config, args.dataset, args.analysis_tag)
    candidates = {
        str(row["sample_id"]): row
        for row in read_jsonl(analysis_root / "candidates.jsonl")
    }
    sample_ids = args.sample_id or ["math500-00135"]
    missing = [sample_id for sample_id in sample_ids if sample_id not in candidates]
    if missing:
        raise ValueError(f"Samples are not in candidates.jsonl: {missing}")
    branch_root = analysis_root / "counterfactual" / args.branch_tag
    rows_path = branch_root / "branch_outputs.jsonl"
    summary_path = branch_root / "branch_summary.json"
    if (rows_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Branch run already exists under {branch_root}; use a new "
            "--branch-tag or pass --overwrite"
        )
    fake_predictions, real_predictions = _prediction_maps(config, args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["base"],
        revision=config["models"]["revision"],
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm, manifest, selected_backend = _build_vllm_engine(args, config)
    generation = config["generation"]
    stop_ids = resolved_stop_token_ids(config, tokenizer)
    all_rows: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        candidate = candidates[sample_id]
        source = fake_predictions[sample_id]
        prefix_ids = _forced_prefix_for_candidate(
            candidate, fake_predictions, real_predictions
        )
        position = int(candidate["first_divergence_index"])
        common_generated_prefix = [
            int(token) for token in source["generated_token_ids"][:position]
        ]
        remaining = (
            int(args.max_continuation_tokens)
            if args.max_continuation_tokens is not None
            else matched_continuation_budget(
                int(generation["max_new_tokens"]), position
            )
        )
        max_model_len = int(config["quantization"]["real"]["max_model_len"])
        if len(prefix_ids) + 1 + remaining > max_model_len:
            raise ValueError(
                f"Counterfactual branch for {sample_id} exceeds max_model_len: "
                f"{len(prefix_ids) + 1 + remaining} > {max_model_len}"
            )
        sampling = SamplingParams(
            temperature=0.0,
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
            seed=int(generation["seed"]),
            max_tokens=remaining,
            stop_token_ids=stop_ids,
        )
        branches = (
            ("fake_token", int(candidate["fake_free_token_id"])),
            ("real_token", int(candidate["real_free_token_id"])),
        )
        sample_rows: list[dict[str, Any]] = []
        for repeat in range(1, args.repeats + 1):
            for branch_name, branch_token in branches:
                print(
                    f"[vllm-branch] {sample_id} {branch_name} "
                    f"repeat={repeat}/{args.repeats} budget={remaining}",
                    flush=True,
                )
                request = {"prompt_token_ids": [*prefix_ids, branch_token]}
                result = llm.generate([request], sampling, use_tqdm=False)[0]
                completion = result.outputs[0]
                continuation_ids = [int(token) for token in completion.token_ids]
                row = _branch_record(
                    tokenizer=tokenizer,
                    source=source,
                    candidate=candidate,
                    branch_name=branch_name,
                    branch_token_id=branch_token,
                    common_generated_prefix=common_generated_prefix,
                    continuation_ids=continuation_ids,
                    finish_reason=getattr(completion, "finish_reason", None),
                    remaining_budget=remaining,
                    repeat=repeat,
                    config=config,
                )
                sample_rows.append(row)
                all_rows.append(row)
        first_fake = next(
            row
            for row in sample_rows
            if row["branch"] == "fake_token" and row["branch_repeat"] == 1
        )
        first_real = next(
            row
            for row in sample_rows
            if row["branch"] == "real_token" and row["branch_repeat"] == 1
        )
        reproducible = {}
        for branch_name, _ in branches:
            hashes = {
                row["generated_token_ids_sha256"]
                for row in sample_rows
                if row["branch"] == branch_name
            }
            reproducible[branch_name] = len(hashes) == 1
        effects.append(
            {
                "sample_id": sample_id,
                "fake_branch_is_correct": bool(first_fake["is_correct"]),
                "real_branch_is_correct": bool(first_real["is_correct"]),
                "correctness_changed": (
                    bool(first_fake["is_correct"]) != bool(first_real["is_correct"])
                ),
                "fake_branch_hit_max_new_tokens": bool(
                    first_fake["hit_max_new_tokens"]
                ),
                "real_branch_hit_max_new_tokens": bool(
                    first_real["hit_max_new_tokens"]
                ),
                "truncation_changed": (
                    bool(first_fake["hit_max_new_tokens"])
                    != bool(first_real["hit_max_new_tokens"])
                ),
                "final_answer_changed": (
                    first_fake.get("final_answer") != first_real.get("final_answer")
                ),
                "generated_token_count_delta_real_minus_fake": (
                    int(first_real["generated_token_count"])
                    - int(first_fake["generated_token_count"])
                ),
                "branch_reproducible": reproducible,
            }
        )
    _write_jsonl(rows_path, all_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_tag": args.analysis_tag,
        "branch_tag": args.branch_tag,
        "samples": sample_ids,
        "repeats": args.repeats,
        "budget_mode": (
            "explicit_continuation_budget"
            if args.max_continuation_tokens is not None
            else "matched_original_total_generation_budget"
        ),
        "same_continuation_backend_for_both_branches": True,
        "requested_quantization_backend": "gptq_marlin",
        "selected_quantization_backend": selected_backend,
        "quantization_tuple_sha256": manifest["tuple_sha256"],
        "effects": effects,
        "causal_scope": (
            "The comparison estimates the conditional effect of forcing one old "
            "divergence token versus the other at a fixed prefix, with every later "
            "token generated by the same vLLM GPTQ-Marlin backend. It does not by "
            "itself identify which upstream kernel or runtime component created the "
            "original token preference."
        ),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _add_vllm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)


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

    capture_vllm = subparsers.add_parser("capture-vllm")
    capture_vllm.add_argument("--config", default="configs/experiment.yaml")
    capture_vllm.add_argument(
        "--dataset", choices=("gsm8k", "math500"), default="math500"
    )
    capture_vllm.add_argument("--analysis-tag", default="controlled_v1")
    capture_vllm.add_argument("--repeats", type=int, default=None)
    capture_vllm.add_argument("--overwrite", action="store_true")
    _add_vllm_args(capture_vllm)

    compare_vllm = subparsers.add_parser("compare-vllm")
    compare_vllm.add_argument("--config", default="configs/experiment.yaml")
    compare_vllm.add_argument(
        "--dataset", choices=("gsm8k", "math500"), default="math500"
    )
    compare_vllm.add_argument("--analysis-tag", default="controlled_v1")

    branch_vllm = subparsers.add_parser("branch-vllm")
    branch_vllm.add_argument("--config", default="configs/experiment.yaml")
    branch_vllm.add_argument(
        "--dataset", choices=("gsm8k", "math500"), default="math500"
    )
    branch_vllm.add_argument("--analysis-tag", default="controlled_v1")
    branch_vllm.add_argument("--branch-tag", required=True)
    branch_vllm.add_argument("--sample-id", action="append", default=None)
    branch_vllm.add_argument("--max-continuation-tokens", type=int, default=None)
    branch_vllm.add_argument("--repeats", type=int, default=1)
    branch_vllm.add_argument("--overwrite", action="store_true")
    _add_vllm_args(branch_vllm)

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
    elif args.command == "compare":
        compare_captures(args, config)
    elif args.command == "capture-vllm":
        if args.repeats is None:
            args.repeats = int(
                config["evaluation"].get("first_divergence_repeats", 2)
            )
        if args.repeats < 1:
            raise ValueError("--repeats must be positive")
        capture_vllm_prefix_replay(args, config)
    elif args.command == "compare-vllm":
        compare_vllm_prefix_replay(args, config)
    else:
        if args.repeats < 1:
            raise ValueError("--repeats must be positive")
        if (
            args.max_continuation_tokens is not None
            and args.max_continuation_tokens < 1
        ):
            raise ValueError("--max-continuation-tokens must be positive")
        run_vllm_forced_branches(args, config)


if __name__ == "__main__":
    main()
