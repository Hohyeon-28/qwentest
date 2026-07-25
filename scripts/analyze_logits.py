from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, results_root
from src.logging_utils import append_jsonl, read_jsonl, seed_everything
from src.quant_utils import load_shared_gptq_fake_model, validate_shared_quant_config
from scripts.run_vllm_marlin import validate_real_checkpoint


def reference_records(config: dict[str, Any], dataset: str, limit: int) -> list[dict[str, Any]]:
    path = results_root(config, dataset) / "bf16" / "predictions.jsonl"
    records = read_jsonl(path)
    if not records:
        raise RuntimeError(f"Run BF16 generation first: {path}")
    return records[:limit]


def topk_record(logits: torch.Tensor, reference_token: int, position: int, k: int) -> dict:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(log_probs, k=min(k, log_probs.numel()))
    return {
        "position": position,
        "reference_token_id": int(reference_token),
        "reference_token_logprob": float(log_probs[reference_token].item()),
        "top_token_ids": [int(item) for item in indices.tolist()],
        "top_logprobs": [float(item) for item in values.tolist()],
    }


@torch.inference_mode()
def capture_hf(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from transformers import AutoModelForCausalLM

    if args.condition == "fake_quant":
        validate_shared_quant_config(config)
        model, _ = load_shared_gptq_fake_model(config, device=args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config["models"]["base"],
            revision=config["models"]["revision"],
            trust_remote_code=bool(config["models"]["trust_remote_code"]),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()
    device = model.get_input_embeddings().weight.device
    records = reference_records(
        config, args.dataset, int(config["evaluation"]["logit_subset_size"])
    )
    output = results_root(config, args.dataset) / "logit_analysis" / f"{args.condition}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    k = int(config["evaluation"]["logit_top_k"])
    for record in records:
        prompt_ids = [int(item) for item in record["input_token_ids"]]
        reference_ids = [int(item) for item in record["generated_token_ids"]]
        if args.max_steps is not None:
            reference_ids = reference_ids[: args.max_steps]
        prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        result = model(input_ids=prompt, use_cache=True)
        cache = result.past_key_values
        steps = []
        logits = result.logits[0, -1]
        for position, token_id in enumerate(reference_ids):
            steps.append(topk_record(logits, token_id, position, k))
            if position + 1 < len(reference_ids):
                next_input = torch.tensor([[token_id]], device=device, dtype=torch.long)
                result = model(input_ids=next_input, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                logits = result.logits[0, -1]
        append_jsonl(
            output,
            {
                "sample_id": record["sample_id"],
                "condition": args.condition,
                "reference": "bf16_free_generation",
                "input_token_ids_sha256": record["input_token_ids_sha256"],
                "steps": steps,
                "full_vocab_logit_vectors_saved": False,
            },
        )


def _vllm_step(item: Any, position: int) -> dict[str, Any] | None:
    if item is None:
        return None
    pairs = []
    for token_id, value in item.items():
        logprob = getattr(value, "logprob", value)
        pairs.append((int(token_id), float(logprob)))
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    if not pairs:
        return None
    return {
        "position": position,
        "top_token_ids": [pair[0] for pair in pairs],
        "top_logprobs": [pair[1] for pair in pairs],
    }


def capture_vllm_real(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from vllm import LLM, SamplingParams

    validate_real_checkpoint(config)
    records = reference_records(
        config, args.dataset, int(config["evaluation"]["logit_subset_size"])
    )
    k = int(config["evaluation"]["logit_top_k"])
    llm = LLM(
        model=config["models"]["real_gptq"],
        tokenizer=config["models"]["base"],
        quantization="gptq_marlin",
        dtype="bfloat16",
        seed=int(config["generation"]["seed"]),
        trust_remote_code=bool(config["models"]["trust_remote_code"]),
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=k)
    output = results_root(config, args.dataset) / "logit_analysis" / "real_quant_marlin.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    for record in records:
        prompt_ids = [int(item) for item in record["input_token_ids"]]
        reference_ids = [int(item) for item in record["generated_token_ids"]]
        if args.max_steps is not None:
            reference_ids = reference_ids[: args.max_steps]
        combined = prompt_ids + reference_ids
        result = llm.generate(
            [{"prompt_token_ids": combined}], sampling, use_tqdm=False
        )[0]
        prompt_logprobs = result.prompt_logprobs or []
        raw_steps = prompt_logprobs[len(prompt_ids) : len(prompt_ids) + len(reference_ids)]
        steps = []
        for position, (reference_id, raw) in enumerate(zip(reference_ids, raw_steps)):
            converted = _vllm_step(raw, position)
            if converted is None:
                continue
            converted["reference_token_id"] = reference_id
            reference_entry = raw.get(reference_id)
            converted["reference_token_logprob"] = (
                float(getattr(reference_entry, "logprob", reference_entry))
                if reference_entry is not None
                else None
            )
            steps.append(converted)
        append_jsonl(
            output,
            {
                "sample_id": record["sample_id"],
                "condition": "real_quant_marlin",
                "reference": "bf16_free_generation",
                "input_token_ids_sha256": record["input_token_ids_sha256"],
                "steps": steps,
                "full_vocab_logit_vectors_saved": False,
            },
        )


def _restricted_distribution(step: dict[str, Any], union: set[int], floor: float) -> dict[int, float]:
    values = dict(zip(step["top_token_ids"], step["top_logprobs"]))
    logits = {token: values.get(token, floor) for token in union}
    maximum = max(logits.values())
    unnormalized = {token: math.exp(value - maximum) for token, value in logits.items()}
    total = sum(unnormalized.values())
    return {token: value / total for token, value in unnormalized.items()}


def compare_steps(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids = set(left["top_token_ids"])
    right_ids = set(right["top_token_ids"])
    union = left_ids | right_ids
    floor = min([*left["top_logprobs"], *right["top_logprobs"], -30.0]) - 5.0
    p = _restricted_distribution(left, union, floor)
    q = _restricted_distribution(right, union, floor)
    kl = sum(p[token] * math.log(max(p[token], 1e-30) / max(q[token], 1e-30)) for token in union)
    left_ref = left.get("reference_token_logprob")
    right_ref = right.get("reference_token_logprob")
    return {
        "top1_agreement": left["top_token_ids"][0] == right["top_token_ids"][0],
        "topk_overlap": len(left_ids & right_ids) / max(len(left_ids | right_ids), 1),
        "truncated_union_kl_left_to_right": kl,
        "reference_token_logprob_delta": (
            left_ref - right_ref
            if left_ref is not None and right_ref is not None
            else None
        ),
    }


def compare_pair(left_records: list[dict], right_records: list[dict]) -> dict[str, Any]:
    left_map = {str(row["sample_id"]): row for row in left_records}
    right_map = {str(row["sample_id"]): row for row in right_records}
    metrics = []
    for sample_id in sorted(set(left_map) & set(right_map)):
        left = left_map[sample_id]
        right = right_map[sample_id]
        if left["input_token_ids_sha256"] != right["input_token_ids_sha256"]:
            raise ValueError(f"Prompt mismatch for {sample_id}")
        left_steps = {int(step["position"]): step for step in left["steps"]}
        right_steps = {int(step["position"]): step for step in right["steps"]}
        for position in sorted(set(left_steps) & set(right_steps)):
            metrics.append(compare_steps(left_steps[position], right_steps[position]))
    return {
        "steps_compared": len(metrics),
        "top1_agreement": mean(item["top1_agreement"] for item in metrics) if metrics else None,
        "mean_topk_jaccard_overlap": (
            mean(item["topk_overlap"] for item in metrics) if metrics else None
        ),
        "mean_truncated_union_kl_left_to_right": (
            mean(item["truncated_union_kl_left_to_right"] for item in metrics)
            if metrics
            else None
        ),
        "mean_reference_token_logprob_delta": (
            mean(
                item["reference_token_logprob_delta"]
                for item in metrics
                if item["reference_token_logprob_delta"] is not None
            )
            if any(item["reference_token_logprob_delta"] is not None for item in metrics)
            else None
        ),
        "full_vocab_logit_mse": None,
        "full_vocab_logit_cosine_similarity": None,
        "limitation": (
            "vLLM exposes top-k prompt logprobs, not full Marlin logits/hidden states. "
            "KL is normalized over the union of captured top-k tokens and is approximate."
        ),
    }


def compare_captures(args: argparse.Namespace, config: dict[str, Any]) -> None:
    root = results_root(config, args.dataset) / "logit_analysis"
    captures = {
        name: read_jsonl(root / f"{name}.jsonl")
        for name in ("bf16", "fake_quant", "real_quant_marlin")
    }
    for name, records in captures.items():
        if not records:
            raise RuntimeError(f"Missing capture for {name}")
    result = {
        "bf16_vs_fake": compare_pair(captures["bf16"], captures["fake_quant"]),
        "bf16_vs_real": compare_pair(captures["bf16"], captures["real_quant_marlin"]),
        "fake_vs_real": compare_pair(captures["fake_quant"], captures["real_quant_marlin"]),
    }
    (root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("capture-hf", "capture-vllm-real", "compare"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default="configs/experiment.yaml")
        sub.add_argument("--dataset", choices=("gsm8k", "math500"), default="gsm8k")
        sub.add_argument("--max-steps", type=int, default=None)
        if command == "capture-hf":
            sub.add_argument("--condition", choices=("bf16", "fake_quant"), required=True)
            sub.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["generation"]["seed"]))
    if args.command == "capture-hf":
        capture_hf(args, config)
    elif args.command == "capture-vllm-real":
        capture_vllm_real(args, config)
    else:
        compare_captures(args, config)


if __name__ == "__main__":
    main()
