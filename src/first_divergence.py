from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable

import torch

from src.logging_utils import token_ids_sha256


def _by_sample_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        indexed[sample_id] = record
    return indexed


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for position, (left_id, right_id) in enumerate(zip(left, right)):
        if int(left_id) != int(right_id):
            return position
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def build_divergence_rows(
    fake_records: Iterable[dict[str, Any]],
    real_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    fake = _by_sample_id(fake_records)
    real = _by_sample_id(real_records)
    if set(fake) != set(real):
        missing_fake = sorted(set(real) - set(fake))
        missing_real = sorted(set(fake) - set(real))
        raise ValueError(
            "Fake/Real sample sets differ: "
            f"missing_fake={missing_fake[:5]}, missing_real={missing_real[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(fake):
        fake_record = fake[sample_id]
        real_record = real[sample_id]
        fake_prompt_hash = fake_record.get("input_token_ids_sha256")
        real_prompt_hash = real_record.get("input_token_ids_sha256")
        if fake_prompt_hash != real_prompt_hash:
            raise ValueError(f"Prompt hash mismatch for {sample_id}")
        fake_tuple = fake_record.get("quantization_tuple_sha256")
        real_tuple = real_record.get("quantization_tuple_sha256")
        if fake_tuple and real_tuple and fake_tuple != real_tuple:
            raise ValueError(f"Quantization tuple mismatch for {sample_id}")

        fake_ids = [int(token) for token in fake_record.get("generated_token_ids", [])]
        real_ids = [int(token) for token in real_record.get("generated_token_ids", [])]
        position = first_divergence(fake_ids, real_ids)
        if position is None:
            continue
        if position >= len(fake_ids) or position >= len(real_ids):
            # A stop-only divergence cannot provide two next-token candidates.
            continue
        common_prefix = fake_ids[:position]
        if common_prefix != real_ids[:position]:
            raise AssertionError(f"Invalid common prefix for {sample_id}")
        prompt_ids = [int(token) for token in fake_record["input_token_ids"]]
        prefix_ids = prompt_ids + common_prefix

        fake_truncated = bool(fake_record.get("hit_max_new_tokens"))
        real_truncated = bool(real_record.get("hit_max_new_tokens"))
        fake_correct = bool(fake_record.get("is_correct"))
        real_correct = bool(real_record.get("is_correct"))
        correctness_flip = fake_correct != real_correct
        any_truncation = fake_truncated or real_truncated
        if correctness_flip and any_truncation:
            group = "flip_truncated"
        elif correctness_flip:
            group = "flip_complete"
        elif any_truncation:
            group = "nonflip_truncated"
        else:
            group = "nonflip_complete"

        rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": fake_record.get("dataset_index"),
                "first_divergence_index": position,
                "prompt_token_count": len(prompt_ids),
                "common_generated_prefix_count": len(common_prefix),
                "forced_prefix_token_count": len(prefix_ids),
                "forced_prefix_sha256": token_ids_sha256(prefix_ids),
                "fake_free_token_id": fake_ids[position],
                "real_free_token_id": real_ids[position],
                "fake_correct": fake_correct,
                "real_correct": real_correct,
                "correctness_flip": correctness_flip,
                "fake_hit_max_new_tokens": fake_truncated,
                "real_hit_max_new_tokens": real_truncated,
                "any_truncation": any_truncation,
                "fake_generated_token_count": len(fake_ids),
                "real_generated_token_count": len(real_ids),
                "max_generated_token_count": max(len(fake_ids), len(real_ids)),
                "group": group,
                "input_token_ids_sha256": fake_prompt_hash,
                "quantization_tuple_sha256": fake_tuple or real_tuple,
            }
        )
    return rows


def _match_cost(flip: dict[str, Any], control: dict[str, Any]) -> float:
    truncation_penalty = 8.0 if bool(flip["any_truncation"]) != bool(
        control["any_truncation"]
    ) else 0.0
    prefix_cost = abs(
        math.log1p(int(flip["first_divergence_index"]))
        - math.log1p(int(control["first_divergence_index"]))
    )
    length_cost = abs(
        math.log1p(int(flip["max_generated_token_count"]))
        - math.log1p(int(control["max_generated_token_count"]))
    )
    return truncation_penalty + prefix_cost + length_cost


def select_flip_and_matched_controls(
    rows: Iterable[dict[str, Any]],
    *,
    max_flips: int | None = None,
) -> list[dict[str, Any]]:
    source = [dict(row) for row in rows]
    flips = [row for row in source if bool(row["correctness_flip"])]
    flips.sort(key=lambda row: str(row["sample_id"]))
    if max_flips is not None:
        flips = flips[:max_flips]
    available = {
        str(row["sample_id"]): row
        for row in source
        if not bool(row["correctness_flip"])
    }

    selected: list[dict[str, Any]] = []
    for row in flips:
        item = dict(row)
        item["selection_role"] = "outcome_flip"
        item["matched_to_sample_id"] = None
        item["match_cost"] = None
        selected.append(item)

    # Match the harder truncated cases first so the scarce truncated controls
    # are not consumed by complete cases.
    matching_order = sorted(
        flips,
        key=lambda row: (not bool(row["any_truncation"]), str(row["sample_id"])),
    )
    for flip in matching_order:
        if not available:
            break
        control = min(
            available.values(),
            key=lambda row: (_match_cost(flip, row), str(row["sample_id"])),
        )
        available.pop(str(control["sample_id"]))
        item = dict(control)
        item["selection_role"] = "matched_nonflip_control"
        item["matched_to_sample_id"] = str(flip["sample_id"])
        item["match_cost"] = _match_cost(flip, control)
        selected.append(item)

    selected.sort(
        key=lambda row: (
            0 if row["selection_role"] == "outcome_flip" else 1,
            str(row["sample_id"]),
        )
    )
    return selected


def top_two(logits: torch.Tensor) -> tuple[int, float, int, float]:
    values, indices = torch.topk(logits.float(), k=2)
    return (
        int(indices[0].item()),
        float(values[0].item()),
        int(indices[1].item()),
        float(values[1].item()),
    )


def eos_margin(logits: torch.Tensor, eos_token_ids: Iterable[int]) -> float:
    eos_ids = sorted({int(token) for token in eos_token_ids})
    if not eos_ids:
        raise ValueError("At least one EOS token id is required")
    values = logits.float()
    eos_best = float(values[eos_ids].max().item())
    non_eos = values.clone()
    non_eos[eos_ids] = -torch.inf
    non_eos_best = float(non_eos.max().item())
    return eos_best - non_eos_best


def compare_logit_pair(
    fake_logits: torch.Tensor,
    real_logits: torch.Tensor,
    *,
    eos_token_ids: Iterable[int],
) -> dict[str, Any]:
    if fake_logits.shape != real_logits.shape:
        raise ValueError(
            f"Logit shape mismatch: fake={fake_logits.shape}, real={real_logits.shape}"
        )
    fake_top1, fake_top1_logit, fake_top2, fake_top2_logit = top_two(fake_logits)
    real_top1, real_top1_logit, real_top2, real_top2_logit = top_two(real_logits)
    delta = real_logits.float() - fake_logits.float()
    if fake_top1 == real_top1:
        fake_candidate_gap = 0.0
        real_candidate_gap = 0.0
    else:
        fake_candidate_gap = float(
            (fake_logits[fake_top1] - fake_logits[real_top1]).item()
        )
        real_candidate_gap = float(
            (real_logits[fake_top1] - real_logits[real_top1]).item()
        )
    return {
        "controlled_top1_flip": fake_top1 != real_top1,
        "fake_top1_token_id": fake_top1,
        "real_top1_token_id": real_top1,
        "fake_top2_token_id": fake_top2,
        "real_top2_token_id": real_top2,
        "fake_top1_margin": fake_top1_logit - fake_top2_logit,
        "real_top1_margin": real_top1_logit - real_top2_logit,
        "fake_candidate_gap": fake_candidate_gap,
        "real_candidate_gap": real_candidate_gap,
        "candidate_gap_shift": real_candidate_gap - fake_candidate_gap,
        "fake_eos_margin": eos_margin(fake_logits, eos_token_ids),
        "real_eos_margin": eos_margin(real_logits, eos_token_ids),
        "logit_delta_max_abs": float(delta.abs().max().item()),
        "logit_delta_rms": float(delta.square().mean().sqrt().item()),
    }


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    items = [float(value) for value in values]
    if not items:
        return {"n": 0, "mean": None, "median": None, "max": None}
    return {
        "n": len(items),
        "mean": sum(items) / len(items),
        "median": median(items),
        "max": max(items),
    }


def gap_relation(value: float, *, atol: float = 0.0) -> str:
    """Classify a two-token logit gap without hiding exact ties."""

    gap = float(value)
    if abs(gap) <= float(atol):
        return "tie"
    return "fake_token_preferred" if gap > 0.0 else "real_token_preferred"


def candidate_relation_changed(
    fake_gap: float,
    real_gap: float,
    *,
    atol: float = 0.0,
) -> bool:
    return gap_relation(fake_gap, atol=atol) != gap_relation(real_gap, atol=atol)


def matched_continuation_budget(
    max_new_tokens: int,
    common_generated_prefix_count: int,
    *,
    forced_token_count: int = 1,
) -> int:
    """Preserve the original total generation budget after a forced branch."""

    remaining = (
        int(max_new_tokens)
        - int(common_generated_prefix_count)
        - int(forced_token_count)
    )
    if remaining < 1:
        raise ValueError(
            "No continuation budget remains after the forced branch: "
            f"max_new_tokens={max_new_tokens}, "
            f"common_generated_prefix_count={common_generated_prefix_count}, "
            f"forced_token_count={forced_token_count}"
        )
    return remaining
