from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from .answer_parser import answers_equivalent


def summarize_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "samples": 0,
            "correct": 0,
            "accuracy": None,
            "avg_reasoning_tokens": None,
            "avg_generated_tokens": None,
            "total_latency_seconds": 0.0,
            "tokens_per_second": None,
        }
    correct = sum(bool(record.get("is_correct")) for record in records)
    generated = sum(int(record.get("generated_token_count") or 0) for record in records)
    total_latency = sum(float(record.get("total_generation_latency_seconds") or 0) for record in records)
    reasoning_counts = [int(record.get("reasoning_token_count") or 0) for record in records]
    generation_counts = [int(record.get("generated_token_count") or 0) for record in records]
    return {
        "samples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "avg_reasoning_tokens": mean(reasoning_counts),
        "avg_generated_tokens": mean(generation_counts),
        "total_latency_seconds": total_latency,
        "tokens_per_second": generated / total_latency if total_latency > 0 else None,
        "answer_extraction_failures": sum(
            bool(record.get("answer_extraction_failed")) for record in records
        ),
    }


def bucket_label(value: int, boundaries: list[int]) -> str:
    for start, next_start in zip(boundaries, boundaries[1:]):
        if start <= value < next_start:
            return f"{start}-{next_start - 1}"
    return f"{boundaries[-1]}+"


def contingency(fake: dict[str, Any], real: dict[str, Any]) -> str:
    return (
        f"fake_{'correct' if fake.get('is_correct') else 'wrong'}"
        f"__real_{'correct' if real.get('is_correct') else 'wrong'}"
    )


def compare_conditions(
    bf16: list[dict[str, Any]],
    fake: list[dict[str, Any]],
    real: list[dict[str, Any]],
    boundaries: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    maps = {
        "bf16": {str(row["sample_id"]): row for row in bf16},
        "fake_quant": {str(row["sample_id"]): row for row in fake},
        "real_quant_marlin": {str(row["sample_id"]): row for row in real},
    }
    common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    comparisons: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for sample_id in common_ids:
        b = maps["bf16"][sample_id]
        f = maps["fake_quant"][sample_id]
        r = maps["real_quant_marlin"][sample_id]
        hashes = {
            b.get("input_token_ids_sha256"),
            f.get("input_token_ids_sha256"),
            r.get("input_token_ids_sha256"),
        }
        if len(hashes) != 1:
            raise ValueError(f"Prompt token mismatch for sample_id={sample_id}")
        fake_tuple_hash = f.get("quantization_tuple_sha256")
        real_tuple_hash = r.get("quantization_tuple_sha256")
        if not fake_tuple_hash or not real_tuple_hash:
            raise ValueError(
                f"Missing (q,s,z,g) fingerprint for sample_id={sample_id}; rerun "
                "Fake and Real with --overwrite"
            )
        if fake_tuple_hash != real_tuple_hash:
            raise ValueError(
                f"Fake/Real (q,s,z,g) mismatch for sample_id={sample_id}: "
                f"{fake_tuple_hash} != {real_tuple_hash}"
            )
        if f.get("quantization_source_checkpoint") != r.get(
            "quantization_source_checkpoint"
        ):
            raise ValueError(
                f"Fake/Real checkpoint mismatch for sample_id={sample_id}"
            )
        fake_real_agree = answers_equivalent(
            str(b.get("dataset", "gsm8k")),
            f.get("final_answer"),
            r.get("final_answer"),
        )
        category = contingency(f, r)
        counts[category] += 1
        comparisons.append(
            {
                "sample_id": sample_id,
                "question": b.get("question"),
                "ground_truth": b.get("ground_truth"),
                "input_token_ids_sha256": b.get("input_token_ids_sha256"),
                "quantization_tuple_sha256": fake_tuple_hash,
                "bf16_final_answer": b.get("final_answer"),
                "fake_final_answer": f.get("final_answer"),
                "real_final_answer": r.get("final_answer"),
                "bf16_correct": bool(b.get("is_correct")),
                "fake_correct": bool(f.get("is_correct")),
                "real_correct": bool(r.get("is_correct")),
                "fake_real_answer_agreement": fake_real_agree,
                "contingency": category,
                "gold_calculation_steps": b.get("gold_calculation_step_count"),
                "bf16_reasoning_tokens": int(b.get("reasoning_token_count") or 0),
                "fake_reasoning_tokens": int(f.get("reasoning_token_count") or 0),
                "real_reasoning_tokens": int(r.get("reasoning_token_count") or 0),
                "bf16_length_bucket": bucket_label(
                    int(b.get("reasoning_token_count") or 0), boundaries
                ),
            }
        )

    condition_summaries = {
        "bf16": summarize_predictions([maps["bf16"][item] for item in common_ids]),
        "fake_quant": summarize_predictions([maps["fake_quant"][item] for item in common_ids]),
        "real_quant_marlin": summarize_predictions(
            [maps["real_quant_marlin"][item] for item in common_ids]
        ),
    }
    b_acc = condition_summaries["bf16"]["accuracy"]
    f_acc = condition_summaries["fake_quant"]["accuracy"]
    r_acc = condition_summaries["real_quant_marlin"]["accuracy"]
    summary = {
        "samples_compared": len(common_ids),
        "conditions": condition_summaries,
        "accuracy_differences": {
            "bf16_minus_fake": b_acc - f_acc if common_ids else None,
            "bf16_minus_real": b_acc - r_acc if common_ids else None,
            "shared_quant_fake_minus_real_execution_gap": (
                f_acc - r_acc if common_ids else None
            ),
        },
        "fake_real_answer_agreement": (
            sum(row["fake_real_answer_agreement"] for row in comparisons) / len(comparisons)
            if comparisons
            else None
        ),
        "contingency": dict(counts),
        "length_variables": {
            "S_i_gold": {
                "field": "gold_calculation_step_count",
                "definition": (
                    "Number of <<expression=result>> annotations in the GSM8K "
                    "gold rationale"
                ),
            },
            "L_i_m_gen": {
                "field": "generated_reasoning_token_count",
                "definition": (
                    "Number of reasoning tokens actually generated by condition m "
                    "before </think>"
                ),
            },
        },
        "shared_quantization_verified": bool(common_ids),
        "shared_quantization_tuple_sha256": (
            comparisons[0]["quantization_tuple_sha256"] if comparisons else None
        ),
        "causal_warning": (
            "Fake and Real share serialized (q,s,z,g). Their remaining path differs: "
            "Fake dequantizes to dense BF16 Linear; Real repacks and executes Marlin. "
            "The gap includes packing, dequantization, accumulation, kernel, and runtime "
            "implementation effects."
        ),
    }
    errors = [
        row
        for row in comparisons
        if not (
            row["bf16_correct"] == row["fake_correct"] == row["real_correct"]
            and row["fake_real_answer_agreement"]
        )
    ]
    return comparisons, summary, errors


def length_bucket_rows(
    comparisons: list[dict[str, Any]], boundaries: list[int]
) -> list[dict[str, Any]]:
    labels = [
        *(f"{start}-{end - 1}" for start, end in zip(boundaries, boundaries[1:])),
        f"{boundaries[-1]}+",
    ]
    rows: list[dict[str, Any]] = []
    for label in labels:
        selected = [row for row in comparisons if row["bf16_length_bucket"] == label]
        count = len(selected)
        rows.append(
            {
                "bf16_reasoning_length": label,
                "samples": count,
                "bf16_accuracy": (
                    sum(row["bf16_correct"] for row in selected) / count if count else None
                ),
                "fake_accuracy": (
                    sum(row["fake_correct"] for row in selected) / count if count else None
                ),
                "real_accuracy": (
                    sum(row["real_correct"] for row in selected) / count if count else None
                ),
                "shared_quant_execution_gap": (
                    (
                        sum(row["fake_correct"] for row in selected)
                        - sum(row["real_correct"] for row in selected)
                    )
                    / count
                    if count
                    else None
                ),
                "fake_real_answer_agreement": (
                    sum(row["fake_real_answer_agreement"] for row in selected) / count
                    if count
                    else None
                ),
            }
        )
    return rows


def gold_step_rows(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate GSM8K results by S_i^gold (explicit gold calculation steps)."""

    available = sorted(
        {
            int(row["gold_calculation_steps"])
            for row in comparisons
            if row.get("gold_calculation_steps") is not None
        }
    )
    rows: list[dict[str, Any]] = []
    for steps in available:
        selected = [
            row for row in comparisons if row.get("gold_calculation_steps") == steps
        ]
        count = len(selected)
        rows.append(
            {
                "gold_calculation_steps": steps,
                "samples": count,
                "bf16_accuracy": sum(row["bf16_correct"] for row in selected) / count,
                "fake_accuracy": sum(row["fake_correct"] for row in selected) / count,
                "real_accuracy": sum(row["real_correct"] for row in selected) / count,
                "bf16_avg_generated_reasoning_tokens": mean(
                    row["bf16_reasoning_tokens"] for row in selected
                ),
                "fake_avg_generated_reasoning_tokens": mean(
                    row["fake_reasoning_tokens"] for row in selected
                ),
                "real_avg_generated_reasoning_tokens": mean(
                    row["real_reasoning_tokens"] for row in selected
                ),
                "fake_real_answer_agreement": (
                    sum(row["fake_real_answer_agreement"] for row in selected) / count
                ),
                "shared_quant_execution_gap": (
                    sum(row["fake_correct"] for row in selected)
                    - sum(row["real_correct"] for row in selected)
                )
                / count,
            }
        )
    return rows
