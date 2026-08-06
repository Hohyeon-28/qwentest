from __future__ import annotations

import math
import random
from collections import Counter
from statistics import mean
from typing import Any

from .answer_parser import answers_equivalent


def wilson_interval(correct: int, total: int, z: float = 1.95996398454) -> list[float] | None:
    if total <= 0:
        return None
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = 0.5**discordant
    cumulative = probability
    for successes in range(1, tail + 1):
        probability *= (discordant - successes + 1) / successes
        cumulative += probability
    return min(1.0, 2 * cumulative)


def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean = mean(x)
    y_mean = mean(y)
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(x_centered, y_centered)) / denominator


def _permutation_correlation(
    x: list[float],
    y: list[float],
    *,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    observed = _pearson_correlation(x, y)
    if observed is None:
        return {"correlation": None, "permutation_p_two_sided": None}
    generator = random.Random(seed)
    permuted = list(y)
    exceedances = 0
    for _ in range(trials):
        generator.shuffle(permuted)
        candidate = _pearson_correlation(x, permuted)
        if candidate is not None and abs(candidate) >= abs(observed):
            exceedances += 1
    return {
        "correlation": observed,
        "permutation_p_two_sided": (exceedances + 1) / (trials + 1),
    }


def length_trend_statistics(
    comparisons: list[dict[str, Any]],
    *,
    trials: int,
    seed: int,
    require_all_complete: bool = False,
) -> dict[str, Any]:
    eligible = [
        row
        for row in comparisons
        if not row.get("bf16_reasoning_incomplete")
        and (
            not require_all_complete
            or not (
                row.get("bf16_reasoning_incomplete")
                or row.get("fake_reasoning_incomplete")
                or row.get("real_reasoning_incomplete")
            )
        )
    ]
    log_lengths = [math.log1p(row["bf16_reasoning_tokens"]) for row in eligible]
    outcomes = {
        "fake_real_answer_disagreement": [
            float(not row["fake_real_answer_agreement"]) for row in eligible
        ],
        "fake_failure_when_bf16_correct": [
            float(row["bf16_correct"] and not row["fake_correct"])
            for row in eligible
        ],
        "real_failure_when_bf16_correct": [
            float(row["bf16_correct"] and not row["real_correct"])
            for row in eligible
        ],
        "real_failure_when_fake_correct": [
            float(row["fake_correct"] and not row["real_correct"])
            for row in eligible
        ],
    }
    result: dict[str, Any] = {
        "reference_length": "log1p(BF16 completed reasoning tokens)",
        "requires_all_conditions_complete": require_all_complete,
        "samples": len(eligible),
        "permutation_trials": trials,
        "interpretation": (
            "Positive correlation means the event is more common at longer BF16 "
            "reasoning lengths. This is an associational, not causal, analysis."
        ),
        "outcomes": {},
    }
    for offset, (name, values) in enumerate(outcomes.items()):
        result["outcomes"][name] = {
            "events": int(sum(values)),
            **_permutation_correlation(
                log_lengths,
                values,
                trials=trials,
                seed=seed + offset,
            ),
        }
    available = sorted(
        (
            (details["permutation_p_two_sided"], name)
            for name, details in result["outcomes"].items()
            if details["permutation_p_two_sided"] is not None
        ),
        key=lambda item: item[0],
    )
    previous = 0.0
    total_tests = len(available)
    for rank, (p_value, name) in enumerate(available):
        adjusted = min(1.0, (total_tests - rank) * p_value)
        adjusted = max(previous, adjusted)
        result["outcomes"][name]["holm_adjusted_p"] = adjusted
        previous = adjusted
    for name, details in result["outcomes"].items():
        details.setdefault("holm_adjusted_p", None)
    return result


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
            "request_latency_tokens_per_second": None,
            "aggregate_generation_tokens_per_second": None,
            "aggregate_generation_wall_seconds": 0.0,
            "scored_samples": 0,
            "pending_code_evaluation": 0,
            "answer_extraction_failures": 0,
            "max_token_truncations": 0,
            "reasoning_length_censored": 0,
            "reasoning_incomplete": 0,
            "completed_reasoning_traces": 0,
            "avg_completed_reasoning_tokens": None,
        }
    scored = [record for record in records if isinstance(record.get("is_correct"), bool)]
    correct = sum(bool(record.get("is_correct")) for record in scored)
    generated = sum(int(record.get("generated_token_count") or 0) for record in records)
    total_latency = sum(float(record.get("total_generation_latency_seconds") or 0) for record in records)
    batch_seconds: dict[str, float] = {}
    aggregate_available = True
    for index, record in enumerate(records):
        batch_id = record.get("generation_batch_id")
        batch_elapsed = record.get("batch_generation_latency_seconds")
        if batch_id is None:
            if int(record.get("batch_size_used") or 1) == 1 and batch_elapsed is not None:
                batch_id = f"legacy-single-{index}"
            else:
                aggregate_available = False
                continue
        if batch_elapsed is None:
            aggregate_available = False
            continue
        key = str(batch_id)
        batch_seconds[key] = max(batch_seconds.get(key, 0.0), float(batch_elapsed))
    aggregate_seconds = sum(batch_seconds.values()) if aggregate_available else None
    reasoning_counts = [int(record.get("reasoning_token_count") or 0) for record in records]
    completed_reasoning_counts = [
        int(record.get("reasoning_token_count") or 0)
        for record in records
        if bool(record.get("reasoning_complete"))
    ]
    generation_counts = [int(record.get("generated_token_count") or 0) for record in records]
    return {
        "samples": len(records),
        "correct": correct,
        "scored_samples": len(scored),
        "pending_code_evaluation": len(records) - len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "accuracy_ci95_wilson": wilson_interval(correct, len(scored)),
        "avg_reasoning_tokens": mean(reasoning_counts),
        "avg_generated_tokens": mean(generation_counts),
        "total_latency_seconds": total_latency,
        "tokens_per_second": generated / total_latency if total_latency > 0 else None,
        "request_latency_tokens_per_second": (
            generated / total_latency if total_latency > 0 else None
        ),
        "tokens_per_second_definition": (
            "generated tokens / sum of per-request latencies; not aggregate throughput"
        ),
        "aggregate_generation_wall_seconds": aggregate_seconds,
        "aggregate_generation_tokens_per_second": (
            generated / aggregate_seconds
            if aggregate_seconds is not None and aggregate_seconds > 0
            else None
        ),
        "aggregate_throughput_definition": (
            "generated tokens / sum of unique batch wall times"
        ),
        "answer_extraction_failures": sum(
            bool(record.get("answer_extraction_failed")) for record in records
        ),
        "max_token_truncations": sum(
            bool(record.get("hit_max_new_tokens")) for record in records
        ),
        "reasoning_length_censored": sum(
            bool(record.get("reasoning_length_censored")) for record in records
        ),
        "reasoning_incomplete": sum(
            bool(record.get("reasoning_incomplete")) for record in records
        ),
        "completed_reasoning_traces": len(completed_reasoning_counts),
        "avg_completed_reasoning_tokens": (
            mean(completed_reasoning_counts)
            if completed_reasoning_counts
            else None
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
    *,
    permutation_trials: int = 10000,
    permutation_seed: int = 20250724,
    primary_length_outcome: str = "fake_real_answer_disagreement",
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
                "bf16_reasoning_length_censored": bool(
                    b.get("reasoning_length_censored")
                ),
                "fake_reasoning_length_censored": bool(
                    f.get("reasoning_length_censored")
                ),
                "real_reasoning_length_censored": bool(
                    r.get("reasoning_length_censored")
                ),
                "bf16_reasoning_incomplete": bool(b.get("reasoning_incomplete")),
                "fake_reasoning_incomplete": bool(f.get("reasoning_incomplete")),
                "real_reasoning_incomplete": bool(r.get("reasoning_incomplete")),
                "bf16_length_bucket": (
                    None
                    if b.get("reasoning_incomplete")
                    else bucket_label(
                        int(b.get("reasoning_token_count") or 0), boundaries
                    )
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
    pair_records = {
        "bf16_vs_fake": (
            [maps["bf16"][item] for item in common_ids],
            [maps["fake_quant"][item] for item in common_ids],
        ),
        "bf16_vs_real": (
            [maps["bf16"][item] for item in common_ids],
            [maps["real_quant_marlin"][item] for item in common_ids],
        ),
        "fake_vs_real": (
            [maps["fake_quant"][item] for item in common_ids],
            [maps["real_quant_marlin"][item] for item in common_ids],
        ),
    }
    paired_accuracy_tests = {}
    for name, (first, second) in pair_records.items():
        first_only = sum(
            bool(a.get("is_correct")) and not bool(b.get("is_correct"))
            for a, b in zip(first, second)
        )
        second_only = sum(
            not bool(a.get("is_correct")) and bool(b.get("is_correct"))
            for a, b in zip(first, second)
        )
        paired_accuracy_tests[name] = {
            "first_correct_second_wrong": first_only,
            "first_wrong_second_correct": second_only,
            "mcnemar_exact_p_two_sided": exact_mcnemar_p(
                first_only, second_only
            ),
        }
    length_trend = length_trend_statistics(
        comparisons,
        trials=permutation_trials,
        seed=permutation_seed,
    )
    all_complete_length_trend = length_trend_statistics(
        comparisons,
        trials=permutation_trials,
        seed=permutation_seed,
        require_all_complete=True,
    )
    if primary_length_outcome not in length_trend["outcomes"]:
        raise ValueError(
            f"Unknown primary length outcome: {primary_length_outcome}"
        )
    length_trend["primary_outcome"] = primary_length_outcome
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
        "paired_accuracy_tests": paired_accuracy_tests,
        "bf16_length_trend": length_trend,
        "bf16_length_trend_all_conditions_complete": all_complete_length_trend,
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
                "censoring": (
                    "Traces that reach max_new_tokens without </think> are marked "
                    "reasoning_length_censored and excluded from exact length buckets."
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


def length_quantile_rows(
    comparisons: list[dict[str, Any]],
    bins: int,
    *,
    require_all_complete: bool = False,
) -> list[dict[str, Any]]:
    eligible = sorted(
        (
            row
            for row in comparisons
            if not row.get("bf16_reasoning_incomplete")
            and (
                not require_all_complete
                or not (
                    row.get("bf16_reasoning_incomplete")
                    or row.get("fake_reasoning_incomplete")
                    or row.get("real_reasoning_incomplete")
                )
            )
        ),
        key=lambda row: (row["bf16_reasoning_tokens"], row["sample_id"]),
    )
    if not eligible or bins <= 0:
        return []
    bins = min(bins, len(eligible))
    groups: list[list[dict[str, Any]]] = [[] for _ in range(bins)]
    for rank, row in enumerate(eligible):
        group_index = min(bins - 1, rank * bins // len(eligible))
        groups[group_index].append(row)

    rows: list[dict[str, Any]] = []
    for index, selected in enumerate(groups, start=1):
        count = len(selected)
        rows.append(
            {
                "bf16_reasoning_length_quantile": f"Q{index}",
                "samples": count,
                "min_bf16_reasoning_tokens": min(
                    row["bf16_reasoning_tokens"] for row in selected
                ),
                "max_bf16_reasoning_tokens": max(
                    row["bf16_reasoning_tokens"] for row in selected
                ),
                "bf16_accuracy": (
                    sum(row["bf16_correct"] for row in selected) / count
                ),
                "fake_accuracy": (
                    sum(row["fake_correct"] for row in selected) / count
                ),
                "real_accuracy": (
                    sum(row["real_correct"] for row in selected) / count
                ),
                "shared_quant_execution_gap": (
                    (
                        sum(row["fake_correct"] for row in selected)
                        - sum(row["real_correct"] for row in selected)
                    )
                    / count
                ),
                "fake_real_answer_agreement": (
                    sum(row["fake_real_answer_agreement"] for row in selected)
                    / count
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
