import math

from src.metrics import (
    exact_mcnemar_p,
    length_quantile_rows,
    length_trend_statistics,
    summarize_predictions,
)


def comparison(index: int) -> dict:
    return {
        "sample_id": f"sample-{index}",
        "bf16_reasoning_tokens": index * 100,
        "bf16_reasoning_length_censored": False,
        "bf16_correct": index % 2 == 0,
        "fake_correct": index % 3 != 0,
        "real_correct": index % 4 != 0,
        "fake_real_answer_agreement": index < 5,
    }


def test_length_quantiles_are_balanced_and_ordered():
    rows = length_quantile_rows([comparison(index) for index in range(1, 11)], 5)
    assert [row["samples"] for row in rows] == [2, 2, 2, 2, 2]
    assert rows[0]["min_bf16_reasoning_tokens"] == 100
    assert rows[-1]["max_bf16_reasoning_tokens"] == 1000


def test_censored_rows_are_excluded_from_quantiles_and_trend():
    records = [comparison(index) for index in range(1, 11)]
    records[-1]["bf16_reasoning_length_censored"] = True
    rows = length_quantile_rows(records, 3)
    assert sum(row["samples"] for row in rows) == 9
    trend = length_trend_statistics(records, trials=100, seed=7)
    assert trend["samples"] == 9


def test_mcnemar_exact_probability_is_two_sided():
    assert math.isclose(
        exact_mcnemar_p(40, 32), 0.40957939592707177, rel_tol=1e-12
    )
    assert exact_mcnemar_p(0, 0) == 1.0


def test_summary_reports_censoring_and_completed_average():
    records = [
        {
            "is_correct": True,
            "generated_token_count": 10,
            "reasoning_token_count": 6,
            "total_generation_latency_seconds": 1.0,
            "answer_extraction_failed": False,
            "hit_max_new_tokens": False,
            "reasoning_length_censored": False,
        },
        {
            "is_correct": False,
            "generated_token_count": 20,
            "reasoning_token_count": 20,
            "total_generation_latency_seconds": 2.0,
            "answer_extraction_failed": True,
            "hit_max_new_tokens": True,
            "reasoning_length_censored": True,
        },
    ]
    summary = summarize_predictions(records)
    assert summary["max_token_truncations"] == 1
    assert summary["reasoning_length_censored"] == 1
    assert summary["completed_reasoning_traces"] == 1
    assert summary["avg_completed_reasoning_tokens"] == 6
