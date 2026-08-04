import torch

from src.first_divergence import (
    build_divergence_rows,
    compare_logit_pair,
    first_divergence,
    select_flip_and_matched_controls,
)


def record(sample_id, generated, *, correct, truncated=False):
    return {
        "sample_id": sample_id,
        "dataset_index": int(sample_id.split("-")[-1]),
        "input_token_ids": [10, 11],
        "input_token_ids_sha256": "same-prompt",
        "quantization_tuple_sha256": "same-tuple",
        "generated_token_ids": generated,
        "is_correct": correct,
        "hit_max_new_tokens": truncated,
    }


def test_first_divergence_handles_equal_and_prefix_sequences():
    assert first_divergence([1, 2], [1, 2]) is None
    assert first_divergence([1, 2], [1, 3]) == 1
    assert first_divergence([1], [1, 2]) == 1


def test_candidate_selection_includes_flips_and_unique_controls():
    fake = [
        record("sample-0", [1, 2, 3], correct=True),
        record("sample-1", [1, 4, 5], correct=False, truncated=True),
        record("sample-2", [1, 6, 7], correct=True),
        record("sample-3", [1, 8, 9], correct=False),
    ]
    real = [
        record("sample-0", [1, 9, 3], correct=False),
        record("sample-1", [1, 9, 5], correct=True, truncated=True),
        record("sample-2", [1, 9, 7], correct=True),
        record("sample-3", [1, 9, 9], correct=False),
    ]
    rows = build_divergence_rows(fake, real)
    selected = select_flip_and_matched_controls(rows)
    assert sum(row["selection_role"] == "outcome_flip" for row in selected) == 2
    controls = [row for row in selected if row["selection_role"] != "outcome_flip"]
    assert len(controls) == 2
    assert len({row["sample_id"] for row in controls}) == 2


def test_logit_comparison_detects_candidate_gap_crossing():
    fake = torch.tensor([0.0, 3.0, 2.9, -1.0])
    real = torch.tensor([0.0, 2.8, 3.1, -1.0])
    result = compare_logit_pair(fake, real, eos_token_ids=[3])
    assert result["controlled_top1_flip"] is True
    assert result["fake_candidate_gap"] > 0
    assert result["real_candidate_gap"] < 0
    assert result["candidate_gap_shift"] < 0
