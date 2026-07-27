from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, results_root
from src.logging_utils import read_jsonl
from src.metrics import (
    bucket_label,
    compare_conditions,
    gold_step_rows,
    length_bucket_rows,
    length_quantile_rows,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def own_length_rows(
    conditions: dict[str, list[dict]], boundaries: list[int]
) -> list[dict]:
    labels = [
        *(f"{a}-{b - 1}" for a, b in zip(boundaries, boundaries[1:])),
        f"{boundaries[-1]}+",
    ]
    rows = []
    for condition, records in conditions.items():
        for label in labels:
            selected = [
                row
                for row in records
                if not row.get("reasoning_length_censored")
                and bucket_label(
                    int(row.get("reasoning_token_count") or 0), boundaries
                )
                == label
            ]
            rows.append(
                {
                    "condition": condition,
                    "own_reasoning_length": label,
                    "samples": len(selected),
                    "accuracy": (
                        sum(bool(row.get("is_correct")) for row in selected) / len(selected)
                        if selected
                        else None
                    ),
                }
            )
        censored = [
            row for row in records if row.get("reasoning_length_censored")
        ]
        rows.append(
            {
                "condition": condition,
                "own_reasoning_length": "censored_at_max_tokens",
                "samples": len(censored),
                "accuracy": (
                    sum(bool(row.get("is_correct")) for row in censored)
                    / len(censored)
                    if censored
                    else None
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=("gsm8k", "math500"), default="gsm8k")
    args = parser.parse_args()
    config = load_config(args.config)
    root = results_root(config, args.dataset)
    conditions = {
        condition: read_jsonl(root / condition / "predictions.jsonl")
        for condition in ("bf16", "fake_quant", "real_quant_marlin")
    }
    for condition, records in conditions.items():
        if not records:
            raise RuntimeError(f"No predictions found for {condition}")
    boundaries = [int(item) for item in config["evaluation"]["length_buckets"]]
    comparisons, summary, errors = compare_conditions(
        conditions["bf16"],
        conditions["fake_quant"],
        conditions["real_quant_marlin"],
        boundaries,
        permutation_trials=int(config["evaluation"].get("permutation_trials", 10000)),
        permutation_seed=int(config["generation"]["seed"]),
        primary_length_outcome=str(
            config["evaluation"].get(
                "primary_length_outcome",
                "fake_real_answer_disagreement",
            )
        ),
    )
    output = root / "comparisons"
    write_jsonl(output / "sample_comparison.jsonl", comparisons)
    write_jsonl(output / "error_cases.jsonl", errors)
    write_csv(output / "length_bucket.csv", length_bucket_rows(comparisons, boundaries))
    write_csv(
        output / "length_quantile.csv",
        length_quantile_rows(
            comparisons,
            int(config["evaluation"].get("length_quantile_bins", 5)),
        ),
    )
    if args.dataset == "gsm8k":
        write_csv(output / "accuracy_by_gold_steps.csv", gold_step_rows(comparisons))
    write_csv(
        output / "length_bucket_own.csv", own_length_rows(conditions, boundaries)
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
