from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.answer_parser import score_record
from src.code_eval import CODE_DATASETS, external_pass_map, read_external_results
from src.config import load_config, results_root
from src.logging_utils import read_jsonl
from src.metrics import summarize_predictions


CONDITIONS = ("bf16", "fake_quant", "real_quant_marlin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import official code-harness pass/fail results into predictions"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=CODE_DATASETS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    prediction_path = (
        results_root(config, args.dataset) / args.condition / "predictions.jsonl"
    )
    records = read_jsonl(prediction_path)
    passed = external_pass_map(read_external_results(args.results), args.dataset)
    expected = {
        str((record.get("code_evaluation") or {}).get("task_id") or record["sample_id"])
        for record in records
    }
    missing = expected - set(passed)
    extra = set(passed) - expected
    if missing or extra:
        raise ValueError(
            f"Task IDs do not match: missing={len(missing)}, extra={len(extra)}"
        )

    updated = []
    for record in records:
        task_id = str(
            (record.get("code_evaluation") or {}).get("task_id")
            or record["sample_id"]
        )
        row = dict(record)
        row["code_execution_passed"] = passed[task_id]
        row["code_evaluation_results_source"] = str(Path(args.results).resolve())
        updated.append(score_record(row, args.dataset))

    temporary = prediction_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in updated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(prediction_path)
    summary = summarize_predictions(updated)
    (prediction_path.parent / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
