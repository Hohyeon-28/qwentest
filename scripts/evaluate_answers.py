from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.answer_parser import score_record
from src.config import load_config, results_root
from src.logging_utils import read_jsonl
from src.metrics import summarize_predictions


CONDITIONS = ("bf16", "fake_quant", "real_quant_marlin")


def evaluate(path: Path, dataset: str) -> dict:
    records = [score_record(record, dataset) for record in read_jsonl(path)]
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
    summary = summarize_predictions(records)
    (path.parent / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=("gsm8k", "math500"), default="gsm8k")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and args.condition is None:
        parser.error("Choose --condition or --all")
    config = load_config(args.config)
    selected = CONDITIONS if args.all else (args.condition,)
    summaries = {}
    for condition in selected:
        path = results_root(config, args.dataset) / condition / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        summaries[condition] = evaluate(path, args.dataset)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
