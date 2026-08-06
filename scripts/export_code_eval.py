from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.code_eval import CODE_DATASETS, export_record
from src.config import load_config, results_root
from src.logging_utils import read_jsonl


CONDITIONS = ("bf16", "fake_quant", "real_quant_marlin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export generations for an official code-execution harness"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=CODE_DATASETS, required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and args.condition is None:
        parser.error("Choose --condition or --all")

    config = load_config(args.config)
    selected = CONDITIONS if args.all else (args.condition,)
    for condition in selected:
        condition_dir = results_root(config, args.dataset) / condition
        records = read_jsonl(condition_dir / "predictions.jsonl")
        if not records:
            raise FileNotFoundError(condition_dir / "predictions.jsonl")
        exported = [export_record(record, args.dataset) for record in records]
        suffix = "json" if args.dataset == "livecodebench" else "jsonl"
        target = condition_dir / f"official_eval_input.{suffix}"
        if suffix == "json":
            target.write_text(
                json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            with target.open("w", encoding="utf-8") as handle:
                for row in exported:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{condition}: {len(exported)} records -> {target}")


if __name__ == "__main__":
    main()
