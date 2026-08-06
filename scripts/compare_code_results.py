from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.code_eval import CODE_DATASETS
from src.config import load_config, results_root
from src.logging_utils import read_jsonl
from src.metrics import exact_mcnemar_p, wilson_interval


CONDITIONS = ("bf16", "fake_quant", "real_quant_marlin")


def _index(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    return {str(row["sample_id"]): row for row in rows}


def _normalized_code(record: dict) -> str:
    text = str(record.get("final_answer") or "")
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pass@1 and length trends after official code evaluation"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=CODE_DATASETS, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = results_root(config, args.dataset)
    indexed = {
        condition: _index(root / condition / "predictions.jsonl")
        for condition in CONDITIONS
    }
    id_sets = {condition: set(rows) for condition, rows in indexed.items()}
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        raise ValueError(
            "Condition sample IDs differ: "
            + ", ".join(f"{name}={len(ids)}" for name, ids in id_sets.items())
        )
    sample_ids = sorted(id_sets["bf16"])
    comparisons = []
    for sample_id in sample_ids:
        records = {name: indexed[name][sample_id] for name in CONDITIONS}
        for name, record in records.items():
            if not isinstance(record.get("is_correct"), bool):
                raise ValueError(
                    f"{name}/{sample_id} is not scored. Import official harness "
                    "results for all conditions first."
                )
        hashes = {record.get("input_token_ids_sha256") for record in records.values()}
        if len(hashes) != 1:
            raise ValueError(f"Prompt token hashes differ for {sample_id}")
        bf16 = records["bf16"]
        fake = records["fake_quant"]
        real = records["real_quant_marlin"]
        comparisons.append(
            {
                "sample_id": sample_id,
                "bf16_reasoning_tokens": int(bf16.get("reasoning_token_count") or 0),
                "bf16_reasoning_complete": bool(bf16.get("reasoning_complete")),
                "bf16_passed": bool(bf16["is_correct"]),
                "fake_passed": bool(fake["is_correct"]),
                "real_passed": bool(real["is_correct"]),
                "fake_real_correctness_disagreement": bool(fake["is_correct"])
                != bool(real["is_correct"]),
                "fake_real_exact_code_agreement": _normalized_code(fake)
                == _normalized_code(real),
            }
        )

    fake_only = sum(row["fake_passed"] and not row["real_passed"] for row in comparisons)
    real_only = sum(row["real_passed"] and not row["fake_passed"] for row in comparisons)
    summary = {
        "dataset": args.dataset,
        "samples": len(comparisons),
        "metric": "official-harness deterministic pass@1",
        "pass_at_1": {},
        "fake_real": {
            "correctness_agreement_rate": (
                sum(not row["fake_real_correctness_disagreement"] for row in comparisons)
                / len(comparisons)
                if comparisons
                else None
            ),
            "exact_code_agreement_rate": (
                sum(row["fake_real_exact_code_agreement"] for row in comparisons)
                / len(comparisons)
                if comparisons
                else None
            ),
            "fake_only_pass": fake_only,
            "real_only_pass": real_only,
            "exact_mcnemar_p": exact_mcnemar_p(fake_only, real_only),
        },
    }
    for condition, key in (
        ("bf16", "bf16_passed"),
        ("fake_quant", "fake_passed"),
        ("real_quant_marlin", "real_passed"),
    ):
        passed = sum(row[key] for row in comparisons)
        summary["pass_at_1"][condition] = {
            "passed": passed,
            "rate": passed / len(comparisons) if comparisons else None,
            "ci95_wilson": wilson_interval(passed, len(comparisons)),
        }

    completed = sorted(
        (row for row in comparisons if row["bf16_reasoning_complete"]),
        key=lambda row: row["bf16_reasoning_tokens"],
    )
    quintiles = []
    for index in range(5):
        group = completed[index * len(completed) // 5 : (index + 1) * len(completed) // 5]
        if not group:
            continue
        quintiles.append(
            {
                "quintile": index + 1,
                "samples": len(group),
                "min_bf16_reasoning_tokens": group[0]["bf16_reasoning_tokens"],
                "max_bf16_reasoning_tokens": group[-1]["bf16_reasoning_tokens"],
                "bf16_pass_at_1": sum(row["bf16_passed"] for row in group) / len(group),
                "fake_pass_at_1": sum(row["fake_passed"] for row in group) / len(group),
                "real_pass_at_1": sum(row["real_passed"] for row in group) / len(group),
                "fake_real_correctness_disagreement_rate": sum(
                    row["fake_real_correctness_disagreement"] for row in group
                )
                / len(group),
            }
        )
    summary["bf16_length_quintiles"] = quintiles

    output = root / "code_comparisons"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "length_quintiles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quintiles[0]) if quintiles else [])
        if quintiles:
            writer.writeheader()
            writer.writerows(quintiles)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
