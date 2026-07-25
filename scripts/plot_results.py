from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, results_root
from src.logging_utils import read_jsonl


LABELS = {
    "bf16": "BF16",
    "fake_quant": "Fake INT4 (shared GPTQ tuple)",
    "real_quant_marlin": "GPTQ-Marlin INT4",
}
COLORS = {
    "bf16": "#4c78a8",
    "fake_quant": "#f58518",
    "real_quant_marlin": "#54a24b",
}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=("gsm8k", "math500"), default="gsm8k")
    args = parser.parse_args()
    config = load_config(args.config)
    root = results_root(config, args.dataset)
    output = root / "plots"
    output.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    summary = json.loads((root / "comparisons" / "summary.json").read_text(encoding="utf-8"))
    buckets = pd.read_csv(root / "comparisons" / "length_bucket.csv")
    conditions = ("bf16", "fake_quant", "real_quant_marlin")

    fig, ax = plt.subplots(figsize=(7, 4))
    values = [summary["conditions"][item]["accuracy"] for item in conditions]
    ax.bar([LABELS[item] for item in conditions], values, color=[COLORS[item] for item in conditions])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final-answer accuracy")
    ax.set_title(f"{args.dataset}: accuracy by condition")
    _save(fig, output / "accuracy_by_precision.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(buckets))
    fake_drop = buckets["bf16_accuracy"] - buckets["fake_accuracy"]
    real_drop = buckets["bf16_accuracy"] - buckets["real_accuracy"]
    ax.plot(x, fake_drop, marker="o", label="BF16 - Fake")
    ax.plot(x, real_drop, marker="o", label="BF16 - Real")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, buckets["bf16_reasoning_length"])
    ax.set_ylabel("Accuracy drop")
    ax.set_xlabel("BF16 reasoning-token bucket")
    ax.legend()
    _save(fig, output / "accuracy_drop_by_length.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, buckets["fake_real_answer_agreement"], color="#b279a2")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, buckets["bf16_reasoning_length"])
    ax.set_ylabel("Fake/Real final-answer agreement")
    ax.set_xlabel("BF16 reasoning-token bucket")
    _save(fig, output / "agreement_by_length.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    latency = [
        summary["conditions"][item]["total_latency_seconds"] for item in conditions
    ]
    ax.bar([LABELS[item] for item in conditions], latency, color=[COLORS[item] for item in conditions])
    ax.set_ylabel("Total generation latency (seconds)")
    ax.set_title("Latency is descriptive; Fake uses BF16 GEMM")
    _save(fig, output / "latency_by_precision.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for condition in conditions:
        records = read_jsonl(root / condition / "predictions.jsonl")
        lengths = [int(row.get("reasoning_token_count") or 0) for row in records]
        ax.hist(
            lengths,
            bins=30,
            alpha=0.45,
            label=LABELS[condition],
            color=COLORS[condition],
        )
    ax.set_xlabel("Reasoning tokens")
    ax.set_ylabel("Samples")
    ax.legend()
    _save(fig, output / "token_length_distribution.png")

    if args.dataset == "gsm8k":
        gold_steps = pd.read_csv(root / "comparisons" / "accuracy_by_gold_steps.csv")
        x_gold = gold_steps["gold_calculation_steps"]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for condition, column in (
            ("bf16", "bf16_accuracy"),
            ("fake_quant", "fake_accuracy"),
            ("real_quant_marlin", "real_accuracy"),
        ):
            ax.plot(
                x_gold,
                gold_steps[column],
                marker="o",
                label=LABELS[condition],
                color=COLORS[condition],
            )
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$S_i^{gold}$: gold calculation steps")
        ax.set_ylabel("Final-answer accuracy")
        ax.legend()
        _save(fig, output / "accuracy_by_gold_steps.png")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for condition, column in (
            ("bf16", "bf16_avg_generated_reasoning_tokens"),
            ("fake_quant", "fake_avg_generated_reasoning_tokens"),
            ("real_quant_marlin", "real_avg_generated_reasoning_tokens"),
        ):
            ax.plot(
                x_gold,
                gold_steps[column],
                marker="o",
                label=LABELS[condition],
                color=COLORS[condition],
            )
        ax.set_xlabel(r"$S_i^{gold}$: gold calculation steps")
        ax.set_ylabel(r"Mean $L_{i,m}^{gen}$: generated reasoning tokens")
        ax.legend()
        _save(fig, output / "generated_reasoning_tokens_by_gold_steps.png")


if __name__ == "__main__":
    main()
