from __future__ import annotations

import re
from typing import Any


GSM8K_CALCULATION_RE = re.compile(r"<<.*?>>", flags=re.DOTALL)


def count_gsm8k_gold_calculation_steps(answer: str) -> int:
    """Count GSM8K's explicit calculator annotations as gold calculation steps."""

    rationale = answer.split("####", maxsplit=1)[0]
    return len(GSM8K_CALCULATION_RE.findall(rationale))


def load_samples(
    config: dict[str, Any],
    dataset_name: str,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset_config = config["datasets"][dataset_name]
    kwargs: dict[str, Any] = {
        "path": dataset_config["path"],
        "split": dataset_config["split"],
    }
    if dataset_config.get("name"):
        kwargs["name"] = dataset_config["name"]
    dataset = load_dataset(**kwargs)
    limit = max_samples if max_samples is not None else dataset_config.get("max_samples")
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        if dataset_name == "gsm8k":
            question = row["question"]
            ground_truth = row["answer"]
            category = "gsm8k"
            gold_calculation_step_count = count_gsm8k_gold_calculation_steps(
                ground_truth
            )
            gold_calculation_step_method = "gsm8k_<<expression=result>>_annotation_count"
        elif dataset_name == "math500":
            question = row["problem"]
            ground_truth = row["answer"]
            category = row.get("subject") or row.get("type") or "math"
            gold_calculation_step_count = None
            gold_calculation_step_method = None
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        sample_id = str(row.get("id", f"{dataset_name}-{index:05d}"))
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "dataset_index": index,
                "question": question,
                "ground_truth": ground_truth,
                "category": category,
                # S_i^gold: difficulty/solution-complexity proxy from the gold solution.
                "gold_calculation_step_count": gold_calculation_step_count,
                "gold_calculation_step_method": gold_calculation_step_method,
            }
        )
    return samples
