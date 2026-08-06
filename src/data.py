from __future__ import annotations

import re
from typing import Any


GSM8K_CALCULATION_RE = re.compile(r"<<.*?>>", flags=re.DOTALL)
CODE_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. Solve the programming problem and "
    "return the final executable Python solution in exactly one ```python code "
    "block. Do not place tests or explanatory prose inside that block."
)


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
    if dataset_config.get("version_tag"):
        kwargs["version_tag"] = dataset_config["version_tag"]
    if dataset_config.get("trust_remote_code") is not None:
        kwargs["trust_remote_code"] = bool(dataset_config["trust_remote_code"])
    dataset = load_dataset(**kwargs)
    limit = max_samples if max_samples is not None else dataset_config.get("max_samples")
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        system_message = None
        code_evaluation = None
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
        elif dataset_name == "humaneval":
            task_id = str(row["task_id"])
            question = (
                "Complete the Python function below so that it satisfies its "
                "docstring. Return a complete executable solution.\n\n"
                f"```python\n{row['prompt'].rstrip()}\n```"
            )
            ground_truth = str(row["canonical_solution"])
            category = "humaneval"
            gold_calculation_step_count = None
            gold_calculation_step_method = None
            system_message = CODE_SYSTEM_MESSAGE
            code_evaluation = {
                "evaluator": "openai_human_eval",
                "task_id": task_id,
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
        elif dataset_name == "mbpp":
            task_id = str(row["task_id"])
            tests = [str(item) for item in (row.get("test_list") or [])]
            test_imports = [str(item) for item in (row.get("test_imports") or [])]
            test_setup_code = str(row.get("test_setup_code") or "")
            if test_imports:
                test_setup_code = "\n".join(test_imports)
            shown_tests = "\n".join(tests)
            question = (
                f"{row['prompt'].strip()}\n\n"
                "Your solution must satisfy these assertions:\n"
                f"```python\n{shown_tests}\n```"
            )
            ground_truth = str(row["code"])
            category = "mbpp"
            gold_calculation_step_count = None
            gold_calculation_step_method = None
            system_message = CODE_SYSTEM_MESSAGE
            code_evaluation = {
                "evaluator": "mbpp_tests",
                "task_id": task_id,
                "test_list": tests,
                "test_setup_code": test_setup_code,
                "test_imports": test_imports,
                "challenge_test_list": [
                    str(item) for item in (row.get("challenge_test_list") or [])
                ],
            }
        elif dataset_name == "livecodebench":
            task_id = str(row["question_id"])
            starter_code = str(row.get("starter_code") or "")
            if starter_code:
                format_instruction = (
                    "Use the following starter code and return the completed "
                    f"solution.\n```python\n{starter_code}\n```"
                )
            else:
                format_instruction = (
                    "Write a complete program that reads from stdin and writes to "
                    "stdout. Do not hard-code the sample inputs."
                )
            question = (
                f"### Question:\n{str(row['question_content']).strip()}\n\n"
                f"### Format:\n{format_instruction}"
            )
            ground_truth = ""
            category = str(row.get("difficulty") or "livecodebench")
            gold_calculation_step_count = None
            gold_calculation_step_method = None
            system_message = CODE_SYSTEM_MESSAGE
            code_evaluation = {
                "evaluator": "livecodebench",
                "task_id": task_id,
                "question_title": row.get("question_title"),
                "question_content": row.get("question_content"),
                "platform": row.get("platform"),
                "contest_id": row.get("contest_id"),
                "contest_date": row.get("contest_date"),
                "starter_code": starter_code,
                "difficulty": row.get("difficulty"),
                "release_version": dataset_config.get("version_tag"),
            }
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        raw_sample_id = row.get("task_id")
        if raw_sample_id is None:
            raw_sample_id = row.get("question_id")
        if raw_sample_id is None:
            raw_sample_id = row.get("id")
        sample_id = str(
            raw_sample_id
            if raw_sample_id is not None
            else f"{dataset_name}-{index:05d}"
        )
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "dataset_index": index,
                "question": question,
                "ground_truth": ground_truth,
                "category": category,
                "system_message": system_message,
                "code_evaluation": code_evaluation,
                # S_i^gold: difficulty/solution-complexity proxy from the gold solution.
                "gold_calculation_step_count": gold_calculation_step_count,
                "gold_calculation_step_method": gold_calculation_step_method,
            }
        )
    return samples
