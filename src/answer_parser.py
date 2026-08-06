from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .code_eval import CODE_DATASETS, extract_python_code


NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _last_boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return None
    start = starts[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    if "####" in text:
        text = text.rsplit("####", maxsplit=1)[-1]
    matches = NUMBER_RE.findall(text)
    return matches[-1].replace(",", "") if matches else None


def extract_math_answer(text: str) -> str | None:
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed
    final_markers = (
        r"(?:final answer|answer)\s*(?:is|:)\s*(.+)",
        r"\\text\{Answer\}\s*[:=]\s*(.+)",
    )
    for pattern in final_markers:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip().rstrip(".")
    matches = NUMBER_RE.findall(text)
    return matches[-1].replace(",", "") if matches else None


def extract_answer(dataset: str, text: str) -> str | None:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(text)
    if dataset == "math500":
        return extract_math_answer(text)
    raise ValueError(f"Unsupported dataset: {dataset}")


def ground_truth_answer(dataset: str, raw: str) -> str | None:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(raw)
    if dataset == "math500":
        return _last_boxed(raw) or raw.strip()
    raise ValueError(f"Unsupported dataset: {dataset}")


def _decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _normalize_latex(value: str) -> str:
    normalized = value.strip()
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"\,", "").replace(" ", "")
    normalized = normalized.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    normalized = normalized.rstrip(".")
    return normalized


def _latex_fraction_to_sympy(value: str) -> str:
    previous = None
    converted = value
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while previous != converted:
        previous = converted
        converted = pattern.sub(r"((\1)/(\2))", converted)
    converted = converted.replace("^", "**")
    converted = converted.replace(r"\cdot", "*").replace(r"\times", "*")
    converted = converted.replace(r"\pi", "pi")
    return converted


def answers_equivalent(dataset: str, prediction: str | None, truth: str | None) -> bool:
    if prediction is None or truth is None:
        return False
    predicted_decimal = _decimal(prediction)
    truth_decimal = _decimal(truth)
    if predicted_decimal is not None and truth_decimal is not None:
        return predicted_decimal == truth_decimal

    predicted_normalized = _normalize_latex(prediction)
    truth_normalized = _normalize_latex(truth)
    if predicted_normalized == truth_normalized:
        return True

    try:
        import sympy

        predicted_expr = sympy.sympify(_latex_fraction_to_sympy(predicted_normalized))
        truth_expr = sympy.sympify(_latex_fraction_to_sympy(truth_normalized))
        difference = sympy.simplify(predicted_expr - truth_expr)
        return bool(difference == 0 or math.isclose(float(difference), 0.0, abs_tol=1e-9))
    except Exception:
        return False


def score_record(record: dict[str, Any], dataset: str) -> dict[str, Any]:
    thinking_enabled = bool(
        record.get(
            "generation_enable_thinking",
            record.get("reasoning_complete") is not None,
        )
    )
    reasoning_incomplete = thinking_enabled and not bool(
        record.get("reasoning_complete")
    )
    if reasoning_incomplete:
        # An unclosed thinking segment has no valid final-answer segment,
        # regardless of whether generation ended by length or EOS.
        predicted = None
    else:
        final_text = (
            record.get("final_text", "")
            if thinking_enabled
            else record.get("generated_text", "")
        )
        predicted = (
            extract_python_code(final_text)
            if dataset in CODE_DATASETS
            else extract_answer(dataset, final_text)
        )
    if dataset in CODE_DATASETS:
        updated = dict(record)
        updated["final_answer"] = predicted
        updated["normalized_ground_truth"] = None
        if "code_execution_passed" in record:
            updated["is_correct"] = bool(record["code_execution_passed"])
            updated["evaluation_status"] = "scored_by_external_code_harness"
        else:
            updated["is_correct"] = None
            updated["evaluation_status"] = "pending_external_code_execution"
        updated["answer_extraction_failed"] = predicted is None
        updated["reasoning_incomplete"] = reasoning_incomplete
        return updated
    truth = ground_truth_answer(dataset, str(record["ground_truth"]))
    updated = dict(record)
    updated["final_answer"] = predicted
    updated["normalized_ground_truth"] = truth
    updated["is_correct"] = answers_equivalent(dataset, predicted, truth)
    updated["answer_extraction_failed"] = predicted is None
    updated["reasoning_incomplete"] = reasoning_incomplete
    return updated
