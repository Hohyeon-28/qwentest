from src.answer_parser import (
    answers_equivalent,
    extract_gsm8k_answer,
    extract_math_answer,
    ground_truth_answer,
    score_record,
)
from src.data import count_gsm8k_gold_calculation_steps


def test_gsm8k_uses_last_number_and_ground_truth_marker():
    assert extract_gsm8k_answer("First 10, then the answer is 1,234.5") == "1234.5"
    assert ground_truth_answer("gsm8k", "work\n#### -42") == "-42"


def test_math_boxed_nested_braces():
    assert extract_math_answer(r"Thus \boxed{\frac{1}{2}}.") == r"\frac{1}{2}"
    assert answers_equivalent("math500", r"\frac{1}{2}", "0.5")


def test_missing_answer_is_incorrect():
    assert not answers_equivalent("gsm8k", None, "1")


def test_censored_reasoning_is_not_scored_from_an_incidental_number():
    record = {
        "ground_truth": r"\boxed{42}",
        "generated_text": "<think>unfinished calculation ending in 42",
        "final_text": "<think>unfinished calculation ending in 42",
        "reasoning_length_censored": True,
    }
    scored = score_record(record, "math500")
    assert scored["final_answer"] is None
    assert not scored["is_correct"]
    assert scored["answer_extraction_failed"]


def test_gsm8k_gold_calculation_steps_count_annotations_only():
    solution = (
        "She buys 3 packs at $4 each. <<3*4=12>>\n"
        "Then adds $2. <<12+2=14>>\n#### 14"
    )
    assert count_gsm8k_gold_calculation_steps(solution) == 2
