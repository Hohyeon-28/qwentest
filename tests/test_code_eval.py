from src.answer_parser import score_record
from src.code_eval import (
    export_record,
    extract_python_code,
    external_pass_map,
    humaneval_completion,
)
from src.metrics import summarize_predictions


def test_extracts_last_python_block_from_final_answer():
    text = "explanation\n```python\ndef first(): pass\n```\n```python\ndef answer():\n    return 1\n```"
    assert extract_python_code(text) == "def answer():\n    return 1"


def test_code_record_remains_unscored_until_harness_result_is_imported():
    record = {
        "sample_id": "HumanEval/0",
        "ground_truth": "",
        "final_text": "```python\ndef f():\n    return 1\n```",
        "generated_text": "",
        "reasoning_complete": True,
        "generation_enable_thinking": True,
    }
    pending = score_record(record, "humaneval")
    assert pending["is_correct"] is None
    assert pending["evaluation_status"] == "pending_external_code_execution"
    scored = score_record({**pending, "code_execution_passed": True}, "humaneval")
    assert scored["is_correct"] is True


def test_humaneval_export_uses_official_task_and_completion_fields():
    row = {
        "sample_id": "HumanEval/0",
        "final_answer": "def f():\n    return 1",
        "code_evaluation": {"task_id": "HumanEval/0", "entry_point": "f"},
    }
    assert export_record(row, "humaneval") == {
        "task_id": "HumanEval/0",
        "completion": "    return 1",
    }


def test_humaneval_full_solution_preserves_helpers_in_append_only_completion():
    code = "import math\n\ndef f(x):\n    return helper(x)\n\ndef helper(x):\n    return math.ceil(x)"
    completion = humaneval_completion(code, "def f(x):\n    \"\"\"doc\"\"\"\n", "f")
    assert completion.startswith("    return helper(x)")
    assert "import math" in completion
    assert "def helper" in completion


def test_external_result_formats_are_normalized():
    assert external_pass_map([{"task_id": "x", "passed": True}], "humaneval") == {
        "x": True
    }
    assert external_pass_map(
        [{"question_id": "y", "graded_list": [False]}], "livecodebench"
    ) == {"y": False}


def test_summary_does_not_count_pending_code_as_incorrect():
    records = [
        {
            "is_correct": None,
            "generated_token_count": 10,
            "reasoning_token_count": 5,
            "reasoning_complete": True,
            "batch_size_used": 1,
            "generation_batch_id": "a",
            "batch_generation_latency_seconds": 2.0,
            "total_generation_latency_seconds": 2.0,
        }
    ]
    summary = summarize_predictions(records)
    assert summary["accuracy"] is None
    assert summary["pending_code_evaluation"] == 1
    assert summary["aggregate_generation_tokens_per_second"] == 5.0
