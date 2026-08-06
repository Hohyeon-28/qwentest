import sys
from types import SimpleNamespace

from src.data import load_samples


class FakeDataset(list):
    def select(self, indices):
        return FakeDataset(self[index] for index in indices)


def _with_fake_datasets(rows, callback):
    observed = {}

    def load_dataset(**kwargs):
        observed.update(kwargs)
        return FakeDataset(rows)

    previous = sys.modules.get("datasets")
    sys.modules["datasets"] = SimpleNamespace(load_dataset=load_dataset)
    try:
        result = callback()
    finally:
        if previous is None:
            del sys.modules["datasets"]
        else:
            sys.modules["datasets"] = previous
    return result, observed


def test_mbpp_sanitized_schema_uses_prompt_and_test_imports():
    config = {
        "datasets": {
            "mbpp": {
                "path": "google-research-datasets/mbpp",
                "name": "sanitized",
                "split": "test",
                "max_samples": None,
            }
        }
    }
    rows = [
        {
            "task_id": 11,
            "prompt": "Write f.",
            "code": "def f(): return 1",
            "test_imports": ["import math"],
            "test_list": ["assert f() == 1"],
        }
    ]
    samples, observed = _with_fake_datasets(
        rows, lambda: load_samples(config, "mbpp")
    )
    assert observed["name"] == "sanitized"
    assert samples[0]["sample_id"] == "11"
    assert samples[0]["question"].startswith("Write f.")
    assert samples[0]["code_evaluation"]["test_setup_code"] == "import math"


def test_livecodebench_release_uses_materialized_official_json_files():
    config = {
        "datasets": {
            "livecodebench": {
                "path": "livecodebench/code_generation_lite",
                "name": None,
                "split": "test",
                "max_samples": None,
                "version_tag": "release_v6",
                "trust_remote_code": True,
            }
        }
    }
    rows = [
        {
            "question_id": "abc",
            "question_title": "A",
            "question_content": "Solve A",
            "platform": "codeforces",
            "contest_id": "1",
            "contest_date": "2025-01-01",
            "starter_code": "",
            "difficulty": "easy",
        }
    ]
    samples, observed = _with_fake_datasets(
        rows, lambda: load_samples(config, "livecodebench")
    )
    assert observed["path"] == "json"
    assert observed["split"] == "test"
    assert observed["data_files"]["test"] == [
        "hf://datasets/livecodebench/code_generation_lite/test.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test2.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test3.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test4.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test5.jsonl",
        "hf://datasets/livecodebench/code_generation_lite/test6.jsonl",
    ]
    assert "version_tag" not in observed
    assert "trust_remote_code" not in observed
    assert samples[0]["sample_id"] == "abc"
