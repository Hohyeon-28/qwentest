from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any


CODE_DATASETS = ("humaneval", "mbpp", "livecodebench")
FENCED_CODE_RE = re.compile(
    r"```(?P<language>[A-Za-z0-9_+.-]*)\s*\n(?P<code>.*?)```",
    flags=re.DOTALL,
)


def extract_python_code(text: str) -> str | None:
    """Extract the executable answer without ever executing model output."""

    candidates = list(FENCED_CODE_RE.finditer(text or ""))
    python_blocks = [
        match.group("code").strip()
        for match in candidates
        if match.group("language").lower() in ("python", "py")
    ]
    generic_blocks = [match.group("code").strip() for match in candidates]
    selected = python_blocks[-1] if python_blocks else generic_blocks[-1] if generic_blocks else ""
    if not selected:
        stripped = (text or "").strip()
        if stripped and ("def " in stripped or "import " in stripped or "class " in stripped):
            selected = stripped
    return selected or None


def humaneval_completion(code: str, prompt: str, entry_point: str) -> str:
    """Convert a full generated function into HumanEval's append-only format."""

    if prompt and code.startswith(prompt):
        return code[len(prompt) :]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
        ),
        None,
    )
    if target is None or not target.body:
        return code
    lines = code.splitlines()
    first_body_line = target.body[0].lineno
    last_body_line = target.end_lineno or target.body[-1].end_lineno
    body = "\n".join(lines[first_body_line - 1 : last_body_line]).rstrip()
    extras = [
        ast.get_source_segment(code, node)
        for node in tree.body
        if node is not target
    ]
    extras = [item for item in extras if item]
    return body + (("\n\n" + "\n\n".join(extras)) if extras else "")


def export_record(record: dict[str, Any], dataset: str) -> dict[str, Any]:
    code = record.get("final_answer") or extract_python_code(
        str(record.get("final_text") or record.get("generated_text") or "")
    )
    metadata = record.get("code_evaluation") or {}
    task_id = str(metadata.get("task_id") or record["sample_id"])
    if dataset == "humaneval":
        completion = code or ""
        prompt = str(metadata.get("prompt") or "")
        completion = humaneval_completion(
            completion, prompt, str(metadata.get("entry_point") or "")
        )
        return {"task_id": task_id, "completion": completion}
    if dataset == "mbpp":
        return {
            "task_id": task_id,
            "completion": code or "",
            "test_setup_code": metadata.get("test_setup_code", ""),
            "test_list": metadata.get("test_list", []),
            "challenge_test_list": metadata.get("challenge_test_list", []),
        }
    if dataset == "livecodebench":
        return {
            "question_title": metadata.get("question_title"),
            "question_content": metadata.get("question_content"),
            "platform": metadata.get("platform"),
            "question_id": task_id,
            "contest_id": metadata.get("contest_id"),
            "contest_date": metadata.get("contest_date"),
            "starter_code": metadata.get("starter_code", ""),
            "difficulty": metadata.get("difficulty"),
            "output_list": [str(record.get("generated_text") or "")],
            "code_list": [code or ""],
        }
    raise ValueError(f"Unsupported code dataset: {dataset}")


def read_external_results(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list: {source}")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def external_pass_map(rows: list[dict[str, Any]], dataset: str) -> dict[str, bool]:
    passed: dict[str, bool] = {}
    for row in rows:
        task_id = row.get("task_id") or row.get("question_id") or row.get("sample_id")
        if task_id is None:
            raise ValueError("External result has no task_id/question_id/sample_id")
        if "passed" in row:
            value = bool(row["passed"])
        elif row.get("graded_list"):
            value = bool(row["graded_list"][0])
        elif "pass@1" in row:
            value = float(row["pass@1"]) >= 1.0
        else:
            raise ValueError(f"External {dataset} result has no pass field: {task_id}")
        passed[str(task_id)] = value
    return passed
