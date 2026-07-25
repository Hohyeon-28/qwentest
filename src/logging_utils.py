from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def token_ids_sha256(token_ids: list[int]) -> str:
    raw = ",".join(str(item) for item in token_ids).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {target}:{line_no}") from exc
    return records


def completed_ids(path: str | Path) -> set[str]:
    return {str(row["sample_id"]) for row in read_jsonl(path)}


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def collect_environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "datasets", "accelerate", "vllm")
        },
    }
    try:
        import torch

        environment["torch_cuda_version"] = torch.version.cuda
        environment["cuda_available"] = torch.cuda.is_available()
        environment["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        )
        environment["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        environment["cuda_available"] = False
        environment["gpus"] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        environment["nvidia_driver"] = result.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        environment["nvidia_driver"] = None
    if extra:
        environment.update(extra)
    return environment


def write_environment(path: str | Path, extra: dict[str, Any] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    current = collect_environment(extra)
    if target.exists():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
            previous.update(current)
            current = previous
        except (json.JSONDecodeError, OSError):
            pass
    target.write_text(
        json.dumps(current, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
