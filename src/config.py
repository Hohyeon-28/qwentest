from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CHOICES = ("gsm8k", "math500", "humaneval", "mbpp", "livecodebench")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def results_root(config: dict[str, Any], dataset: str | None = None) -> Path:
    raw = Path(config["experiment"]["output_dir"])
    root = raw if raw.is_absolute() else PROJECT_ROOT / raw
    return root / dataset if dataset else root


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")


def save_effective_config(
    config: dict[str, Any], dataset: str, condition: str | None = None
) -> Path:
    output = results_root(config, dataset)
    if condition is not None:
        output = output / condition
    output.mkdir(parents=True, exist_ok=True)
    target = output / "config.json"
    target.write_text(
        json.dumps(public_config(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
