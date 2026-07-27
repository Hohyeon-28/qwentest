from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, public_config, results_root
from src.data import load_samples
from src.logging_utils import read_jsonl


CONDITIONS = ("bf16", "fake_quant", "real_quant_marlin")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    root = results_root(config, dataset)
    errors: list[str] = []
    expected_samples = load_samples(config, dataset)
    expected_ids = {str(row["sample_id"]) for row in expected_samples}
    records = {
        condition: read_jsonl(root / condition / "predictions.jsonl")
        for condition in CONDITIONS
    }

    condition_report: dict[str, Any] = {}
    for condition, rows in records.items():
        ids = [str(row.get("sample_id")) for row in rows]
        duplicate_ids = sorted(
            sample_id
            for sample_id, count in Counter(ids).items()
            if count > 1
        )
        missing = sorted(expected_ids - set(ids))
        unexpected = sorted(set(ids) - expected_ids)
        censored = sum(bool(row.get("reasoning_length_censored")) for row in rows)
        incomplete = sum(bool(row.get("reasoning_incomplete")) for row in rows)
        capped = sum(bool(row.get("hit_max_new_tokens")) for row in rows)
        fraction = censored / len(rows) if rows else 1.0
        incomplete_fraction = incomplete / len(rows) if rows else 1.0
        condition_report[condition] = {
            "samples": len(rows),
            "unique_sample_ids": len(set(ids)),
            "duplicate_sample_ids": duplicate_ids,
            "missing_sample_ids": missing,
            "unexpected_sample_ids": unexpected,
            "max_token_truncations": capped,
            "reasoning_length_censored": censored,
            "reasoning_length_censored_fraction": fraction,
            "reasoning_incomplete": incomplete,
            "reasoning_incomplete_fraction": incomplete_fraction,
        }
        if len(rows) != len(expected_samples):
            _fail(
                errors,
                f"{condition}: expected {len(expected_samples)} samples, found {len(rows)}",
            )
        if duplicate_ids:
            _fail(errors, f"{condition}: duplicate sample IDs: {duplicate_ids[:10]}")
        if missing or unexpected:
            _fail(
                errors,
                f"{condition}: sample ID set mismatch "
                f"(missing={missing[:10]}, unexpected={unexpected[:10]})",
            )
        configured_max = int(config["generation"]["max_new_tokens"])
        protocol_values = {
            (
                row.get("generation_max_new_tokens"),
                row.get("generation_deterministic"),
                row.get("generation_seed"),
                row.get("generation_enable_thinking"),
                tuple(row.get("generation_stop_token_ids") or []),
            )
            for row in rows
        }
        expected_protocol = {
            (
                configured_max,
                bool(config["generation"]["deterministic"]),
                int(config["generation"]["seed"]),
                bool(config["generation"]["enable_thinking"]),
                tuple(
                    dict.fromkeys(
                        int(item)
                        for item in config["generation"].get("stop_token_ids", [])
                    )
                ),
            )
        }
        if protocol_values != expected_protocol:
            _fail(
                errors,
                f"{condition}: mixed or unexpected generation protocol: "
                f"{sorted(protocol_values, key=str)}",
            )
        over_limit = [
            row["sample_id"]
            for row in rows
            if int(row.get("generated_token_count") or 0) > configured_max
        ]
        if over_limit:
            _fail(
                errors,
                f"{condition}: generated_token_count exceeded {configured_max}: "
                f"{over_limit[:10]}",
            )
        allowed_fraction = float(
            config["evaluation"].get("max_reasoning_censored_fraction", 0.0)
        )
        allowed_incomplete_fraction = float(
            config["evaluation"].get(
                "max_reasoning_incomplete_fraction", allowed_fraction
            )
        )
        if fraction > allowed_fraction:
            _fail(
                errors,
                f"{condition}: reasoning censoring {fraction:.2%} exceeds "
                f"configured maximum {allowed_fraction:.2%}",
            )
        if incomplete_fraction > allowed_incomplete_fraction:
            _fail(
                errors,
                f"{condition}: incomplete reasoning {incomplete_fraction:.2%} "
                "exceeds configured maximum "
                f"{allowed_incomplete_fraction:.2%}",
            )

        saved_config_path = root / condition / "config.json"
        if not saved_config_path.exists():
            _fail(errors, f"{condition}: missing effective config {saved_config_path}")
        else:
            saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
            if saved_config != public_config(config):
                _fail(errors, f"{condition}: effective config does not match current config")

    maps = {
        condition: {str(row["sample_id"]): row for row in rows}
        for condition, rows in records.items()
    }
    common_ids = set.intersection(*(set(mapping) for mapping in maps.values()))
    for sample_id in sorted(common_ids):
        aligned = [maps[condition][sample_id] for condition in CONDITIONS]
        if len({row.get("input_token_ids_sha256") for row in aligned}) != 1:
            _fail(errors, f"{sample_id}: prompt token hash mismatch")
        if len({row.get("question") for row in aligned}) != 1:
            _fail(errors, f"{sample_id}: question mismatch")
        if len({row.get("ground_truth") for row in aligned}) != 1:
            _fail(errors, f"{sample_id}: ground-truth mismatch")

    fake_rows = records["fake_quant"]
    real_rows = records["real_quant_marlin"]
    fake_tuple_hashes = {
        row.get("quantization_tuple_sha256") for row in fake_rows
    } - {None}
    real_tuple_hashes = {
        row.get("quantization_tuple_sha256") for row in real_rows
    } - {None}
    if len(fake_tuple_hashes) != 1 or fake_tuple_hashes != real_tuple_hashes:
        _fail(
            errors,
            "Fake/Real per-sample (q,s,z,g) fingerprints are missing or different",
        )
    source_checkpoints = {
        row.get("quantization_source_checkpoint")
        for row in [*fake_rows, *real_rows]
    } - {None}
    if source_checkpoints != {config["models"]["real_gptq"]}:
        _fail(
            errors,
            f"Fake/Real source checkpoint mismatch: {sorted(source_checkpoints)}",
        )

    selected_backends = {
        str(row.get("selected_quantization_backend"))
        for row in real_rows
    } - {"None"}
    if not selected_backends or any(
        "gptq_marlin" not in backend for backend in selected_backends
    ):
        _fail(errors, f"Real backend is not GPTQ-Marlin: {sorted(selected_backends)}")

    fake_manifest = root / "fake_quant" / "quantization_manifest.json"
    real_manifest = root / "real_quant_marlin" / "quantization_manifest.json"
    manifest_report: dict[str, Any] = {}
    if not fake_manifest.exists() or not real_manifest.exists():
        _fail(errors, "Missing Fake or Real quantization manifest")
    else:
        fake_sha = _sha256(fake_manifest)
        real_sha = _sha256(real_manifest)
        manifest_report = {
            "fake_manifest_sha256": fake_sha,
            "real_manifest_sha256": real_sha,
            "byte_identical": fake_sha == real_sha,
        }
        if fake_sha != real_sha:
            _fail(errors, "Fake and Real quantization manifests are not byte-identical")

    max_input_tokens = max(
        (
            int(row.get("input_token_count") or 0)
            for rows in records.values()
            for row in rows
        ),
        default=0,
    )
    total_budget = max_input_tokens + int(config["generation"]["max_new_tokens"])
    real_context = int(config["quantization"]["real"]["max_model_len"])
    if total_budget > real_context:
        _fail(
            errors,
            f"prompt+generation budget {total_budget} exceeds Real max_model_len "
            f"{real_context}",
        )

    return {
        "valid": not errors,
        "dataset": dataset,
        "expected_samples": len(expected_samples),
        "conditions": condition_report,
        "alignment_samples_checked": len(common_ids),
        "shared_quantization_tuple_sha256": (
            next(iter(fake_tuple_hashes))
            if len(fake_tuple_hashes) == 1 and fake_tuple_hashes == real_tuple_hashes
            else None
        ),
        "quantization_manifests": manifest_report,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": int(config["generation"]["max_new_tokens"]),
        "real_max_model_len": real_context,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation for completed experiment outputs"
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset", choices=("gsm8k", "math500"), default="gsm8k")
    args = parser.parse_args()
    config = load_config(args.config)
    report = validate(config, args.dataset)
    target = results_root(config, args.dataset) / "validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
