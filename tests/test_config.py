from pathlib import Path

from src.config import results_root


def test_results_root_uses_external_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EXPERIMENT_OUTPUT_ROOT", str(tmp_path))
    config = {"experiment": {"output_dir": "results_cuda130_39k_v1"}}

    assert results_root(config, "gsm8k") == (
        tmp_path / "results_cuda130_39k_v1" / "gsm8k"
    )


def test_absolute_output_path_overrides_external_root(
    monkeypatch, tmp_path: Path
) -> None:
    external = tmp_path / "external"
    absolute = tmp_path / "explicit"
    monkeypatch.setenv("EXPERIMENT_OUTPUT_ROOT", str(external))
    config = {"experiment": {"output_dir": str(absolute)}}

    assert results_root(config) == absolute
