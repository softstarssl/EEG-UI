from __future__ import annotations

from pathlib import Path

import pytest

from eeg_ui.config import ConfigError, load_config


def test_default_config_loads() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    settings = load_config(config_path)

    assert settings.inference.backend == "http_reve"
    assert settings.inference.threshold == 0.5
    assert ".edf" in settings.files.allowed_extensions
    assert settings.server.port == 8765
    assert settings.server.base_url == "http://127.0.0.1:8765"


def test_missing_optional_sections_use_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("inference:\n  backend: mock_reve\n", encoding="utf-8")

    settings = load_config(path)

    assert settings.app.port == 7860
    assert settings.files.max_size_mb == 1024
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8765


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_invalid_threshold_is_rejected(tmp_path: Path, threshold: float) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"inference:\n  threshold: {threshold}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="between 0 and 1"):
        load_config(path)
