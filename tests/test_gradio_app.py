from __future__ import annotations

from pathlib import Path

import pytest

from eeg_ui.config import load_config
from eeg_ui.ui.gradio_app import create_app, run_prediction


def test_prediction_callback_reports_mock_result(tmp_path: Path) -> None:
    file_path = tmp_path / "subject.npy"
    file_path.write_bytes(b"demo-eeg")
    settings = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")

    logs, result, file_info = run_prediction(str(file_path), "depression", settings)

    assert "Done." in logs
    assert "Mock mode" in result
    assert "Prediction" in result
    assert "subject.npy" in file_info


def test_gradio_application_can_be_created() -> None:
    pytest.importorskip("gradio")
    settings = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")

    app = create_app(settings)

    assert app is not None
