from __future__ import annotations

import pytest

from eeg_ui.config import InferenceSettings
from eeg_ui.inference.factory import create_inference_service
from eeg_ui.inference.mock_reve import MockReveInferenceService
from eeg_ui.inference.real_reve import RealReveInferenceService


def test_mock_backend_uses_mock_service() -> None:
    service = create_inference_service(InferenceSettings(backend="mock_reve"))

    assert isinstance(service, MockReveInferenceService)


def test_real_backend_uses_placeholder_service() -> None:
    service = create_inference_service(InferenceSettings(backend="real_reve"))

    assert isinstance(service, RealReveInferenceService)


def test_unknown_backend_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown inference backend"):
        create_inference_service(InferenceSettings(backend="not-a-backend"))
