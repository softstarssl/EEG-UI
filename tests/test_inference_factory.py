from __future__ import annotations

import pytest

from eeg_ui.config import InferenceSettings, ReveSettings
from eeg_ui.inference.factory import create_inference_service
from eeg_ui.inference.mock_reve import MockReveInferenceService


def test_mock_backend_uses_mock_service() -> None:
    service = create_inference_service(InferenceSettings(backend="mock_reve"))

    assert isinstance(service, MockReveInferenceService)


def test_real_backend_without_torch_reports_clear_error() -> None:
    """Factory returns a real service, but predict fails if torch is absent."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch is not installed — real_reve backend requires it.")

    try:
        from braindecode.models import REVE  # noqa: F401
    except ImportError:
        pytest.skip("braindecode is not installed — real_reve backend requires it.")

    from eeg_ui.inference.real_reve import RealReveInferenceService

    service = create_inference_service(
        InferenceSettings(backend="real_reve"),
        ReveSettings(),
    )
    assert isinstance(service, RealReveInferenceService)


def test_unknown_backend_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown inference backend"):
        create_inference_service(InferenceSettings(backend="not-a-backend"))
