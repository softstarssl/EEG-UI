from __future__ import annotations

from eeg_ui.config import InferenceSettings
from eeg_ui.inference.base import BaseInferenceService
from eeg_ui.inference.mock_reve import MockReveInferenceService
from eeg_ui.inference.real_reve import RealReveInferenceService


def create_inference_service(settings: InferenceSettings) -> BaseInferenceService:
    if settings.backend == "mock_reve":
        return MockReveInferenceService(threshold=settings.threshold)
    if settings.backend == "real_reve":
        return RealReveInferenceService()
    raise ValueError(
        f"Unknown inference backend '{settings.backend}'. Use 'mock_reve' or 'real_reve'."
    )
