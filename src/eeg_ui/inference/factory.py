from __future__ import annotations

from eeg_ui.config import InferenceSettings, ReveSettings
from eeg_ui.inference.base import BaseInferenceService
from eeg_ui.inference.mock_reve import MockReveInferenceService
from eeg_ui.inference.real_reve import RealReveInferenceService


def create_inference_service(
    inference_settings: InferenceSettings,
    reve_settings: ReveSettings | None = None,
) -> BaseInferenceService:
    if inference_settings.backend == "mock_reve":
        return MockReveInferenceService(threshold=inference_settings.threshold)
    if inference_settings.backend == "real_reve":
        if reve_settings is None:
            raise ValueError("REVE settings are required for the real_reve backend.")
        return RealReveInferenceService(
            settings=reve_settings,
            device=inference_settings.device,
            threshold=inference_settings.threshold,
        )
    raise ValueError(
        f"Unknown inference backend '{inference_settings.backend}'. Use 'mock_reve' or 'real_reve'."
    )
