from __future__ import annotations

from eeg_ui.inference.base import BaseInferenceService
from eeg_ui.schemas import DiseaseTask, EEGRecord, PredictionResult


class RealReveUnavailableError(RuntimeError):
    """Raised until a real REVE model and classifier are supplied."""


class RealReveInferenceService(BaseInferenceService):
    def predict(self, record: EEGRecord, task: DiseaseTask) -> PredictionResult:
        raise RealReveUnavailableError(
            "Real REVE backend is not available. Please provide a model checkpoint and classifier configuration."
        )
