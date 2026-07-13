from __future__ import annotations

from abc import ABC, abstractmethod

from eeg_ui.schemas import DiseaseTask, EEGRecord, PredictionResult


class BaseInferenceService(ABC):
    @abstractmethod
    def predict(self, record: EEGRecord, task: DiseaseTask) -> PredictionResult:
        """Return a standardized binary classification result."""
