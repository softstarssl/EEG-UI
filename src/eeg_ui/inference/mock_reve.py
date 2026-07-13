from __future__ import annotations

import hashlib

from eeg_ui.inference.base import BaseInferenceService
from eeg_ui.schemas import DiseaseTask, EEGRecord, PredictionResult


class MockReveInferenceService(BaseInferenceService):
    """Deterministic stand-in for a future REVE classifier."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def predict(self, record: EEGRecord, task: DiseaseTask) -> PredictionResult:
        try:
            size = record.file_path.stat().st_size
        except OSError:
            size = 0
        seed = f"{record.file_path.name}|{size}|{task.value}".encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        probability = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)
        label = int(probability >= self.threshold)
        return PredictionResult(
            task=task,
            label=label,
            positive_probability=probability,
            threshold=self.threshold,
            decision=task.decision_text(label),
            backend="mock_reve",
            is_mock=True,
            message="Mock result. No real model is connected.",
        )
