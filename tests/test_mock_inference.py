from __future__ import annotations

from pathlib import Path

from eeg_ui.inference.mock_reve import MockReveInferenceService
from eeg_ui.schemas import DiseaseTask, EEGRecord


def test_mock_result_is_complete_and_stable(tmp_path: Path) -> None:
    file_path = tmp_path / "subject.npy"
    file_path.write_bytes(b"demo-eeg")
    record = EEGRecord(file_path=file_path, file_type=".npy")
    service = MockReveInferenceService(threshold=0.5)

    first = service.predict(record, DiseaseTask.DEPRESSION)
    second = service.predict(record, DiseaseTask.DEPRESSION)

    assert first == second
    assert 0.0 <= first.positive_probability <= 1.0
    assert first.label in {0, 1}
    assert first.label == int(first.positive_probability >= first.threshold)
    assert first.is_mock is True
    assert first.decision in {"Depression Negative", "Depression Positive"}


def test_tasks_use_different_mock_seeds(tmp_path: Path) -> None:
    file_path = tmp_path / "subject.npy"
    file_path.write_bytes(b"demo-eeg")
    record = EEGRecord(file_path=file_path, file_type=".npy")
    service = MockReveInferenceService(threshold=0.5)

    depression = service.predict(record, DiseaseTask.DEPRESSION)
    adhd = service.predict(record, DiseaseTask.ADHD)

    assert depression.positive_probability != adhd.positive_probability
