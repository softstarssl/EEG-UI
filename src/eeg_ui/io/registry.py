from __future__ import annotations

from eeg_ui.config import FileSettings
from eeg_ui.io.base import validate_file
from eeg_ui.io.mat_loader import load_mat
from eeg_ui.io.mne_loader import load_with_mne
from eeg_ui.io.numpy_loader import load_numpy
from eeg_ui.schemas import EEGRecord


def load_eeg_record(file_path: str | None, settings: FileSettings) -> EEGRecord:
    validated = validate_file(file_path, settings)
    suffix = validated.path.suffix.lower()
    if suffix in {".edf", ".bdf", ".set", ".fif", ".raw", ".cnt"}:
        record = load_with_mne(validated.path)
    elif suffix == ".mat":
        record = load_mat(validated.path)
    else:
        record = load_numpy(validated.path)

    record.metadata["file_metadata"] = validated.metadata
    return record
