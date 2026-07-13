from __future__ import annotations

from pathlib import Path

import numpy as np

from eeg_ui.schemas import EEGRecord


def load_numpy(path: Path) -> EEGRecord:
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            signal = loaded if loaded.ndim == 2 else None
            message = (
                "A two-dimensional NumPy array was read. Its EEG semantics and sampling rate are not verified."
                if signal is not None
                else "The NumPy array is not two-dimensional, so an EEG channel-by-time signal could not be inferred."
            )
            return EEGRecord(
                file_path=path,
                file_type=".npy",
                signal=signal,
                metadata={"signal_shape": tuple(loaded.shape), "read_message": message},
            )

        archive = loaded
        try:
            shapes = {name: tuple(archive[name].shape) for name in archive.files}
        finally:
            archive.close()
        return EEGRecord(
            file_path=path,
            file_type=".npz",
            metadata={
                "npz_arrays": shapes,
                "read_message": "NumPy archive contents were listed, but the EEG array and sampling rate are unknown.",
            },
        )
    except Exception as error:
        return EEGRecord(
            file_path=path,
            file_type=path.suffix.lower(),
            metadata={"read_message": f"Could not read NumPy metadata: {error}. Mock mode can still run."},
        )
