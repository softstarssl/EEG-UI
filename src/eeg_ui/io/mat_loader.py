from __future__ import annotations

from pathlib import Path

from eeg_ui.schemas import EEGRecord


def load_mat(path: Path) -> EEGRecord:
    try:
        from scipy.io import loadmat
    except ImportError:
        return EEGRecord(
            file_path=path,
            file_type=".mat",
            metadata={
                "read_message": "SciPy is not installed, so MATLAB metadata could not be read. Mock mode can still run."
            },
        )

    try:
        contents = loadmat(path, variable_names=None)
        variable_names = sorted(name for name in contents if not name.startswith("__"))
        return EEGRecord(
            file_path=path,
            file_type=".mat",
            metadata={
                "mat_variables": variable_names,
                "read_message": (
                    "MATLAB variables were found, but this demo cannot infer an EEG schema or sampling rate "
                    "without a dataset-specific loader."
                ),
            },
        )
    except Exception as error:
        return EEGRecord(
            file_path=path,
            file_type=".mat",
            metadata={"read_message": f"Could not read MATLAB metadata: {error}. Mock mode can still run."},
        )
