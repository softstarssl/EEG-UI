from __future__ import annotations

from pathlib import Path

import numpy as np

from eeg_ui.schemas import EEGRecord

# Keys mirror the inference server's auto-detection (deployment/preprocess.py
# ::_read_mat_eeg) so the metadata panel reports the same values the server
# will actually use.
_DATA_KEYS = ("data", "eeg", "X", "signal", "signals", "epochs", "raw", "record")
_FS_KEYS = ("fs", "srate", "sample_rate", "Fs", "sfreq", "sampling_frequency")
_CH_KEYS = (
    "ch_names",
    "channel_names",
    "labels",
    "chanlabels",
    "channels",
    "channel_labels",
    "electrode_names",
)


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
        try:
            contents = loadmat(path, simplify_cells=True)
        except NotImplementedError:
            # MATLAB v7.3 uses HDF5 — fall back to h5py
            import h5py

            contents = {}
            with h5py.File(str(path), "r") as f:
                for key in f.keys():
                    arr = np.array(f[key])
                    if arr.ndim == 2 and arr.shape[0] == 1:
                        arr = arr[0]
                    contents[key] = arr

        variable_names = sorted(name for name in contents if not name.startswith("__"))
        shape, sampling_rate, channel_names, data_key = _detect_schema(contents)

        if data_key is None:
            read_message = (
                "MATLAB variables were found, but no 2D EEG data matrix could be detected."
            )
        elif sampling_rate is None:
            read_message = (
                f"Detected EEG data in '{data_key}'. This file has no sampling-rate "
                "field, so the value entered in the 'Sampling rate (Hz)' field will be "
                "used (defaults to 200 Hz if left unchanged)."
            )
        else:
            read_message = f"Detected EEG data in '{data_key}'."

        return EEGRecord(
            file_path=path,
            file_type=".mat",
            sampling_rate=sampling_rate,
            channel_names=channel_names,
            metadata={
                "mat_variables": variable_names,
                "signal_shape": shape,
                "read_message": read_message,
            },
        )
    except Exception as error:
        return EEGRecord(
            file_path=path,
            file_type=".mat",
            metadata={"read_message": f"Could not read MATLAB metadata: {error}. Mock mode can still run."},
        )


def _detect_schema(contents: dict) -> tuple[tuple | None, float | None, list[str], str | None]:
    """Extract (shape, sampling_rate, channel_names, data_key) from a loaded .mat."""
    data = None
    data_key = None
    for key in _DATA_KEYS:
        val = contents.get(key)
        if isinstance(val, np.ndarray) and val.ndim >= 2:
            data, data_key = val, key
            break

    if data is None:
        arrays = [
            (k, v)
            for k, v in contents.items()
            if isinstance(v, np.ndarray) and v.ndim >= 2 and not k.startswith("_")
        ]
        if arrays:
            data_key, data = max(arrays, key=lambda item: item[1].size)

    if data is None:
        return None, None, [], None

    # Match the server's orientation heuristic: channels × samples.
    if data.ndim > 2:
        data = data.squeeze()
    if data.ndim == 2:
        if data.shape[0] > data.shape[1]:
            data = data.T
    else:
        return tuple(data.shape), None, [], data_key

    shape = tuple(data.shape)

    sampling_rate = None
    for key in _FS_KEYS:
        if key in contents:
            val = contents[key]
            if isinstance(val, np.ndarray):
                val = val.flat[0]
            sampling_rate = float(val)
            break

    channel_names: list[str] = []
    n_channels = shape[0] if shape else 0
    for key in _CH_KEYS:
        val = contents.get(key)
        if isinstance(val, np.ndarray) and val.dtype.kind in ("U", "S", "O"):
            names = [str(v).strip() for v in val.ravel()]
            if len(names) == n_channels:
                channel_names = names
                break

    return shape, sampling_rate, channel_names, data_key
