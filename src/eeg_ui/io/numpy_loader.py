from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eeg_ui.schemas import EEGRecord


def _find_sibling(path: Path, suffix: str) -> Path | None:
    """Look for a sibling file by replacing the last suffix pattern.

    E.g. for ``subject_001_eeg.npy`` with ``_ch_pos.npy``, tries:
      1. ``subject_001_eeg_ch_pos.npy``
      2. ``subject_001_ch_pos.npy``
    """
    stem = path.name
    # Try appending suffix to the full stem
    candidate = path.with_name(stem + suffix)
    if candidate.exists():
        return candidate
    # Try replacing trailing _eeg with suffix
    if stem.endswith(".npy"):
        base = stem[:-4]
        for pattern in ("_eeg", "_epochs", "_signal"):
            if base.endswith(pattern):
                candidate = path.with_name(base[: -len(pattern)] + suffix)
                if candidate.exists():
                    return candidate
    return None


def load_numpy(path: Path) -> EEGRecord:
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            shape = tuple(loaded.shape)

            # --- Determine signal semantics from shape ---
            if loaded.ndim == 3:
                # (n_epochs, n_channels, n_times) — preprocessed REVE input
                signal = loaded.astype(np.float32)
                signal_shape = shape
                message = (
                    f"3D preprocessed REVE input: {shape[0]} epochs × "
                    f"{shape[1]} channels × {shape[2]} time points."
                )
            elif loaded.ndim == 2:
                # Could be (n_channels, n_times) — single epoch
                signal = loaded.astype(np.float32)
                signal_shape = shape
                message = (
                    f"2D array ({shape[0]}×{shape[1]}). "
                    "Treated as (channels, time); EEG semantics are not verified."
                )
            else:
                signal = None
                signal_shape = shape
                message = (
                    f"{loaded.ndim}D array is not a recognised EEG layout. "
                    "Expected 2D (channels, time) or 3D (epochs, channels, time)."
                )

            # --- Look for electrode positions ---
            coordinates = None
            channel_names: list[str] = []
            pos_path = _find_sibling(path, "_ch_pos.npy")
            if pos_path is not None:
                try:
                    pos = np.load(pos_path, allow_pickle=False)
                    coordinates = pos.astype(np.float32)
                except Exception:
                    pass

            # --- Look for metadata JSON ---
            meta_path = _find_sibling(path, "_meta.json")
            if meta_path is not None:
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        sidecar = json.load(fh)
                    channel_names = sidecar.get("channels", [])
                except Exception:
                    pass

            # --- Sampling rate ---
            sampling_rate = None
            if channel_names:
                # We have channel info, assume 200 Hz (REVE default)
                sampling_rate = 200.0

            return EEGRecord(
                file_path=path,
                file_type=".npy",
                signal=signal,
                sampling_rate=sampling_rate,
                channel_names=channel_names,
                coordinates=coordinates,
                metadata={
                    "signal_shape": signal_shape,
                    "read_message": message,
                },
            )

        # .npz archive
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
            metadata={
                "read_message": f"Could not read NumPy metadata: {error}. Mock mode can still run."
            },
        )
