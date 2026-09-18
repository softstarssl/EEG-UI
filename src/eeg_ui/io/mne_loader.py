from __future__ import annotations

from pathlib import Path

from eeg_ui.schemas import EEGRecord


def load_with_mne(path: Path) -> EEGRecord:
    try:
        import mne
    except ImportError:
        return EEGRecord(
            file_path=path,
            file_type=path.suffix.lower(),
            metadata={
                "read_message": "MNE is not installed, so EEG metadata could not be read. Mock mode can still run."
            },
        )

    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
        ".raw": mne.io.read_raw_egi,
        ".cnt": mne.io.read_raw_cnt,
    }
    try:
        raw = readers[path.suffix.lower()](path, preload=False, verbose="ERROR")
        return EEGRecord(
            file_path=path,
            file_type=path.suffix.lower(),
            sampling_rate=float(raw.info["sfreq"]),
            channel_names=list(raw.ch_names),
            metadata={
                "signal_shape": (len(raw.ch_names), int(raw.n_times)),
                "read_message": "Basic EEG metadata was read with MNE.",
            },
        )
    except Exception as error:  # Parsing must not prevent a mock demonstration.
        return EEGRecord(
            file_path=path,
            file_type=path.suffix.lower(),
            metadata={
                "read_message": f"Could not read EEG metadata: {error}. Mock mode can still run."
            },
        )
