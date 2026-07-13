from __future__ import annotations

from eeg_ui.schemas import EEGRecord


def validate_reve_input(record: EEGRecord) -> list[str]:
    """Report minimum future-REVE input gaps without altering the EEG signal."""
    issues: list[str] = []
    if record.signal is None:
        issues.append("Signal data is unavailable.")
    elif record.signal.ndim != 2:
        issues.append("Signal must have shape [channels, time].")
    if record.sampling_rate is None:
        issues.append("Sampling rate is unavailable.")
    if not record.channel_names:
        issues.append("Channel names are unavailable.")
    if record.coordinates is None:
        issues.append("Electrode coordinates are unavailable.")
    return issues
