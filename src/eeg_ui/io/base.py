from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eeg_ui.config import FileSettings
from eeg_ui.schemas import FileMetadata


class FileValidationError(ValueError):
    """Raised when an uploaded file cannot be used by the demo."""


@dataclass(frozen=True)
class ValidatedFile:
    path: Path
    metadata: FileMetadata


def validate_file(file_path: str | Path | None, settings: FileSettings) -> ValidatedFile:
    if not file_path:
        raise FileValidationError("Upload an EEG file before running prediction.")

    path = Path(file_path)
    extension = path.suffix.lower()
    if extension not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise FileValidationError(f"Unsupported file type '{extension or 'unknown'}'. Allowed types: {allowed}.")
    if not path.is_file():
        raise FileValidationError("The uploaded file is no longer available. Please upload it again.")

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise FileValidationError("The uploaded file is empty. Choose a non-empty EEG file.")
    if size_bytes > settings.max_size_mb * 1024 * 1024:
        raise FileValidationError(
            f"The file is larger than the {settings.max_size_mb} MB upload limit."
        )

    return ValidatedFile(
        path=path,
        metadata=FileMetadata(
            file_name=path.name,
            file_type=extension,
            size_bytes=size_bytes,
            extension_valid=True,
        ),
    )
