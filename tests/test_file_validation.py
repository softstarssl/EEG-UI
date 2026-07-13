from __future__ import annotations

from pathlib import Path

import pytest

from eeg_ui.config import FileSettings
from eeg_ui.io.base import FileValidationError, validate_file


def test_supported_extension_and_uppercase_pass(tmp_path: Path) -> None:
    file_path = tmp_path / "record.NPY"
    file_path.write_bytes(b"not-empty")

    validated = validate_file(file_path, FileSettings())

    assert validated.metadata.extension_valid is True
    assert validated.metadata.file_type == ".npy"


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "record.txt"
    file_path.write_text("data", encoding="utf-8")

    with pytest.raises(FileValidationError, match="Unsupported"):
        validate_file(file_path, FileSettings())


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "record.edf"
    file_path.touch()

    with pytest.raises(FileValidationError, match="empty"):
        validate_file(file_path, FileSettings())
