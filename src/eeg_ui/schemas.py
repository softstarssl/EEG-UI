from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class DiseaseTask(str, Enum):
    DEPRESSION = "depression"
    ADHD = "adhd"

    @property
    def display_name(self) -> str:
        return "Depression" if self is DiseaseTask.DEPRESSION else "ADHD"

    def decision_text(self, label: int) -> str:
        suffix = "Positive" if label else "Negative"
        return f"{self.display_name} {suffix}"


@dataclass
class EEGRecord:
    file_path: Path
    file_type: str
    signal: np.ndarray | None = None
    sampling_rate: float | None = None
    channel_names: list[str] = field(default_factory=list)
    coordinates: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FileMetadata:
    file_name: str
    file_type: str
    size_bytes: int
    extension_valid: bool
    read_message: str | None = None


@dataclass(frozen=True)
class PredictionResult:
    task: DiseaseTask
    label: int
    positive_probability: float
    threshold: float
    decision: str
    backend: str
    is_mock: bool
    message: str
