from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests

from eeg_ui.config import ServerSettings
from eeg_ui.inference.base import BaseInferenceService
from eeg_ui.schemas import DiseaseTask, EEGRecord, PredictionResult

LOGGER = logging.getLogger(__name__)

_TASK_MAP = {DiseaseTask.DEPRESSION: "depression", DiseaseTask.ADHD: "adhd"}


class HttpReveUnavailableError(RuntimeError):
    """Raised when the remote inference server cannot be reached or used."""


class HttpReveInferenceService(BaseInferenceService):
    """Delegate inference to the deployment server over HTTP.

    Sends the raw EEG file to ``POST /api/predict/raw``.  The server runs the
    full pipeline — resample → bandpass → notch → z-score → clip → 10s windows —
    and returns a subject-level aggregated prediction from the latest
    MDD/ADHD eye-state + LoRA models.
    """

    def __init__(self, server: ServerSettings, threshold: float) -> None:
        self._server = server
        self._threshold = threshold

    def predict(
        self,
        record: EEGRecord,
        task: DiseaseTask,
        pos_path: str | None = None,
        fs: float | None = None,
        # age_group: str | None = None,
        # sex: str | None = None,
    ) -> PredictionResult:
        """Run inference, optionally forwarding an electrode-position file.

        ``pos_path`` points to a ``.npy`` file of shape ``(C, 3)``; it is sent
        as the ``pos`` multipart field so the server can use exact coordinates
        instead of falling back to montage name-matching. ``fs`` is forwarded
        as a query parameter and used as the sampling-rate fallback for ``.mat``
        files that lack a sampling-rate field.
        # ``age_group`` (``"child"``/``"adult"``/``"older"``) and ``sex``
        # (``"female"``/``"male"``/``"unknown"``) are forwarded as query
        # parameters and only affect the ADHD model.
        """
        url = f"{self._server.base_url}/api/predict/raw"
        task_value = _TASK_MAP[task]

        files: dict[str, tuple[str, Any, str]] = {}
        try:
            with ExitStack() as stack:
                eeg_handle = stack.enter_context(record.file_path.open("rb"))
                files["eeg"] = (
                    record.file_path.name, eeg_handle, "application/octet-stream"
                )
                if pos_path:
                    pos_file = Path(pos_path)
                    pos_handle = stack.enter_context(pos_file.open("rb"))
                    files["pos"] = (pos_file.name, pos_handle, "application/octet-stream")
                params: dict[str, str] = {"task": task_value, "mode": "subject"}
                if fs is not None:
                    params["fs"] = str(fs)
                # if age_group:
                #     params["age_group"] = age_group
                # if sex:
                #     params["sex"] = sex
                response = requests.post(
                    url,
                    files=files,
                    params=params,
                    timeout=300,
                )
        except (requests.RequestException, OSError) as error:
            raise HttpReveUnavailableError(
                f"Could not reach the inference server at {self._server.base_url}: {error}"
            ) from error

        if response.status_code != 200:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            raise HttpReveUnavailableError(
                f"Inference server returned HTTP {response.status_code}: {detail}"
            )

        payload = response.json()

        probabilities = payload.get("probabilities") or {}
        values = list(probabilities.values())
        positive_probability = (
            float(values[1]) if len(values) > 1 else (float(values[0]) if values else 0.0)
        )
        label = int(positive_probability >= self._threshold)

        return PredictionResult(
            task=task,
            label=label,
            positive_probability=positive_probability,
            threshold=self._threshold,
            decision=task.decision_text(label),
            backend="http_reve",
            is_mock=False,
            message=f"Predicted label: {payload.get('predicted_label', 'Unknown')}.",
            details=_format_details(payload.get("preprocessing") or {}),
        )


def _format_details(preprocessing: dict) -> str:
    """Render the server's preprocessing summary as a short note."""
    n_segments = preprocessing.get("n_segments")
    n_channels = preprocessing.get("n_channels")
    window_sec = preprocessing.get("window_sec")
    target_fs = preprocessing.get("target_fs")
    duration = preprocessing.get("duration_sec")

    parts = ["Auto-preprocessed on the inference server"]
    if n_segments is not None and n_channels is not None:
        parts.append(f"{n_segments} segment(s) × {n_channels} channel(s)")
    if window_sec is not None and target_fs is not None:
        parts.append(f"{window_sec:g}s windows @ {target_fs:g} Hz")
    if duration is not None:
        parts.append(f"{duration:g}s of recording")
    return "; ".join(parts) + "."
