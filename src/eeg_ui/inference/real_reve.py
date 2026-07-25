from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from eeg_ui.config import ReveSettings
from eeg_ui.inference.base import BaseInferenceService
from eeg_ui.schemas import DiseaseTask, EEGRecord, PredictionResult

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful handling of missing heavy dependencies
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = True
_BRAINDECODE_AVAILABLE = True

try:
    import torch
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from braindecode.models import REVE  # noqa: F401
except ImportError:
    _BRAINDECODE_AVAILABLE = False

# Checkpoints may store pickled config objects from the training project.
# Add common locations so that unpickling succeeds.
_TRAINING_ROOTS = [
    Path(__file__).resolve().parents[3] / "eeg",  # sibling of EEG-UI
    Path("E:/Desktop/eeg"),
]
for _p in _TRAINING_ROOTS:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class RealReveUnavailableError(RuntimeError):
    """Raised when the real REVE backend cannot be used."""


class RealReveInferenceService(BaseInferenceService):
    """REVE EEG classifier using locally trained checkpoints.

    Loads separate models for Depression and ADHD tasks.  Each model is
    a REVE encoder + linear classification head trained via linear probing.
    """

    def __init__(self, settings: ReveSettings, device: str, threshold: float) -> None:
        if not _TORCH_AVAILABLE:
            raise RealReveUnavailableError(
                "PyTorch is not installed.  Install it with ``pip install torch`` "
                "to use the real REVE backend."
            )
        if not _BRAINDECODE_AVAILABLE:
            raise RealReveUnavailableError(
                "braindecode is not installed.  Install it with "
                "``pip install braindecode`` to use the real REVE backend."
            )

        self._settings = settings
        self._threshold = threshold
        self._device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._models: dict[DiseaseTask, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, record: EEGRecord, task: DiseaseTask) -> PredictionResult:
        """Run multi-epoch REVE inference and return a binary prediction."""
        model = self._get_model(task)
        eeg, pos = self._prepare_input(record)

        if eeg is None:
            raise RealReveUnavailableError(
                "No signal data found in the uploaded file. "
                "For real REVE inference, upload a preprocessed .npy file "
                "with shape (n_epochs, n_channels, n_times)."
            )

        with torch.no_grad():
            result = _predict_multi_epoch(
                model=model,
                eeg=eeg,
                pos=pos,
                device=self._device,
            )

        positive_prob = result["positive_probability"]
        label = int(positive_prob >= self._threshold)

        return PredictionResult(
            task=task,
            label=label,
            positive_probability=positive_prob,
            threshold=self._threshold,
            decision=task.decision_text(label),
            backend="real_reve",
            is_mock=False,
            message=(
                f"Real REVE inference — {result['n_epochs']} epochs, "
                f"consensus {result['consensus']:.1%}. "
                f"Class probabilities: {result['class_probs']}."
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self, task: DiseaseTask):
        """Return (and cache) the model for *task*."""
        if task not in self._models:
            self._models[task] = _load_reve_model(
                checkpoint_path=self._checkpoint_for(task),
                device=self._device,
                settings=self._settings,
            )
        return self._models[task]

    def _checkpoint_for(self, task: DiseaseTask) -> str:
        checkpoint = (
            self._settings.depression_checkpoint
            if task == DiseaseTask.DEPRESSION
            else self._settings.adhd_checkpoint
        )
        if not checkpoint:
            raise RealReveUnavailableError(
                f"No checkpoint configured for task '{task.value}'. "
                f"Set 'reve.{task.value}_checkpoint' in configs/default.yaml."
            )
        path = Path(checkpoint)
        if not path.exists():
            raise RealReveUnavailableError(
                f"Checkpoint not found: {path}. "
                f"Please verify the path in configs/default.yaml."
            )
        return str(path)

    def _prepare_input(self, record: EEGRecord):
        """Extract (eeg, pos) from *record*, handling 2D → 3D promotion."""
        signal = record.signal
        if signal is None:
            return None, None

        if signal.ndim == 2:
            # (n_channels, n_times) → (1, n_channels, n_times)
            signal = signal[np.newaxis, :, :]

        pos = record.coordinates
        if pos is None:
            # Create zero-filled positions as fallback
            n_channels = signal.shape[1]
            pos = np.zeros((n_channels, 3), dtype=np.float32)

        return signal, pos


# ======================================================================
# Standalone helpers (mirror scripts/predict.py logic)
# ======================================================================


def _load_reve_model(
    checkpoint_path: str,
    device: torch.device,
    settings: ReveSettings,
):
    """Load a REVE model from a training checkpoint."""
    from braindecode.models import REVE

    LOGGER.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try to get config from checkpoint; fall back to YAML settings
    exp_cfg = ckpt.get("config")
    if exp_cfg is not None:
        model_cfg = exp_cfg.model
        n_outputs = getattr(model_cfg, "n_outputs", 2)
        n_times = getattr(model_cfg, "n_times", settings.n_times)
        input_window = getattr(model_cfg, "input_window_seconds", settings.input_window_seconds)
        sfreq = getattr(model_cfg, "sfreq", settings.sfreq)
        use_attn = getattr(model_cfg, "use_attention_pooling", settings.use_attention_pooling)
    else:
        n_outputs = 2
        n_times = settings.n_times
        input_window = settings.input_window_seconds
        sfreq = settings.sfreq
        use_attn = settings.use_attention_pooling

    LOGGER.info(
        "REVE config: n_outputs=%d, n_times=%d, sfreq=%g, attn_pool=%s",
        n_outputs, n_times, sfreq, use_attn,
    )

    model = REVE(
        n_outputs=n_outputs,
        n_chans=None,
        n_times=n_times,
        input_window_seconds=input_window,
        sfreq=sfreq,
        attention_pooling=use_attn,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    LOGGER.info("Model loaded (%s params)", sum(p.numel() for p in model.parameters()))
    return model


def _predict_multi_epoch(
    model,
    eeg: np.ndarray,
    pos: np.ndarray,
    device: torch.device,
) -> dict:
    """Run inference across all epochs and aggregate.

    Parameters
    ----------
    eeg : np.ndarray  shape (n_epochs, n_channels, n_times)
    pos : np.ndarray  shape (n_channels, 3)

    Returns
    -------
    dict with keys:
        positive_probability : float   probability of the positive (patient) class
        class_probs : list[float]      average probability per class
        n_epochs : int
        consensus : float              fraction of epochs voting majority
    """
    n_epochs = eeg.shape[0]
    all_probs: list[np.ndarray] = []
    all_preds: list[int] = []

    for i in range(n_epochs):
        epoch_t = torch.from_numpy(eeg[i]).unsqueeze(0).to(device)   # (1, C, T)
        pos_t = torch.from_numpy(pos).unsqueeze(0).to(device)        # (1, C, 3)

        logits = model(epoch_t, pos=pos_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]        # (n_classes,)

        all_probs.append(probs)
        all_preds.append(int(probs.argmax()))

    avg_probs = np.mean(all_probs, axis=0).tolist()                  # per-class averages

    # Binary: positive class = index 1
    positive_prob = avg_probs[1] if len(avg_probs) > 1 else avg_probs[0]

    # Consensus
    from collections import Counter
    votes = Counter(all_preds)
    majority = votes.most_common(1)[0][0]
    consensus = votes[majority] / n_epochs

    return {
        "positive_probability": float(positive_prob),
        "class_probs": avg_probs,
        "n_epochs": n_epochs,
        "consensus": consensus,
    }
