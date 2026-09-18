from __future__ import annotations

import base64
import html
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eeg_ui.config import Settings
from eeg_ui.inference.factory import create_inference_service
from eeg_ui.inference.http_reve import HttpReveInferenceService
from eeg_ui.io.base import FileValidationError
from eeg_ui.io.registry import load_eeg_record
from eeg_ui.schemas import DiseaseTask, EEGRecord, PredictionResult

LOGGER = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"


def _image_data_uri(filename: str) -> str:
    path = _ASSETS_DIR / filename
    if not path.exists():
        return ""
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    mime = "image/png" if filename.endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


_BRAIN_IMG = _image_data_uri("brain.png")
_EEG_IMG = _image_data_uri("eeg_wave.png")

# ADHD demographics options — values must match the server's one-hot encoding.
# _AGE_GROUP_CHOICES = [("Child", "child"), ("Adult", "adult"), ("Older", "older")]
# _SEX_CHOICES = [("Female", "female"), ("Male", "male"), ("Unknown", "unknown")]


# ---------------------------------------------------------------------------
# Presentational helpers
# ---------------------------------------------------------------------------

def _display(value: object | None) -> str:
    return "Unknown" if value is None or value == "" else str(value)


def _esc(value: object | None) -> str:
    return html.escape(_display(value))


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    return f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024**2 else f"{size_bytes / 1024**2:.1f} MB"


def _format_shape(shape: object) -> str | None:
    if not isinstance(shape, tuple):
        return None
    return " × ".join(str(dim) for dim in shape)


def format_file_info(record: EEGRecord) -> str:
    metadata = record.metadata.get("file_metadata")
    shape = record.metadata.get("signal_shape")

    file_name = metadata.file_name if metadata else record.file_path.name
    file_type = metadata.file_type if metadata else record.file_type
    size_bytes = metadata.size_bytes if metadata else None
    extension_valid = metadata.extension_valid if metadata else None

    rows = [
        ("File name", _esc(file_name)),
        ("Extension", _esc(file_type)),
        ("File size", _esc(_format_size(size_bytes) if size_bytes is not None else None)),
        ("Extension validation", "Passed" if extension_valid else "Unknown"),
        ("Signal shape", _esc(_format_shape(shape))),
    ]

    if isinstance(shape, tuple) and len(shape) == 2:
        rows.append(("Channels", _esc(shape[0])))
        rows.append(("Time points", _esc(shape[1])))

    if record.sampling_rate is not None:
        rows.append(("Sampling rate", _esc(f"{record.sampling_rate:g} Hz")))

    if record.channel_names:
        rows.append(("Channels recorded", _esc(len(record.channel_names))))

    read_message = _esc(record.metadata.get("read_message"))

    rows_html = "\n".join(
        f'<div class="kv"><span class="k">{key}</span><span class="v">{value}</span></div>'
        for key, value in rows
    )
    note = f'<div class="card-note">{read_message}</div>' if read_message else ""

    return (
        '<div class="nr-card">'
        '<div class="card-eyebrow">File information</div>'
        f'<div class="kv-list">{rows_html}</div>'
        f"{note}"
        "</div>"
    )


def format_result(result: PredictionResult) -> str:
    task_name = _esc(result.task.display_name)
    decision = _esc(result.decision)
    is_positive = bool(result.label)

    badge_class = "badge--pos" if is_positive else "badge--neg"
    tag_class = "tag--pos" if is_positive else "tag--neg"
    fill_class = "meter-fill--pos" if is_positive else "meter-fill--neg"
    tag_text = f"{result.label} · {'Positive' if is_positive else 'Negative'}"
    prob_pct = result.positive_probability * 100
    thr_pct = result.threshold * 100

    if result.is_mock:
        banner = (
            '<div class="mock-banner">'
            "<span>Mock mode — no real model is connected. This is a stable simulated result "
            "for demonstrating the interface only.</span></div>"
        )
    else:
        banner = ""

    details_note = (
        f'<div class="card-note">{_esc(result.details)}</div>'
        if result.details
        else ""
    )

    detail = (
        f'<div class="detail"><span class="k">Task</span><span class="v">{task_name}</span></div>'
        f'<div class="detail"><span class="k">Prediction</span>'
        f'<span class="v"><span class="tag {tag_class}">{tag_text}</span></span></div>'
    )

    return (
        '<div class="result-card">'
        '<div class="card-eyebrow">AI Diagnosis Result</div>'
        f'<div class="nr-eeg-strip"><img src="{_EEG_IMG}" alt="EEG waveform"/></div>'
        f'<div class="decision-row"><span class="badge {badge_class}">{decision}</span></div>'
        '<div class="meter-row">'
        '<span class="meter-title">Positive-class probability</span>'
        f'<span class="meter-value">{result.positive_probability:.1%}</span>'
        "</div>"
        '<div class="meter">'
        f'<div class="meter-fill {fill_class}" style="width:{prob_pct:.1f}%"></div>'
        f'<div class="meter-threshold" style="left:{thr_pct:.1f}%"></div>'
        "</div>"
        f'<div class="detail-grid">{detail}</div>'
        f"{banner}"
        f"{details_note}"
        "</div>"
    )


def initial_file_info() -> str:
    return (
        '<div class="nr-card">'
        '<div class="card-eyebrow">File information</div>'
        '<div class="empty-hint">Upload an EEG file to view validated metadata.</div>'
        "</div>"
    )


def initial_result(settings: Settings | None = None) -> str:
    return (
        '<div class="result-card">'
        '<div class="card-eyebrow">AI Diagnosis Result</div>'
        f'<div class="nr-eeg-strip"><img src="{_EEG_IMG}" alt="EEG waveform"/></div>'
        '<div class="empty-hint">Select a task, upload an EEG file, then click '
        "<strong>Run Prediction</strong> to see the result here.</div>"
        "</div>"
    )


def error_result(message: str) -> str:
    return (
        '<div class="result-card">'
        '<div class="card-eyebrow">AI Diagnosis Result</div>'
        f'<div class="nr-eeg-strip"><img src="{_EEG_IMG}" alt="EEG waveform"/></div>'
        f'<div class="error-box">{_esc(message)}</div>'
        "</div>"
    )


def describe_file(file_path: str | None, settings: Settings) -> str:
    if not file_path:
        return initial_file_info()
    try:
        return format_file_info(load_eeg_record(file_path, settings.files))
    except FileValidationError as error:
        return (
            '<div class="nr-card">'
            '<div class="card-eyebrow">File information</div>'
            f'<div class="error-box">{_esc(str(error))}</div>'
            "</div>"
        )
    except Exception:
        LOGGER.exception("Unexpected error while preparing file metadata")
        return (
            '<div class="nr-card">'
            '<div class="card-eyebrow">File information</div>'
            '<div class="error-box">Metadata could not be prepared. Please upload the file again.</div>'
            "</div>"
        )


def run_prediction(
    file_path: str | None,
    task_value: str | None,
    settings: Settings,
    progress: Callable[..., Any] | None = None,
    pos_path: str | None = None,
    fs: float | None = None,
    # age_group: str | None = None,
    # sex: str | None = None,
) -> tuple[str, str, str]:
    logs: list[str] = []

    def update(fraction: float, description: str) -> None:
        logs.append(description)
        if progress is not None:
            progress(fraction, desc=description)

    try:
        if not task_value:
            raise ValueError("Select a classification task before running prediction.")
        try:
            task = DiseaseTask(task_value)
        except ValueError as error:
            raise ValueError("The selected classification task is invalid.") from error

        update(0.25, "[1/4] Validating uploaded file...")
        update(0.50, "[2/4] Reading EEG metadata...")
        record = load_eeg_record(file_path, settings.files)
        update(0.75, "[3/4] Running inference...")
        service = create_inference_service(
            settings.inference, settings.reve, settings.server
        )
        # demographics: dict[str, str] = {}
        # if task is DiseaseTask.ADHD:
        #     if age_group:
        #         demographics["age_group"] = age_group
        #     if sex:
        #         demographics["sex"] = sex
        if isinstance(service, HttpReveInferenceService):
            result = service.predict(
                record,
                task,
                pos_path=pos_path,
                fs=fs,
                # age_group=demographics.get("age_group"),
                # sex=demographics.get("sex"),
            )
        else:
            result = service.predict(record, task)
        update(1.0, "[4/4] Formatting prediction result...")
        logs.append("Done.")
        return "\n".join(logs), format_result(result), format_file_info(record)
    except (FileValidationError, ValueError, RuntimeError) as error:
        logs.append(f"Error: {error}")
        return "\n".join(logs), error_result(str(error)), describe_file(file_path, settings)
    except Exception:
        LOGGER.exception("Unexpected prediction failure")
        logs.append("Error: The prediction service failed unexpectedly. Please try again.")
        return (
            "\n".join(logs),
            error_result("The prediction service failed unexpectedly. Please try again."),
            describe_file(file_path, settings),
        )


# ---------------------------------------------------------------------------
# Static markup
# ---------------------------------------------------------------------------

def _hero_html(title: str) -> str:
    return (
        '<div class="nr-hero">'
        '<div class="nr-hero-main">'
        '<div class="nr-brand">'
        '<div class="nr-logo">🧠</div>'
        "<div>"
        f"<h1>{html.escape(title)}</h1>"
        '<p class="nr-tagline">AI-based Diagnosis</p>'
        "</div>"
        "</div>"
        '<div class="nr-badges">'
        '<span class="nr-pill">Research preview</span>'
        '<span class="nr-pill">Data stays local</span>'
        '<span class="nr-pill nr-pill--amber">Not for clinical use</span>'
        "</div>"
        "</div>"
        f'<div class="nr-hero-art"><img src="{_BRAIN_IMG}" alt="Brain illustration"/></div>'
        "</div>"
    )


_INPUT_HEAD = (
    '<div class="nr-section-title">1 · Configure</div>'
    '<div class="nr-section-sub">Choose a task, then upload one subject\'s raw EEG file '
    "(.edf, .bdf, .set, .fif, .raw, .cnt, .mat, .npy). Supports unprocessed data; "
    "one subject per run.</div>"
)

_INPUT_FOOT = (
    '<div class="nr-section-title nr-steps-title">2 · Quick guide</div>'
    '<div class="nr-steps">'
    '<div class="nr-step"><span class="num">1</span><span>Pick <strong>Depression</strong> or '
    "<strong>ADHD</strong> as the classification task.</span></div>"
    '<div class="nr-step"><span class="num">2</span><span>Upload an EEG file — drag &amp; drop '
    "or browse your disk.</span></div>"
    '<div class="nr-step"><span class="num">3</span><span>Click <strong>Run Prediction</strong> '
    "to classify the recording.</span></div>"
    "</div>"
)

_FOOTER = (
    '<div class="nr-footer">NeuroRegen AI · Research prototype · '
    "EEG files are sent only to your local inference server (127.0.0.1), never to the internet · "
    "Not for medical or clinical use.</div>"
)


_CSS = """
/* ---------- Base container ---------- */
.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
}

/* ---------- Hero ---------- */
.nr-hero {
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 32px;
  background: linear-gradient(120deg, #4f46e5 0%, #6d28d9 45%, #9333ea 100%);
  color: #ffffff;
  border-radius: 22px;
  padding: 34px 38px;
  margin-bottom: 26px;
  box-shadow: 0 18px 40px -12px rgba(79, 70, 229, 0.45);
}
.nr-hero::before {
  content: "";
  position: absolute;
  top: -70px; right: -40px;
  width: 260px; height: 260px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0) 70%);
}
.nr-hero::after {
  content: "";
  position: absolute;
  bottom: -90px; left: 30%;
  width: 220px; height: 220px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0) 70%);
}
.nr-brand { display: flex; align-items: center; gap: 16px; position: relative; z-index: 1; }
.nr-logo {
  width: 52px; height: 52px; flex: 0 0 52px;
  display: flex; align-items: center; justify-content: center;
  font-size: 26px;
  border-radius: 14px;
  background: rgba(255,255,255,0.16);
  border: 1px solid rgba(255,255,255,0.24);
  backdrop-filter: blur(4px);
}
.nr-hero-main { position: relative; z-index: 1; min-width: 0; }
.nr-hero-art {
  position: relative; z-index: 1; flex: 0 0 auto;
  width: 160px;
  opacity: 0.95;
  filter: drop-shadow(0 6px 16px rgba(0, 0, 0, 0.25));
}
.nr-hero-art img { width: 100%; height: auto; display: block; }
.nr-hero h1 { margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.4px; line-height: 1.2; }
.nr-tagline { margin: 4px 0 0; font-size: 15px; font-weight: 500; opacity: 0.92; }
.nr-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px; position: relative; z-index: 1; }
.nr-pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 12.5px; font-weight: 600;
  padding: 5px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.16);
  border: 1px solid rgba(255,255,255,0.28);
}
.nr-pill--amber { background: rgba(255, 255, 255, 0.94); color: #b45309; border-color: transparent; }

/* ---------- Panels / cards ---------- */
.nr-panel {
  background: linear-gradient(160deg, #1e1b4b 0%, #0f172a 100%) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 18px !important;
  padding: 24px !important;
  box-shadow: 0 12px 32px -12px rgba(15,23,42,0.55) !important;
  height: 100%;
  --block-label-text-color: #ffffff;
}
.nr-section-title { font-size: 17px; font-weight: 700; color: #ffffff; margin: 0 0 4px; }
.nr-section-sub { font-size: 13px; color: #cbd5e1; margin: 0 0 24px; line-height: 1.55; }
.nr-field { margin-bottom: 24px !important; }

.nr-card, .result-card {
  background: #ffffff;
  border: 1px solid #e7eaf3;
  border-radius: 18px;
  padding: 24px;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 12px 32px -12px rgba(15,23,42,0.10);
  margin-bottom: 16px;
}
.result-card { position: relative; }
.card-eyebrow {
  font-size: 14px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
  color: #6366f1; margin: 0 0 14px;
}
.nr-eeg-strip {
  position: absolute;
  top: 14px;
  right: 24px;
  width: 280px;
  opacity: 0.92;
  pointer-events: none;
}
.nr-eeg-strip img { width: 100%; height: auto; display: block; }

/* ---------- Decision badge ---------- */
.decision-row { margin: 4px 0 22px; }
.badge {
  display: inline-flex; align-items: center; gap: 9px;
  font-size: 20px; font-weight: 700;
  padding: 10px 18px; border-radius: 12px;
}
.badge::before { content: ""; width: 9px; height: 9px; border-radius: 50%; }
.badge--pos { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.badge--pos::before { background: #ef4444; box-shadow: 0 0 0 3px rgba(239,68,68,0.15); }
.badge--neg { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
.badge--neg::before { background: #10b981; box-shadow: 0 0 0 3px rgba(16,185,129,0.15); }

/* ---------- Probability meter ---------- */
.meter-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.meter-title { font-size: 13.5px; font-weight: 600; color: #475569; }
.meter-value { font-size: 22px; font-weight: 800; color: #0f172a; font-variant-numeric: tabular-nums; }
.meter { position: relative; height: 12px; border-radius: 999px; background: #eef0f6; overflow: hidden; }
.meter-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; }
.meter-fill--pos { background: linear-gradient(90deg, #f43f5e, #ef4444); }
.meter-fill--neg { background: linear-gradient(90deg, #10b981, #34d399); }
.meter-threshold {
  position: absolute; top: -3px; bottom: -3px; width: 3px; border-radius: 2px;
  background: #ffffff; border: 1px solid #94a3b8; transform: translateX(-50%);
}
.meter-caption { margin-top: 8px; font-size: 12.5px; color: #64748b; }

/* ---------- Detail grid ---------- */
.detail-grid {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
  margin-top: 20px;
}
.detail {
  background: #f8fafc; border: 1px solid #eef1f6; border-radius: 12px;
  padding: 12px 14px; display: flex; flex-direction: column; gap: 3px;
}
.detail .k { font-size: 11.5px; font-weight: 600; letter-spacing: 0.3px; text-transform: uppercase; color: #94a3b8; }
.detail .v { font-size: 14.5px; font-weight: 650; color: #0f172a; }
.tag { display: inline-block; padding: 2px 9px; border-radius: 7px; font-size: 13px; font-weight: 700; }
.tag--pos { background: #fef2f2; color: #dc2626; }
.tag--neg { background: #ecfdf5; color: #059669; }

/* ---------- Banners / notes / states ---------- */
.mock-banner {
  display: flex; align-items: center; gap: 10px;
  margin-top: 18px; padding: 12px 14px; border-radius: 12px;
  background: #fffbeb; border: 1px solid #fde68a; color: #92400e; font-size: 13px; line-height: 1.5;
}
.real-banner {
  display: flex; align-items: center; gap: 10px;
  margin-top: 18px; padding: 12px 14px; border-radius: 12px;
  background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; font-size: 13px; line-height: 1.5;
}
.disclaimer { margin-top: 14px; font-size: 12px; color: #94a3b8; line-height: 1.5; }
.card-note {
  margin-top: 14px; padding: 10px 12px; border-radius: 10px;
  background: #f8fafc; border: 1px solid #eef1f6; color: #475569; font-size: 12.5px; line-height: 1.5;
}
.empty-hint { color: #94a3b8; font-size: 14px; padding: 8px 0; line-height: 1.6; }
.error-box {
  padding: 14px 16px; border-radius: 12px;
  background: #fef2f2; border: 1px solid #fecaca; color: #b91c1c; font-size: 14px; font-weight: 600;
}

/* ---------- File-info key/value list ---------- */
.kv-list { display: flex; flex-direction: column; }
.kv { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 9px 0; border-bottom: 1px solid #f1f4f9; }
.kv:last-child { border-bottom: none; }
.kv .k { font-size: 13px; color: #64748b; font-weight: 500; flex: 0 0 auto; }
.kv .v { font-size: 13.5px; color: #0f172a; font-weight: 650; text-align: right; word-break: break-all; }

/* ---------- Steps / footer ---------- */
.nr-steps-title { margin: 24px 0 4px; }
.nr-steps { display: flex; flex-direction: column; gap: 9px; margin-top: 14px; }
.nr-step { display: flex; gap: 10px; align-items: flex-start; font-size: 13px; color: #cbd5e1; line-height: 1.5; }
.nr-step .num {
  flex: 0 0 22px; width: 22px; height: 22px; border-radius: 7px;
  background: rgba(129, 140, 248, 0.24); color: #e0e7ff; font-weight: 700; font-size: 12px;
  display: flex; align-items: center; justify-content: center; margin-top: 1px;
}
.nr-footer { text-align: center; font-size: 12px; color: #94a3b8; margin-top: 6px; padding: 8px 0 16px; line-height: 1.6; }

/* ---------- Buttons ---------- */
button.primary, button[variant="primary"] {
  border-radius: 10px !important;
  font-weight: 600 !important;
  box-shadow: 0 8px 18px -8px rgba(79, 70, 229, 0.6) !important;
}
"""


def create_app(settings: Settings):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Gradio is not installed. Run 'pip install -r requirements.txt' first.") from error

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        font=["Segoe UI", "PingFang SC", "Microsoft YaHei", "ui-sans-serif", "system-ui", "sans-serif"],
    )

    def handle_run(
        uploaded: str | None,
        pos_uploaded: str | None,
        selected_task: str | None,
        fs_value: float | None,
        # age_group: str | None,
        # sex: str | None,
        progress=gr.Progress(),
    ) -> tuple[str, str, str]:
        return run_prediction(
            uploaded, selected_task, settings, progress, pos_uploaded, fs_value,
            # age_group, sex
        )

    # def _toggle_adhd_fields(selected_task: str | None) -> tuple[Any, Any]:
    #     visible = selected_task == DiseaseTask.ADHD.value
    #     return gr.update(visible=visible), gr.update(visible=visible)

    def _toggle_fs_field(uploaded: str | None) -> Any:
        is_mat = bool(uploaded) and Path(uploaded).suffix.lower() == ".mat"
        return gr.update(visible=is_mat)

    def _reset() -> tuple[Any, ...]:
        return (
            None,  # file_input
            None,  # pos_file
            DiseaseTask.DEPRESSION.value,  # task
            gr.update(value=200, visible=False),  # fs
            # gr.update(value=None, visible=False),  # age_group
            # gr.update(value=None, visible=False),  # sex
            initial_file_info(),  # file_info
            "",  # log
            initial_result(settings),  # result
        )

    with gr.Blocks(title=settings.app.title) as demo:
        gr.HTML(_hero_html(settings.app.title), apply_default_css=False)

        with gr.Row(equal_height=True):
            with gr.Column(scale=5, min_width=340):
                with gr.Group(elem_classes=["nr-panel"]):
                    gr.HTML(_INPUT_HEAD, apply_default_css=False)
                    task = gr.Dropdown(
                        choices=[(item.display_name, item.value) for item in DiseaseTask],
                        value=DiseaseTask.DEPRESSION.value,
                        label="Classification task",
                        elem_classes=["nr-field"],
                    )
                    # age_group = gr.Dropdown(
                    #     choices=_AGE_GROUP_CHOICES,
                    #     value=None,
                    #     label="Age group (ADHD)",
                    #     elem_classes=["nr-field"],
                    #     visible=False,
                    # )
                    # sex = gr.Dropdown(
                    #     choices=_SEX_CHOICES,
                    #     value=None,
                    #     label="Sex (ADHD)",
                    #     elem_classes=["nr-field"],
                    #     visible=False,
                    # )
                    file_input = gr.File(
                        label="EEG file",
                        file_types=list(settings.files.allowed_extensions),
                        type="filepath",
                        elem_classes=["nr-field"],
                    )
                    fs_input = gr.Number(
                        label="Sampling rate (Hz) — .mat only",
                        value=200,
                        precision=0,
                        minimum=1,
                        elem_classes=["nr-field"],
                        visible=False,
                    )
                    pos_file = gr.File(
                        label="Electrode positions (optional .npy, C×3)",
                        file_types=[".npy"],
                        type="filepath",
                        elem_classes=["nr-field"],
                    )
                    with gr.Row(elem_classes=["nr-field"]):
                        run_button = gr.Button("Run Prediction", variant="primary")
                        reset_button = gr.Button("Reset")
                    gr.HTML(_INPUT_FOOT, apply_default_css=False)

            with gr.Column(scale=7, min_width=420):
                result_output = gr.HTML(initial_result(settings), apply_default_css=False)
                file_info = gr.HTML(initial_file_info(), apply_default_css=False)
                with gr.Accordion("Run status & log", open=False):
                    log_output = gr.Textbox(label="Log", lines=8, interactive=False)

        gr.HTML(_FOOTER, apply_default_css=False)

        file_input.change(
            lambda uploaded: describe_file(uploaded, settings),
            inputs=file_input,
            outputs=file_info,
        )
        file_input.change(
            _toggle_fs_field,
            inputs=file_input,
            outputs=fs_input,
        )
        # task.change(
        #     _toggle_adhd_fields,
        #     inputs=task,
        #     outputs=[age_group, sex],
        # )
        run_button.click(
            handle_run,
            inputs=[file_input, pos_file, task, fs_input],  # , age_group, sex
            outputs=[log_output, result_output, file_info],
            concurrency_limit=1,
        )
        reset_button.click(
            _reset,
            outputs=[file_input, pos_file, task, fs_input, file_info, log_output, result_output],  # , age_group, sex
        )

    # Gradio 6 moved `theme`/`css` from the Blocks constructor to `launch()`.
    # Attach them so the caller can pass them through at launch time.
    demo.theme = theme
    demo.app_css = _CSS

    return demo.queue(default_concurrency_limit=1)
