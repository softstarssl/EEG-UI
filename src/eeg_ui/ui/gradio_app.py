from __future__ import annotations

import html
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eeg_ui.config import Settings
from eeg_ui.inference.factory import create_inference_service
from eeg_ui.io.base import FileValidationError
from eeg_ui.io.registry import load_eeg_record
from eeg_ui.schemas import DiseaseTask, EEGRecord, PredictionResult

LOGGER = logging.getLogger(__name__)

DISCLAIMER = "Initial model, continuously improving"


def _unknown(value: object | None) -> str:
    return "Unknown" if value is None or value == "" else str(value)


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    return f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024**2 else f"{size_bytes / 1024**2:.1f} MB"


def format_file_info(record: EEGRecord) -> str:
    metadata = record.metadata.get("file_metadata")
    shape = record.metadata.get("signal_shape")
    channels = shape[0] if isinstance(shape, tuple) and len(shape) == 2 else None
    timepoints = shape[1] if isinstance(shape, tuple) and len(shape) == 2 else None
    rows = [
        ("File name", metadata.file_name if metadata else record.file_path.name),
        ("Extension", metadata.file_type if metadata else record.file_type),
        ("File size", _format_size(metadata.size_bytes) if metadata else "Unknown"),
        ("Extension validation", "Passed" if metadata and metadata.extension_valid else "Unknown"),
        ("Signal shape", shape),
    ]
    table_rows = "\n".join(
        f"| {name} | {html.escape(_unknown(value)).replace('|', '&#124;')} |" for name, value in rows
    )
    message = html.escape(_unknown(record.metadata.get("read_message"))).replace("|", "&#124;")
    return f"## File information\n\n| Field | Value |\n| --- | --- |\n{table_rows}\n\n{message}"


def format_result(result: PredictionResult) -> str:
    return (
        "## AI Diagnosis Result\n\n"
        f"| Field | Value |\n| --- | --- |\n"
        f"| Task | {result.task.display_name} |\n"
        f"| Prediction | {result.label} |\n"
        f"| Decision | {result.decision} |\n"
        f"| Positive-class probability | {result.positive_probability:.1%} |\n\n"
        f"{DISCLAIMER}"
    )


def initial_file_info() -> str:
    return "## File information\n\nUpload an EEG file to view validated metadata."


def initial_result(settings: Settings | None = None) -> str:
    return f"## AI Diagnosis Result\n\n{DISCLAIMER}"


def describe_file(file_path: str | None, settings: Settings) -> str:
    if not file_path:
        return initial_file_info()
    try:
        return format_file_info(load_eeg_record(file_path, settings.files))
    except FileValidationError as error:
        return f"## File information\n\n**File validation error:** {html.escape(str(error))}"
    except Exception:
        LOGGER.exception("Unexpected error while preparing file metadata")
        return "## File information\n\n**File error:** Metadata could not be prepared. Please upload the file again."


def run_prediction(
    file_path: str | None,
    task_value: str | None,
    settings: Settings,
    progress: Callable[..., Any] | None = None,
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
        result = create_inference_service(settings.inference, settings.reve).predict(record, task)
        update(1.0, "[4/4] Formatting prediction result...")
        logs.append("Done.")
        return "\n".join(logs), format_result(result), format_file_info(record)
    except (FileValidationError, ValueError, RuntimeError) as error:
        logs.append(f"Error: {error}")
        return "\n".join(logs), f"## AI Diagnosis Result\n\n**Could not run prediction.** {html.escape(str(error))}", describe_file(file_path, settings)
    except Exception:
        LOGGER.exception("Unexpected prediction failure")
        logs.append("Error: The prediction service failed unexpectedly. Please try again.")
        return "\n".join(logs), "## AI Diagnosis Result\n\n**Could not run prediction.** Please try again.", describe_file(file_path, settings)


def _reset(settings: Settings) -> tuple[None, str, str, str, str]:
    return None, DiseaseTask.DEPRESSION.value, initial_file_info(), "", initial_result(settings)


def create_app(settings: Settings):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Gradio is not installed. Run 'pip install -r requirements.txt' first.") from error

    def handle_run(
        uploaded: str | None,
        selected_task: str | None,
        progress=gr.Progress(),
    ) -> tuple[str, str, str]:
        return run_prediction(uploaded, selected_task, settings, progress)

    with gr.Blocks(
        title="NeuroRegen AI",
        css="""
            * { font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif !important; }
            h1, h2, h3, h4 { font-variant-ligatures: none; }
        """,
    ) as demo:
        gr.Markdown("# NeuroRegen AI")
        gr.Markdown("### AI-based Diagnosis")
        with gr.Row():
            with gr.Column():
                task = gr.Dropdown(
                    choices=[(item.display_name, item.value) for item in DiseaseTask],
                    value=DiseaseTask.DEPRESSION.value,
                    label="Task",
                )
                file_input = gr.File(
                    label="EEG file",
                    file_types=list(settings.files.allowed_extensions),
                    type="filepath",
                )
                with gr.Row():
                    run_button = gr.Button("Run Prediction", variant="primary")
                    reset_button = gr.Button("Reset")
            with gr.Column():
                result_output = gr.Markdown(initial_result(settings))
                log_output = gr.Textbox(label="Run status and log", lines=8, interactive=False)
                file_info = gr.Markdown(initial_file_info())
        file_input.change(
            lambda uploaded: describe_file(uploaded, settings),
            inputs=file_input,
            outputs=file_info,
        )
        run_button.click(
            handle_run,
            inputs=[file_input, task],
            outputs=[log_output, result_output, file_info],
            concurrency_limit=1,
        )
        reset_button.click(
            lambda: _reset(settings),
            outputs=[file_input, task, file_info, log_output, result_output],
        )

    return demo.queue(default_concurrency_limit=1)
