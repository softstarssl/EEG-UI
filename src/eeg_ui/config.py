from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the application configuration is invalid."""


@dataclass(frozen=True)
class AppSettings:
    title: str = "EEG Disease Classification Demo"
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


@dataclass(frozen=True)
class InferenceSettings:
    backend: str = "mock_reve"
    threshold: float = 0.5
    device: str = "auto"


@dataclass(frozen=True)
class FileSettings:
    allowed_extensions: tuple[str, ...] = (".edf", ".set", ".fif", ".mat", ".npy")
    max_size_mb: int = 1024


@dataclass(frozen=True)
class ReveSettings:
    model_id: str = "brain-bzh/reve-base"
    depression_checkpoint: str | None = None
    adhd_checkpoint: str | None = None
    target_sampling_rate: int = 200
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 99.5
    zscore: bool = True
    clip_std: float = 15.0
    window_seconds: int = 10
    window_overlap_seconds: int = 0
    patch_seconds: int = 1
    n_times: int = 2000
    sfreq: float = 200.0
    input_window_seconds: float = 10.0
    use_attention_pooling: bool = True


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8765
    url: str | None = None

    @property
    def base_url(self) -> str:
        if self.url:
            return self.url.rstrip("/")
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    inference: InferenceSettings
    files: FileSettings
    reve: ReveSettings
    server: ServerSettings = ServerSettings()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration section '{name}' must be a mapping.")
    return value


def _normalise_extensions(extensions: Any) -> tuple[str, ...]:
    if not isinstance(extensions, list) or not extensions:
        raise ConfigError("files.allowed_extensions must be a non-empty list.")
    normalised = tuple(str(value).lower() for value in extensions)
    if any(not value.startswith(".") for value in normalised):
        raise ConfigError("Every allowed file extension must start with '.'.")
    return normalised


def load_config(path: Path) -> Settings:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise ConfigError(f"Could not read configuration file: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML configuration: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping.")

    app_data = _section(raw, "app")
    inference_data = _section(raw, "inference")
    files_data = _section(raw, "files")
    reve_data = _section(raw, "reve")
    server_data = _section(raw, "server")

    try:
        threshold = float(inference_data.get("threshold", 0.5))
        max_size_mb = int(files_data.get("max_size_mb", 1024))
        server_port = int(server_data.get("port", 8765))
    except (TypeError, ValueError) as error:
        raise ConfigError(f"Invalid numeric configuration value: {error}") from error

    if not 0.0 <= threshold <= 1.0:
        raise ConfigError("inference.threshold must be between 0 and 1.")

    if max_size_mb <= 0:
        raise ConfigError("files.max_size_mb must be positive.")

    try:
        app = AppSettings(**app_data)
        inference = InferenceSettings(
            backend=str(inference_data.get("backend", "mock_reve")),
            threshold=threshold,
            device=str(inference_data.get("device", "auto")),
        )
        files = FileSettings(
            allowed_extensions=_normalise_extensions(
                files_data.get("allowed_extensions", list(FileSettings().allowed_extensions))
            ),
            max_size_mb=max_size_mb,
        )
        reve = ReveSettings(**reve_data)
        server = ServerSettings(
            host=str(server_data.get("host", "127.0.0.1")),
            port=server_port,
            url=server_data.get("url"),
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"Invalid configuration value: {error}") from error

    return Settings(app=app, inference=inference, files=files, reve=reve, server=server)
