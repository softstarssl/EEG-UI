from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eeg_ui.config import load_config
from eeg_ui.ui.gradio_app import create_app


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    app = create_app(config)
    app.launch(
        server_name=config.app.host,
        server_port=config.app.port,
        share=config.app.share,
        theme=app.theme,
        css=app.app_css,
    )


if __name__ == "__main__":
    main()
