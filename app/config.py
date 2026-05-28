"""Runtime configuration, read once at import time from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Which architecture to serve: "cnn" or "baseline".
MODEL_TYPE: str = os.environ.get("MODEL_TYPE", "cnn").lower()

_DEFAULT_CHECKPOINTS = {
    "cnn": PROJECT_ROOT / "models" / "cnn_scratch.pth",
    "baseline": PROJECT_ROOT / "models" / "baseline.joblib",
}

_env_path = os.environ.get("MODEL_PATH")
MODEL_PATH: Path = Path(_env_path) if _env_path else _DEFAULT_CHECKPOINTS.get(
    MODEL_TYPE, _DEFAULT_CHECKPOINTS["cnn"]
)

# Maximum upload size accepted by /predict (in bytes). 10 MiB by default.
MAX_UPLOAD_BYTES: int = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
