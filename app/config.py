"""Runtime configuration, read once at import time from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Default model used by the legacy /predict endpoint (no model_type in the
# path). The landing page and /predict/{model_type} can address any loaded
# model regardless of this setting.
MODEL_TYPE: str = os.environ.get("MODEL_TYPE", "resnet").lower()

DEFAULT_CHECKPOINTS = {
    "cnn": PROJECT_ROOT / "models" / "cnn_scratch.pth",
    "resnet": PROJECT_ROOT / "models" / "resnet18_best.pt",
    "baseline": PROJECT_ROOT / "models" / "baseline.joblib",
}

_env_path = os.environ.get("MODEL_PATH")
MODEL_PATH: Path = Path(_env_path) if _env_path else DEFAULT_CHECKPOINTS.get(
    MODEL_TYPE, DEFAULT_CHECKPOINTS["resnet"]
)

# Maximum upload size accepted by /predict (in bytes). 10 MiB by default.
MAX_UPLOAD_BYTES: int = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
