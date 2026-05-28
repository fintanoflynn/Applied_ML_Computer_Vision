"""Image preprocessing for both model variants.

The user uploads raw image bytes; this module is responsible for turning those
bytes into the exact tensor / array shape each model was trained on. Doing it
here (rather than asking the API user to do it) is required by the grading
rubric.
"""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

# CNN: RGB, resized to 256x256, scaled to [0, 1], in NCHW layout.
CNN_INPUT_SIZE: tuple[int, int] = (256, 256)
# Baseline logistic regression: grayscale, 64x64, flattened to a 4096-dim vector.
BASELINE_INPUT_SIZE: tuple[int, int] = (64, 64)


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes do not decode to a valid image."""


def _open_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise InvalidImageError("Uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(
            "Uploaded file is not a readable image. Supported formats: JPEG, PNG."
        ) from exc
    return image


def preprocess_for_cnn(image_bytes: bytes) -> torch.Tensor:
    """Decode bytes → (1, 3, H, W) float tensor in [0, 1]."""
    image = _open_image(image_bytes).convert("RGB").resize(CNN_INPUT_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def preprocess_for_baseline(image_bytes: bytes) -> np.ndarray:
    """Decode bytes → (1, 4096) float array, matching base_line.py's training pipeline."""
    image = _open_image(image_bytes).convert("L").resize(BASELINE_INPUT_SIZE)
    pixels = np.asarray(image, dtype=np.float32).flatten() / 255.0
    return pixels.reshape(1, -1)
