"""Load the CNN or baseline checkpoint and expose a unified prediction API.

A single ``Predictor`` instance is constructed at FastAPI startup. The rest of
the app does not need to know which architecture is loaded — it just calls
``predictor.predict_proba(image_bytes)``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from app import config
from app.classes import CLASS_NAMES, NUM_CLASSES
from app.preprocessing import (
    BASELINE_INPUT_SIZE,
    CNN_INPUT_SIZE,
    RESNET_INPUT_SIZE,
    preprocess_for_baseline,
    preprocess_for_cnn,
    preprocess_for_resnet,
)

# Make `from src.models.cnn_scratch import CNN` work regardless of where uvicorn
# was launched from.
sys.path.insert(0, str(config.PROJECT_ROOT))


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no checkpoint was loaded."""


@dataclass(frozen=True)
class Predictor:
    """Unified interface around either CNN or baseline."""

    model_type: str
    checkpoint_path: Path
    device: str
    input_size: tuple[int, int]
    _predict_proba_fn: Callable[[bytes], np.ndarray]

    def predict_proba(self, image_bytes: bytes) -> np.ndarray:
        """Return a 1-D ``(num_classes,)`` array of class probabilities."""
        return self._predict_proba_fn(image_bytes)


def _load_cnn(checkpoint_path: Path) -> Predictor:
    from src.models.cnn_scratch import CNN  # local import to avoid torch cost on baseline path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # The checkpoint stores the dataset channel mean/std the model was normalised
    # with at training time. We must reapply the same normalisation at inference,
    # or the model sees a different input distribution and predicts poorly.
    mean = std = None
    if isinstance(checkpoint, dict):
        # Tolerate the various keys used to wrap the weights.
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model_state")
            or checkpoint
        )
        if checkpoint.get("mean") is not None and checkpoint.get("std") is not None:
            mean = torch.tensor(checkpoint["mean"], device=device).view(1, 3, 1, 1)
            std = torch.tensor(checkpoint["std"], device=device).view(1, 3, 1, 1)
        saved_classes = checkpoint.get("classes")
        if saved_classes is not None and list(saved_classes) != CLASS_NAMES:
            raise RuntimeError(
                "CNN checkpoint was trained on a different class ordering than "
                "app/classes.py. Predictions would be mislabelled."
            )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device).eval()

    def _predict(image_bytes: bytes) -> np.ndarray:
        tensor = preprocess_for_cnn(image_bytes).to(device)
        if mean is not None:
            tensor = (tensor - mean) / std
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    return Predictor(
        model_type="cnn",
        checkpoint_path=checkpoint_path,
        device=device,
        input_size=CNN_INPUT_SIZE,
        _predict_proba_fn=_predict,
    )


def _load_resnet(checkpoint_path: Path) -> Predictor:
    from src.models.resnet_transfer import build_resnet18  # local import: needs src on path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18(num_classes=NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # train.py saves {"model_state": ..., "classes": ..., ...}; also tolerate a
    # "state_dict" key or a raw state_dict for hand-saved checkpoints.
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        saved_classes = checkpoint.get("classes")
        if saved_classes is not None and list(saved_classes) != CLASS_NAMES:
            raise RuntimeError(
                "ResNet checkpoint was trained on a different class ordering than "
                "app/classes.py. Predictions would be mislabelled. Re-export the "
                "checkpoint or align CLASS_NAMES with the training class list."
            )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device).eval()

    def _predict(image_bytes: bytes) -> np.ndarray:
        tensor = preprocess_for_resnet(image_bytes).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    return Predictor(
        model_type="resnet",
        checkpoint_path=checkpoint_path,
        device=device,
        input_size=RESNET_INPUT_SIZE,
        _predict_proba_fn=_predict,
    )


def _load_baseline(checkpoint_path: Path) -> Predictor:
    sklearn_model = joblib.load(checkpoint_path)

    # The sklearn classifier knows its own label ordering. Map it onto the
    # canonical CLASS_NAMES index order so callers always see the same layout.
    model_classes = list(getattr(sklearn_model, "classes_", []))
    if not model_classes:
        raise RuntimeError(
            "Baseline checkpoint has no `classes_` attribute — was it really a "
            "fitted sklearn classifier?"
        )

    permutation = np.array(
        [model_classes.index(name) for name in CLASS_NAMES if name in model_classes]
    )
    if len(permutation) != NUM_CLASSES:
        raise RuntimeError(
            f"Baseline classifier has {len(model_classes)} classes, but the API "
            f"expects {NUM_CLASSES}. Did training cover all PlantVillage classes?"
        )

    def _predict(image_bytes: bytes) -> np.ndarray:
        features = preprocess_for_baseline(image_bytes)
        probs = sklearn_model.predict_proba(features)[0]
        return probs[permutation]

    return Predictor(
        model_type="baseline",
        checkpoint_path=checkpoint_path,
        device="cpu",
        input_size=BASELINE_INPUT_SIZE,
        _predict_proba_fn=_predict,
    )


def load_all_predictors() -> dict[str, Predictor]:
    """Load every model whose default checkpoint file is present on disk.

    Models without a checkpoint are silently skipped so the API can come up
    with whichever subset of models the user has trained. Failures to load a
    present checkpoint are reported but do not block startup of the others.
    """
    loaders = {
        "resnet": (_load_resnet, config.DEFAULT_CHECKPOINTS["resnet"]),
        "cnn": (_load_cnn, config.DEFAULT_CHECKPOINTS["cnn"]),
        "baseline": (_load_baseline, config.DEFAULT_CHECKPOINTS["baseline"]),
    }
    predictors: dict[str, Predictor] = {}
    for name, (loader_fn, ckpt_path) in loaders.items():
        if not ckpt_path.is_file():
            print(f"[startup] No {name} checkpoint at {ckpt_path}; skipping.")
            continue
        try:
            predictors[name] = loader_fn(ckpt_path)
            print(f"[startup] Loaded {name} from {ckpt_path}.")
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"[startup] Failed to load {name} from {ckpt_path}: {exc}")
    return predictors


def load_predictor() -> Predictor | None:
    """Build a predictor based on env-var config, or ``None`` if no checkpoint.

    Returning ``None`` (rather than raising) lets the API still come up — useful
    when teammates have not yet handed over the trained weights. The health
    endpoint and /predict will then surface the missing-model state cleanly.
    """
    if not config.MODEL_PATH.is_file():
        return None

    if config.MODEL_TYPE == "cnn":
        return _load_cnn(config.MODEL_PATH)
    if config.MODEL_TYPE == "resnet":
        return _load_resnet(config.MODEL_PATH)
    if config.MODEL_TYPE == "baseline":
        return _load_baseline(config.MODEL_PATH)

    raise ValueError(
        f"Unknown MODEL_TYPE={config.MODEL_TYPE!r}. Expected 'cnn', 'resnet', or 'baseline'."
    )
