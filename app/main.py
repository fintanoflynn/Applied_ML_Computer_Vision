"""FastAPI service exposing the PlantVillage leaf-disease classifier.

Run locally with::

    uvicorn app.main:app --reload

Interactive docs:

* Swagger UI:  http://127.0.0.1:8000/docs
* ReDoc:       http://127.0.0.1:8000/redoc
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import RedirectResponse

from app import config
from app.classes import CLASS_NAMES, NUM_CLASSES, humanise
from app.model_loader import Predictor, load_predictor
from app.preprocessing import InvalidImageError
from app.schemas import (
    ClassesResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictResponse,
)

ACCEPTED_MIME_PREFIXES = ("image/",)
ACCEPTED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401 — FastAPI lifespan hook
    """Load the model checkpoint once, when the server starts."""
    predictor = load_predictor()
    app.state.predictor = predictor
    if predictor is None:
        print(
            f"[startup] No checkpoint at {config.MODEL_PATH}. "
            f"The API will return 503 on /predict until one is provided."
        )
    else:
        print(
            f"[startup] Loaded {predictor.model_type} model from "
            f"{predictor.checkpoint_path} on {predictor.device}."
        )
    yield


app = FastAPI(
    title="PlantVillage Leaf Disease Classifier",
    description=(
        "REST API that classifies a leaf photo into one of 38 PlantVillage "
        "classes (plant species + disease, or `healthy`).\n\n"
        "**Quick start:**\n\n"
        "1. `POST /predict` with `multipart/form-data` and a `file` field "
        "containing a JPEG or PNG image.\n"
        "2. You get back the top class plus the top-k most likely classes, "
        "with probabilities.\n\n"
        "The service handles all preprocessing internally — you do **not** need "
        "to resize, normalise, or convert the image yourself."
    ),
    version="0.1.0",
    lifespan=lifespan,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded."},
    },
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_predictor() -> Predictor:
    predictor: Predictor | None = app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"No model checkpoint loaded. Expected a file at "
                f"{config.MODEL_PATH}. Set MODEL_PATH / MODEL_TYPE env vars or "
                f"drop a checkpoint into the models/ directory and restart."
            ),
        )
    return predictor


def _build_prediction(idx: int, prob: float) -> Prediction:
    label = CLASS_NAMES[idx]
    plant, condition = humanise(label)
    return Prediction(label=label, plant=plant, condition=condition, probability=float(prob))


def _build_model_info(predictor: Predictor) -> ModelInfo:
    return ModelInfo(
        model_type=predictor.model_type,
        checkpoint_path=str(predictor.checkpoint_path),
        num_classes=NUM_CLASSES,
        input_size=list(predictor.input_size),
        device=predictor.device,
    )


async def _read_image_upload(file: UploadFile) -> bytes:
    """Validate the upload's content type and size, then return its bytes."""
    if file.content_type and not file.content_type.startswith(ACCEPTED_MIME_PREFIXES):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type {file.content_type!r}. "
                f"Send a JPEG or PNG image (image/jpeg, image/png)."
            ),
        )

    body = await file.read()
    if len(body) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(body) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Uploaded file is {len(body)} bytes, which exceeds the "
                f"{config.MAX_UPLOAD_BYTES}-byte limit."
            ),
        )
    return body


def _predict_with_topk(predictor: Predictor, image_bytes: bytes, k: int) -> PredictResponse:
    try:
        probs = predictor.predict_proba(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if probs.shape != (NUM_CLASSES,):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Model returned a probability vector of shape {probs.shape}, "
                f"but the API expected ({NUM_CLASSES},)."
            ),
        )

    top_indices = np.argsort(probs)[::-1][:k]
    top_k = [_build_prediction(int(i), float(probs[i])) for i in top_indices]
    return PredictResponse(
        top_prediction=top_k[0],
        top_k=top_k,
        model=_build_model_info(predictor),
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the root URL to the interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness check",
    tags=["meta"],
)
async def health() -> HealthResponse:
    """Report whether the service is up and a model was successfully loaded."""
    predictor: Predictor | None = app.state.predictor
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_type=predictor.model_type if predictor is not None else None,
    )


@app.get(
    "/classes",
    response_model=ClassesResponse,
    summary="List all supported classes",
    tags=["meta"],
)
async def classes() -> ClassesResponse:
    """Return every PlantVillage label the model can predict."""
    return ClassesResponse(num_classes=NUM_CLASSES, classes=CLASS_NAMES)


@app.get(
    "/model",
    response_model=ModelInfo,
    summary="Describe the loaded model",
    tags=["meta"],
    responses={503: {"model": ErrorResponse}},
)
async def model_info() -> ModelInfo:
    """Return metadata about the model checkpoint that is currently serving predictions."""
    return _build_model_info(_require_predictor())


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single leaf image",
    tags=["predictions"],
    responses={
        400: {"model": ErrorResponse, "description": "Empty file."},
        413: {"model": ErrorResponse, "description": "File too large."},
        415: {"model": ErrorResponse, "description": "Unsupported content type."},
        422: {"model": ErrorResponse, "description": "Image could not be decoded."},
        503: {"model": ErrorResponse, "description": "Model not loaded."},
    },
)
async def predict(
    file: Annotated[
        UploadFile,
        File(description="A JPEG or PNG photo of a single leaf."),
    ],
    top_k: Annotated[
        int,
        Query(
            ge=1,
            le=NUM_CLASSES,
            description="How many candidate classes to return, sorted by descending probability.",
        ),
    ] = 3,
) -> PredictResponse:
    """Classify the uploaded leaf image.

    The service handles all preprocessing — resizing, channel conversion, and
    normalisation — so you can send the original photograph straight from disk.

    **Request:** `multipart/form-data` with a `file` field. Optional query
    parameter `top_k` (default `3`, max `38`) controls how many candidate
    classes are returned.

    **Response:** JSON containing the most likely class, the top-k ranked
    candidates with their probabilities, and information about the model that
    produced the prediction.
    """
    predictor = _require_predictor()
    image_bytes = await _read_image_upload(file)
    return _predict_with_topk(predictor, image_bytes, k=top_k)
