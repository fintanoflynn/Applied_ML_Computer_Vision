"""FastAPI service exposing the PlantVillage leaf-disease classifier.

Run locally with::

    uvicorn app.main:app --reload

Interactive docs:

* Swagger UI:  http://127.0.0.1:8000/docs
* ReDoc:       http://127.0.0.1:8000/redoc
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Path as PathParam, Query, UploadFile, status
from fastapi.responses import HTMLResponse

from app import config
from app.classes import CLASS_NAMES, NUM_CLASSES, humanise
from app.model_loader import Predictor, load_all_predictors
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


def _error_response(description: str, example_detail: str) -> dict:
    """Build an OpenAPI response entry that pins a concrete example body.

    Without an explicit ``content`` example, Swagger renders the same generic
    placeholder for every error status code (because they all share the same
    ``ErrorResponse`` schema), which makes the docs look like every error
    returns the same message. Spelling out the example per status keeps the
    docs honest about what users will actually see.
    """
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {
            "application/json": {"example": {"detail": example_detail}}
        },
    }


SUPPORTED_MODELS = ("resnet", "cnn", "baseline")
ModelType = Literal["resnet", "cnn", "baseline"]


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401 — FastAPI lifespan hook
    """Load every available model checkpoint once, when the server starts."""
    predictors = load_all_predictors()
    app.state.predictors = predictors
    # The default predictor backs the legacy /predict (no model_type) endpoint.
    # Prefer the configured MODEL_TYPE if loaded; fall back to whichever model
    # came up successfully so /predict still works.
    default = predictors.get(config.MODEL_TYPE) or next(iter(predictors.values()), None)
    app.state.default_predictor = default
    if not predictors:
        print(
            "[startup] No model checkpoints loaded. /predict endpoints will "
            "return 503 until a checkpoint is placed in models/."
        )
    else:
        loaded = ", ".join(predictors.keys())
        default_name = default.model_type if default is not None else "none"
        print(f"[startup] Models loaded: {loaded}. Default for /predict: {default_name}.")
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
        503: _error_response(
            "Model not loaded.",
            f"No model checkpoint loaded. Expected a file at "
            f"{config.MODEL_PATH}. Set MODEL_PATH / MODEL_TYPE env vars or "
            f"drop a checkpoint into the models/ directory and restart.",
        ),
    },
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_predictor(model_type: str | None = None) -> Predictor:
    """Return a loaded ``Predictor`` for ``model_type``, or the default.

    Raises 503 if the requested model wasn't loaded (e.g. its checkpoint is
    missing). Raises 404 if ``model_type`` is given but not one of the
    architectures the API supports.
    """
    predictors: dict[str, Predictor] = app.state.predictors
    if model_type is None:
        predictor = app.state.default_predictor
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "No model checkpoints loaded. Drop a checkpoint into the "
                    "models/ directory and restart, or set MODEL_PATH."
                ),
            )
        return predictor

    if model_type not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Unknown model_type {model_type!r}. Supported: "
                f"{', '.join(SUPPORTED_MODELS)}."
            ),
        )

    predictor = predictors.get(model_type)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model {model_type!r} is not loaded on this server. "
                f"Loaded models: {list(predictors.keys()) or '[]'}. "
                f"Expected checkpoint at {config.DEFAULT_CHECKPOINTS.get(model_type)}."
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


_MODEL_BLURBS: dict[str, tuple[str, str]] = {
    "resnet": ("ResNet-18 (transfer learning)", "Test accuracy 92.84%"),
    "cnn":    ("CNN from scratch",              "Validation macro-F1 0.989"),
    "baseline": ("Logistic regression baseline", "Grayscale 32x32"),
}


def _render_landing_html(loaded: list[str]) -> str:
    if not loaded:
        cards = (
            "<p class='empty'>No models are loaded. Drop a checkpoint into "
            "<code>models/</code> and restart the server.</p>"
        )
    else:
        cards = "\n".join(_render_card(name) for name in loaded)
    return _LANDING_TEMPLATE.replace("{{CARDS}}", cards)


def _render_card(model_type: str) -> str:
    title, subtitle = _MODEL_BLURBS.get(model_type, (model_type.title(), ""))
    return (
        f"<section class='card' data-model='{model_type}'>"
        f"  <h2>{title}</h2>"
        f"  <p class='subtitle'>{subtitle}</p>"
        f"  <form onsubmit='predict(event, \"{model_type}\")'>"
        f"    <input type='file' name='file' accept='image/jpeg,image/png' required>"
        f"    <button type='submit'>Classify with {model_type}</button>"
        f"  </form>"
        f"  <div class='result' id='result-{model_type}'></div>"
        f"</section>"
    )


_LANDING_TEMPLATE = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>PlantVillage Leaf Disease Classifier</title>
<style>
 body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }
 h1 { margin-bottom: 0.25rem; }
 .lede { color: #555; margin-top: 0; }
 .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }
 .card { border: 1px solid #ddd; border-radius: 8px; padding: 1.25rem; background: #fafafa; }
 .card h2 { margin: 0 0 0.25rem; font-size: 1.15rem; }
 .subtitle { color: #666; margin: 0 0 1rem; font-size: 0.9rem; }
 input[type=file] { display: block; margin-bottom: 0.75rem; width: 100%; }
 button { padding: 0.5rem 1rem; border: 0; background: #2563eb; color: white; border-radius: 4px; cursor: pointer; font-size: 0.95rem; }
 button:hover { background: #1d4ed8; }
 .result { margin-top: 1rem; font-size: 0.92rem; }
 .result .top { font-weight: 600; font-size: 1.05rem; }
 .result ul { padding-left: 1.25rem; margin: 0.4rem 0 0; }
 .result.error { color: #b91c1c; }
 .links { margin-top: 2rem; font-size: 0.9rem; color: #555; }
 .links a { color: #2563eb; text-decoration: none; margin-right: 1rem; }
 .links a:hover { text-decoration: underline; }
 .empty { color: #b91c1c; }
 code { background: #eee; padding: 0 0.3rem; border-radius: 3px; }
</style>
</head>
<body>
<h1>PlantVillage Leaf Disease Classifier</h1>
<p class='lede'>Upload a leaf photo and pick a model to classify it into one of 38 PlantVillage classes.</p>
<div class='cards'>
{{CARDS}}
</div>
<p class='links'>
 <a href='/docs'>Interactive API docs (Swagger)</a>
 <a href='/redoc'>ReDoc</a>
 <a href='/classes'>All 38 classes</a>
 <a href='/health'>/health</a>
</p>
<script>
async function predict(event, modelType) {
  event.preventDefault();
  const form = event.target;
  const fd = new FormData(form);
  const result = document.getElementById('result-' + modelType);
  result.classList.remove('error');
  result.innerText = 'Classifying...';
  try {
    const resp = await fetch('/predict/' + modelType + '?top_k=3', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok) {
      result.classList.add('error');
      result.innerText = 'Error ' + resp.status + ': ' + (data.detail || 'request failed');
      return;
    }
    const top = data.top_prediction;
    let html = "<div class='top'>" + top.plant + " — " + top.condition +
               " (" + (top.probability * 100).toFixed(1) + "%)</div>";
    html += '<ul>';
    for (const p of data.top_k.slice(1)) {
      html += '<li>' + p.plant + ' — ' + p.condition +
              ' (' + (p.probability * 100).toFixed(1) + '%)</li>';
    }
    html += '</ul>';
    result.innerHTML = html;
  } catch (e) {
    result.classList.add('error');
    result.innerText = 'Request failed: ' + e;
  }
}
</script>
</body>
</html>"""


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Landing page with one upload form per loaded model."""
    loaded = list(app.state.predictors.keys())
    return HTMLResponse(_render_landing_html(loaded))


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness / readiness check",
    tags=["meta"],
)
async def health() -> HealthResponse:
    """Report whether the service is up and which models loaded."""
    predictors: dict[str, Predictor] = app.state.predictors
    default: Predictor | None = app.state.default_predictor
    return HealthResponse(
        status="ok",
        model_loaded=bool(predictors),
        model_type=default.model_type if default is not None else None,
        models_loaded=list(predictors.keys()),
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
    responses={
        503: _error_response(
            "Model not loaded.",
            f"No model checkpoint loaded. Expected a file at "
            f"{config.MODEL_PATH}. Set MODEL_PATH / MODEL_TYPE env vars or "
            f"drop a checkpoint into the models/ directory and restart.",
        ),
    },
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
        400: _error_response(
            "Empty file.",
            "Uploaded file is empty.",
        ),
        413: _error_response(
            "File too large.",
            f"Uploaded file is 15728640 bytes, which exceeds the "
            f"{config.MAX_UPLOAD_BYTES}-byte limit.",
        ),
        415: _error_response(
            "Unsupported content type.",
            "Unsupported content type 'application/pdf'. Send a JPEG or PNG "
            "image (image/jpeg, image/png).",
        ),
        422: _error_response(
            "Image could not be decoded.",
            "Uploaded file is not a readable image. Supported formats: JPEG, PNG.",
        ),
        503: _error_response(
            "Model not loaded.",
            f"No model checkpoint loaded. Expected a file at "
            f"{config.MODEL_PATH}. Set MODEL_PATH / MODEL_TYPE env vars or "
            f"drop a checkpoint into the models/ directory and restart.",
        ),
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


@app.post(
    "/predict/{model_type}",
    response_model=PredictResponse,
    summary="Classify with a specific model",
    tags=["predictions"],
    responses={
        400: _error_response("Empty file.", "Uploaded file is empty."),
        404: _error_response(
            "Unknown model_type.",
            "Unknown model_type 'foo'. Supported: resnet, cnn, baseline.",
        ),
        413: _error_response(
            "File too large.",
            f"Uploaded file is 15728640 bytes, which exceeds the "
            f"{config.MAX_UPLOAD_BYTES}-byte limit.",
        ),
        415: _error_response(
            "Unsupported content type.",
            "Unsupported content type 'application/pdf'. Send a JPEG or PNG "
            "image (image/jpeg, image/png).",
        ),
        422: _error_response(
            "Image could not be decoded.",
            "Uploaded file is not a readable image. Supported formats: JPEG, PNG.",
        ),
        503: _error_response(
            "Requested model not loaded.",
            "Model 'baseline' is not loaded on this server. "
            "Loaded models: ['resnet', 'cnn'].",
        ),
    },
)
async def predict_with_model(
    model_type: Annotated[
        ModelType,
        PathParam(description="Which model to use: 'resnet', 'cnn', or 'baseline'."),
    ],
    file: Annotated[
        UploadFile,
        File(description="A JPEG or PNG photo of a single leaf."),
    ],
    top_k: Annotated[
        int,
        Query(
            ge=1,
            le=NUM_CLASSES,
            description="How many candidate classes to return.",
        ),
    ] = 3,
) -> PredictResponse:
    """Classify the uploaded leaf image with the model named in the path.

    Identical contract to ``POST /predict`` except the model is chosen
    explicitly. Returns **404** if ``model_type`` is not one of the supported
    architectures, and **503** if it is supported but no checkpoint was loaded
    for it on this server.
    """
    predictor = _require_predictor(model_type)
    image_bytes = await _read_image_upload(file)
    return _predict_with_topk(predictor, image_bytes, k=top_k)
