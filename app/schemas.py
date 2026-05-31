"""Pydantic schemas for request and response bodies."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Several fields in this module are conventionally named with a `model_`
# prefix (model_type, model_loaded, etc.). Pydantic v2 reserves that namespace
# for its own internals and warns about it; we opt out at the package level.
_CONFIG = ConfigDict(protected_namespaces=())


class Prediction(BaseModel):
    """A single class prediction with its model probability."""

    label: str = Field(
        ...,
        description="Raw PlantVillage class label, e.g. 'Tomato___Late_blight'.",
        examples=["Tomato___Late_blight"],
    )
    plant: str = Field(
        ...,
        description="Human-readable plant species, parsed from the label.",
        examples=["Tomato"],
    )
    condition: str = Field(
        ...,
        description="Human-readable plant condition, parsed from the label.",
        examples=["Late blight"],
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model-estimated probability for this class, in [0, 1].",
        examples=[0.873],
    )


class ModelInfo(BaseModel):
    """Metadata about the model currently loaded by the service."""

    model_config = _CONFIG

    model_type: str = Field(
        ...,
        description="Which architecture is loaded: 'cnn', 'resnet', or 'baseline'.",
        examples=["cnn"],
    )
    checkpoint_path: str = Field(
        ...,
        description="Filesystem path of the loaded checkpoint file.",
        examples=["models/cnn_scratch.pth"],
    )
    num_classes: int = Field(
        ...,
        description="Number of output classes the model can predict.",
        examples=[38],
    )
    input_size: list[int] = Field(
        ...,
        description="Expected (height, width) the API resizes inputs to before "
                    "feeding the model.",
        examples=[[256, 256]],
    )
    device: str = Field(
        ...,
        description="Compute device used at inference time, e.g. 'cpu' or 'cuda'.",
        examples=["cpu"],
    )


class PredictResponse(BaseModel):
    """Response body returned by the /predict endpoints."""

    model_config = _CONFIG

    top_prediction: Prediction = Field(
        ...,
        description="The single most likely class for the submitted image.",
    )
    top_k: list[Prediction] = Field(
        ...,
        description="The top-k predictions, sorted from most to least likely.",
    )
    model: ModelInfo = Field(
        ...,
        description="Metadata about the model that produced this prediction.",
    )


class ClassesResponse(BaseModel):
    """Response body for GET /classes."""

    num_classes: int = Field(..., description="Total number of supported classes.")
    classes: list[str] = Field(
        ...,
        description="All raw PlantVillage labels the model can predict, in the "
                    "same index order the model uses internally.",
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    model_config = _CONFIG

    status: str = Field(
        ...,
        description="'ok' if the service is up and a model is loaded.",
        examples=["ok"],
    )
    model_loaded: bool = Field(
        ...,
        description="True if a model checkpoint was successfully loaded at startup.",
    )
    model_type: str | None = Field(
        None,
        description="Which architecture is currently loaded, if any.",
    )


class ErrorResponse(BaseModel):
    """Body returned for non-2xx responses.

    The concrete ``detail`` text differs per status code — see the per-status
    examples on each endpoint's response table for what to expect.
    """

    detail: str = Field(
        ...,
        description="Human-readable explanation of what went wrong. The exact "
                    "wording is specific to the status code returned.",
    )
