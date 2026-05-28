# Applied ML — Computer Vision

A computer-vision project that classifies leaf photos into one of 38
[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
classes (plant species + disease, or `healthy`).

This repo contains:

* `src/models/cnn_scratch.py` — a small CNN trained from scratch in PyTorch.
* `src/models/base_line.py` — a logistic-regression baseline on 64×64 grayscale.
* `app/` — a FastAPI service that exposes the trained model as a REST API.

## 1. Install dependencies

The project is managed with [`uv`](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync
```

That installs everything in `pyproject.toml` into a local `.venv/`.

If you prefer plain pip:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e .
```

## 2. Provide a trained model checkpoint

The API loads a checkpoint at startup. Drop your trained file into the
`models/` directory:

| Model type | Default path                | File format                |
|------------|-----------------------------|----------------------------|
| `cnn`      | `models/cnn_scratch.pth`    | PyTorch state-dict (`.pth`) |
| `baseline` | `models/baseline.joblib`    | `joblib.dump`'d sklearn estimator |

The `models/` directory is gitignored, so checkpoints are not committed.

You can override the paths with environment variables:

| Variable           | Default                  | Meaning                              |
|--------------------|--------------------------|--------------------------------------|
| `MODEL_TYPE`       | `cnn`                    | `cnn` or `baseline`                  |
| `MODEL_PATH`       | matches `MODEL_TYPE`     | Path to the checkpoint file          |
| `MAX_UPLOAD_BYTES` | `10485760` (10 MiB)      | Maximum request size for `/predict`  |

## 3. Launch the API

```bash
uv run uvicorn app.main:app --reload
```

(or `uvicorn app.main:app --reload` inside an activated venv).

The service listens on <http://127.0.0.1:8000>.

* Interactive Swagger docs: <http://127.0.0.1:8000/docs>
* ReDoc:                    <http://127.0.0.1:8000/redoc>
* OpenAPI JSON:             <http://127.0.0.1:8000/openapi.json>

If no checkpoint exists at the expected path the API still starts, but
`/predict` returns `503 Service Unavailable` until you provide one.

## 4. Make a request

```bash
curl -X POST "http://127.0.0.1:8000/predict?top_k=3" \
     -F "file=@path/to/leaf.jpg"
```

Response (truncated):

```json
{
  "top_prediction": {
    "label": "Tomato___Late_blight",
    "plant": "Tomato",
    "condition": "Late blight",
    "probability": 0.873
  },
  "top_k": [
    { "label": "Tomato___Late_blight",   "plant": "Tomato", "condition": "Late blight",   "probability": 0.873 },
    { "label": "Tomato___Early_blight",  "plant": "Tomato", "condition": "Early blight",  "probability": 0.082 },
    { "label": "Tomato___Septoria_leaf_spot", "plant": "Tomato", "condition": "Septoria leaf spot", "probability": 0.021 }
  ],
  "model": {
    "model_type": "cnn",
    "checkpoint_path": "models/cnn_scratch.pth",
    "num_classes": 38,
    "input_size": [256, 256],
    "device": "cpu"
  }
}
```

## 5. Endpoints

| Method | Path        | Purpose                                              |
|--------|-------------|------------------------------------------------------|
| `GET`  | `/health`   | Liveness check; reports whether a model is loaded.   |
| `GET`  | `/classes`  | List every PlantVillage label the model can predict. |
| `GET`  | `/model`    | Metadata about the currently loaded checkpoint.      |
| `POST` | `/predict`  | Classify a single leaf image (multipart upload).     |

## 6. Repository layout

```
app/                  FastAPI service
  classes.py          38 PlantVillage labels
  config.py           Env-var driven config
  main.py             Endpoints and lifespan
  model_loader.py     Loads CNN or baseline; unified Predictor
  preprocessing.py    Image bytes -> tensor / feature vector
  schemas.py          Pydantic request/response models
src/                  Training-side code
  data/               Dataset download + Dataset class
  models/             Model architectures (CNN, logistic baseline)
models/               Trained checkpoints (gitignored)
```
