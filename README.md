# Applied ML — Computer Vision

A computer-vision project that classifies leaf photos into one of 38
[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
classes (plant species + disease, or `healthy`).

This repo contains:

* `src/models/resnet_transfer.py` — a ResNet-18 transfer-learning model (our best model).
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

The API loads a single checkpoint at startup, chosen by `MODEL_TYPE`. Drop your
trained file into the `models/` directory at one of the default paths below:

| Model type | Default path                  | File format                        |
|------------|-------------------------------|------------------------------------|
| `resnet`   | `models/resnet18_best.pt`     | PyTorch checkpoint dict (`.pt`)    |
| `cnn`      | `models/cnn_scratch.pth`      | PyTorch checkpoint dict (.pth)     |
| `baseline` | `models/baseline.joblib`      | `joblib.dump`'d sklearn estimator  |

The `models/` directory is gitignored, so checkpoints are not committed.

You can override the type and path with environment variables:

| Variable           | Default                  | Meaning                                   |
|--------------------|--------------------------|-------------------------------------------|
| `MODEL_TYPE`       | `cnn`                    | `resnet`, `cnn`, or `baseline`            |
| `MODEL_PATH`       | matches `MODEL_TYPE`     | Path to the checkpoint file               |
| `MAX_UPLOAD_BYTES` | `10485760` (10 MiB)      | Maximum request size for `/predict`       |

## 3. Launch the API

```bash
uv run uvicorn app.main:app --reload
```

(or `uvicorn app.main:app --reload` inside an activated venv).

The server loads **every** model whose checkpoint exists in `models/` at
startup (ResNet, CNN, and/or baseline) so you can compare them at runtime
without restarting. The service listens on <http://127.0.0.1:8000>.

* **Landing page (browser UI):** <http://127.0.0.1:8000/> — upload a leaf and
  pick which model classifies it (one card per loaded model)
* Interactive Swagger docs: <http://127.0.0.1:8000/docs>
* ReDoc:                    <http://127.0.0.1:8000/redoc>
* OpenAPI JSON:             <http://127.0.0.1:8000/openapi.json>

If no checkpoint exists at the expected path the API still starts, but
`/predict` returns `503 Service Unavailable` until you provide one.

## 4. Make a request

```bash
# Default model (whatever MODEL_TYPE env var picks; falls back to first loaded):
curl -X POST "http://127.0.0.1:8000/predict?top_k=3" \
     -F "file=@path/to/leaf.jpg"

# Explicit model:
curl -X POST "http://127.0.0.1:8000/predict/resnet?top_k=3" \
     -F "file=@path/to/leaf.jpg"
curl -X POST "http://127.0.0.1:8000/predict/cnn?top_k=3" \
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
    "model_type": "resnet",
    "checkpoint_path": "models/resnet18_best.pt",
    "num_classes": 38,
    "input_size": [224, 224],
    "device": "cpu"
  }
}
```

## 5. Endpoints

| Method | Path                       | Purpose                                                |
|--------|----------------------------|--------------------------------------------------------|
| `GET`  | `/`                        | Landing page: upload form + model picker (browser UI). |
| `GET`  | `/health`                  | Liveness check; lists all loaded models.               |
| `GET`  | `/classes`                 | List every PlantVillage label the model can predict.   |
| `GET`  | `/model`                   | Metadata about the default model.                      |
| `POST` | `/predict`                 | Classify a leaf image with the default model.          |
| `POST` | `/predict/{model_type}`    | Classify a leaf image with `resnet`, `cnn`, or `baseline`. |

## 6. Repository layout

```
app/                  FastAPI service
  classes.py          38 PlantVillage labels
  config.py           Env-var driven config
  main.py             Endpoints and lifespan
  model_loader.py     Loads ResNet / CNN / baseline; unified Predictor
  preprocessing.py    Image bytes -> tensor / feature vector
  schemas.py          Pydantic request/response models
src/                  Training-side code
  data/               Dataset download + Dataset class
  models/             Model architectures (ResNet-18, CNN, logistic baseline)
models/               Trained checkpoints (gitignored)
```

## 7. Model performance

The deployed ResNet-18 transfer model is evaluated on a held-out test split with
`src/training/evaluate.py`:

```bash
.venv/Scripts/python -c "from src.training.evaluate import evaluate; evaluate('models/resnet18_best.pt', 'data/raw')"
```

| Metric                                   | Value   |
|------------------------------------------|---------|
| Test accuracy                            | **92.84%** |
| Test set size                            | 8,146 images |
| Random baseline (always predict majority class) | 10.14% (826 / 8,146, *Orange — Citrus greening*) |

The model performs far above the majority-class baseline (92.84% vs 10.14%),
with a weighted F1 of 0.929 across all 38 classes.
