# Applied ML — Computer Vision

A computer-vision project that classifies leaf photos into one of 38
[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
classes (plant species + disease, or `healthy`).

This repo contains:

* `src/models/resnet_transfer.py` — a ResNet-18 transfer-learning model.
* `src/models/cnn_scratch.py` — a small CNN trained from scratch in PyTorch.
* `src/models/base_line.py` — a logistic-regression baseline on 32x32 grayscale.
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

## 2. Download trained models

Download the models from Google Drive:
[Download model checkpoints](https://drive.google.com/drive/folders/1ZpwCm-rw07Xf7l6UxsAXU_bi8rX67wjo?usp=drive_link)

After downloading, place the files in the `models/` directory with these exact names:

| Model type | Required filename |
|------------|-------------------|
| `resnet`   | `models/resnet18_best.pt` |
| `cnn`      | `models/cnn_scratch.pth` |
| `baseline` | `models/baseline.joblib` |

There is two trained model that can be found in the google drive relating to the ResNet-18. 
To be able to use either, you must download the prefer option, and then rename it according to the "Required Filename" as shown above. 


## 3. Provide a trained model checkpoint

At startup, the API checks the `models/` directory and loads every checkpoint that exists at the expected default path. `MODEL_TYPE` only controls which loaded model is used as the default for `/predict`. Drop your trained file into the `models/` directory at one of the default paths below:

| Model type | Default path                  | File format                        |
|------------|-------------------------------|------------------------------------|
| `resnet`   | `models/resnet18_best.pt`     | PyTorch checkpoint dict (`.pt`)    |
| `cnn`      | `models/cnn_scratch.pth`      | PyTorch checkpoint dict (.pth)     |
| `baseline` | `models/baseline.joblib`      | `joblib.dump`'d sklearn estimator  |

The `models/` directory is gitignored, so checkpoints are not committed.
They must be downloaded separately from the Google Drive link above.

You can override the type and path with environment variables:

| Variable           | Default                  | Meaning                                   |
|--------------------|--------------------------|-------------------------------------------|
| `MODEL_TYPE`       | `cnn`                    | `resnet`, `cnn`, or `baseline`            |
| `MODEL_PATH`       | matches `MODEL_TYPE`     | Path to the checkpoint file               |
| `MAX_UPLOAD_BYTES` | `10485760` (10 MiB)      | Maximum request size for `/predict`       |

## 4. Launch the API

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

## 5. Make a request

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

## 6. Endpoints

| Method | Path                       | Purpose                                                |
|--------|----------------------------|--------------------------------------------------------|
| `GET`  | `/`                        | Landing page: upload form + model picker (browser UI). |
| `GET`  | `/health`                  | Liveness check; lists all loaded models.               |
| `GET`  | `/classes`                 | List every PlantVillage label the model can predict.   |
| `GET`  | `/model`                   | Metadata about the default model.                      |
| `POST` | `/predict`                 | Classify a leaf image with the default model.          |
| `POST` | `/predict/{model_type}`    | Classify a leaf image with `resnet`, `cnn`, or `baseline`. |
| `POST` | `/gradcam/{model_type}`    | Grad-CAM overlay (PNG) explaining a `resnet`/`cnn` prediction. |

## 7. Grad-CAM explanations (model interpretability)

The convolutional models (`resnet`, `cnn`) expose a Grad-CAM endpoint that
overlays a heatmap on the leaf, showing which regions drove the prediction.
This is integrated into the **live demo** (a *Show Grad-CAM* checkbox on each
model card), not just a notebook figure. The baseline (logistic regression) has
no convolutional feature maps, so Grad-CAM does not apply to it.

```bash
# Returns an image/png overlay; the explained class is in the X-Predicted-Class header.
curl -X POST "http://127.0.0.1:8000/gradcam/resnet" \
     -F "file=@path/to/leaf.jpg" -o gradcam.png -D -
```

To generate the static figure used in the report (tests whether the model
attends to lesions vs. the studio background — the generalisation concern from
the proposal):

```bash
.venv/Scripts/python -m src.explain.gradcam_report --model resnet --num 8
# -> figures/gradcam/resnet_gradcam.png
```

## 8. Evidence of model performance above random guessing

The deployed ResNet-18 transfer model is evaluated on a held-out test split with
`src/training/evaluate.py`:

```bash
.venv/Scripts/python -c "from src.training.evaluate import evaluate; evaluate('models/resnet18_best.pt', 'data/raw')"
```

| Metric                                   | Value   |
|------------------------------------------|---------|
| Test accuracy                            | **92.84%** |
| Test set size                            | 8,146 images |
| Majority-class baseline accuracy         | **10.14%** (826 / 8,146, *Orange — Citrus greening*) |


The model performs far above the majority-class baseline (94.66% vs 10.14%),
with a weighted F1 of 0.929 across all 38 classes.

### CNN from scratch performance

The CNN from scratch model was evaluated on the validation split.

| Metric | Value |
|--------|-------|
| Test accuracy | **98.92%** |
| Test set size | 8,146 images |
| Majority-class baseline accuracy | **10.14%** |

The CNN performs above the majority-class baseline: 99.42% vs 10.14%.

### Baseline (unbalanced!) cross-validation

The PCA + logistic-regression baseline (`src/models/base_line.py`, 50 PCA
components on 32×32 grayscale) was validated with **5-fold stratified
cross-validation** on the training split. PCA is re-fit inside each fold (via a
scikit-learn `Pipeline`) so the validation rows never leak into the PCA basis.

```bash
.venv/Scripts/python -c "from src.models.base_line import RegressionPCA; m=RegressionPCA(n_components=50); m.load_data(); m.cross_validate(n_splits=5)"
```

| Metric (mean ± std over 5 folds) | Value |
|----------------------------------|-------|
| Accuracy                         | **0.4257 ± 0.0026** |
| Macro-F1                         | **0.3110 ± 0.0033** |
| Training set size                | 43,444 images |

The tight standard deviation (~0.003) shows the estimate is stable across folds.
Accuracy (42.6%) sits well above random guessing (1/38 ≈ 2.6%), confirming the
baseline learns real signal. The gap between accuracy and macro-F1 reflects
class imbalance, the baseline scores higher on the larger classes than on the
rarer ones.

## LLM declaration

Claude Opus 4.7 was used to structure and systemize this README's
documentation. All content is our own.
