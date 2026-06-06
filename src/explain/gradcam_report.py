"""Generate a Grad-CAM figure for the report / poster.

The deployed demo shows Grad-CAM live (see ``app/main.py`` ``/gradcam``); this
script produces the *static* figure for the write-up. Its purpose is to test the
hypothesis from the proposal (Sections 2 and 7): do the models attend to the
leaf lesions, or do they latch onto the uniform studio background?

For each sampled image it lays out two panels — the original leaf and the
Grad-CAM overlay — annotated with the model's prediction and confidence. Running
it for several classes (and, ideally, a few out-of-distribution field photos)
gives the visual evidence for the "where does the model look?" discussion.

Usage (from the repo root, with a trained checkpoint in ``models/``)::

    python -m src.explain.gradcam_report --model resnet --num 8
    python -m src.explain.gradcam_report --model cnn --dir path/to/leaf_folder
    python -m src.explain.gradcam_report --model resnet --images a.jpg b.jpg

The checkpoints are loaded exactly as the API loads them, so the overlays match
what the live demo produces.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Allow running as a script from the repo root (so ``app`` / ``src`` import).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.classes import CLASS_NAMES, humanise  # noqa: E402
from app.model_loader import Predictor, _load_cnn, _load_resnet  # noqa: E402
from app.preprocessing import display_image_for_cnn, display_image_for_resnet  # noqa: E402

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Default PlantVillage location used elsewhere in the repo.
_DEFAULT_DATA_ROOT = _PROJECT_ROOT / "data" / "raw" / "plantvillage" / "plantvillage dataset" / "color"
_DEFAULT_CHECKPOINTS = {
    "resnet": _PROJECT_ROOT / "models" / "resnet18_best.pt",
    "cnn": _PROJECT_ROOT / "models" / "cnn_scratch.pth",
}


def _load_predictor(model_type: str, checkpoint: Path) -> Predictor:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No {model_type} checkpoint at {checkpoint}. Download it into "
            f"models/ (see README) or pass --checkpoint."
        )
    if model_type == "resnet":
        return _load_resnet(checkpoint)
    if model_type == "cnn":
        return _load_cnn(checkpoint)
    raise ValueError(f"Grad-CAM supports 'resnet' or 'cnn', not {model_type!r}.")


def _gather_images(images: list[str], directory: Path | None, num: int) -> list[Path]:
    """Resolve the image set: explicit paths win, else sample from a directory.

    When sampling a directory of class subfolders (PlantVillage layout), we take
    one image from each of the first ``num`` classes so the figure spans
    different diseases rather than ``num`` copies of one class.
    """
    if images:
        return [Path(p) for p in images]

    root = directory or _DEFAULT_DATA_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"No image directory at {root}. Pass --dir or --images explicitly."
        )

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    picked: list[Path] = []
    if subdirs:  # class-subfolder layout: one image per class, spread across classes
        for class_dir in subdirs:
            files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES)
            if files:
                picked.append(files[0])
            if len(picked) >= num:
                break
    else:  # flat folder of images
        picked = sorted(f for f in root.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES)[:num]

    if not picked:
        raise FileNotFoundError(f"Found no .jpg/.jpeg/.png images under {root}.")
    return picked


def _display_base(predictor: Predictor, image_bytes: bytes) -> Image.Image:
    """The image at the model's input geometry, to sit beside the overlay."""
    if predictor.model_type == "resnet":
        return display_image_for_resnet(image_bytes)
    return display_image_for_cnn(image_bytes)


def build_figure(predictor: Predictor, image_paths: list[Path], out_path: Path) -> Path:
    rows = len(image_paths)
    fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row, image_path in enumerate(image_paths):
        image_bytes = image_path.read_bytes()

        probs = predictor.predict_proba(image_bytes)
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        overlay_png, explained_idx = predictor.gradcam(image_bytes, top_idx)
        overlay = Image.open(io.BytesIO(overlay_png))
        base = _display_base(predictor, image_bytes)

        plant, condition = humanise(CLASS_NAMES[explained_idx])
        pred_label = f"{plant} — {condition} ({top_prob * 100:.1f}%)"

        axes[row, 0].imshow(base)
        axes[row, 0].set_title(image_path.name, fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"Grad-CAM: {pred_label}", fontsize=8)
        axes[row, 1].axis("off")

    fig.suptitle(f"Grad-CAM — {predictor.model_type} model", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM figure for the report.")
    parser.add_argument("--model", choices=("resnet", "cnn"), default="resnet")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override the checkpoint path.")
    parser.add_argument("--dir", type=Path, default=None, help="Sample images from this directory.")
    parser.add_argument("--images", nargs="*", default=[], help="Explicit image paths to explain.")
    parser.add_argument("--num", type=int, default=8, help="How many images to sample when using --dir.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path (default: figures/gradcam/<model>_gradcam.png).",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or _DEFAULT_CHECKPOINTS[args.model]
    predictor = _load_predictor(args.model, checkpoint)

    image_paths = _gather_images(args.images, args.dir, args.num)
    out_path = args.out or (_PROJECT_ROOT / "figures" / "gradcam" / f"{args.model}_gradcam.png")

    saved = build_figure(predictor, image_paths, out_path)
    print(f"Saved Grad-CAM figure with {len(image_paths)} images to {saved}")


if __name__ == "__main__":
    main()