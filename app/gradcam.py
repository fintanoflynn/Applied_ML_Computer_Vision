"""Grad-CAM for the convolutional models (ResNet-18 transfer, CNN-from-scratch).

Grad-CAM (Selvaraju et al., 2017) highlights *where* a CNN looked to reach a
class decision. For a chosen convolutional layer it:

1. captures the layer's activation maps on a forward pass, and the gradient of
   the target class score w.r.t. those maps on a backward pass;
2. global-average-pools the gradients into a per-channel importance weight;
3. forms a weighted sum of the activation maps, keeps the positive part
   (``ReLU``), and upsamples it to the input resolution.

We use it to check the models attend to leaf lesions rather than the uniform
studio background — the generalisation concern flagged in the project proposal
(Section 2 / Section 7 distribution-shift risk). It is implemented by hand
(rather than via a third-party package) so the mechanism is explicit and adds
no extra dependency beyond what the project already uses.

The baseline logistic-regression model has no convolutional structure, so
Grad-CAM does not apply to it.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map an array in ``[0, 1]`` to RGB (``[0, 1]``) with a jet-like colormap.

    A small self-contained approximation of the classic "jet" colormap so the
    serving path needs no matplotlib dependency. Low values map to blue, mid to
    green, high to red — the conventional heatmap reading.
    """
    x = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def compute_gradcam(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, int]:
    """Compute a Grad-CAM heatmap for a single image.

    Args:
        model: the CNN, already loaded with trained weights.
        target_layer: the convolutional module to explain (its output is the
            feature map Grad-CAM weights). Use the last conv stage for the most
            class-discriminative, still-localised map.
        input_tensor: a ``(1, 3, H, W)`` tensor, preprocessed and normalised
            exactly as the model expects at inference time.
        class_idx: which class to explain. ``None`` explains the model's own
            top prediction (the usual "why did you say that?" question).
        device: ``"cpu"`` or ``"cuda"``.

    Returns:
        ``(heatmap, class_idx)`` where ``heatmap`` is an ``(H, W)`` float array
        in ``[0, 1]`` aligned to the input image, and ``class_idx`` is the
        class that was explained.
    """
    model.eval()
    # Detach + require grad so gradients reach the activation even when the
    # backbone's conv weights are frozen (the ResNet head-only fine-tune leaves
    # every conv layer with requires_grad=False; without a grad-requiring input
    # the activation would be detached from the graph and backward would fail).
    input_tensor = input_tensor.detach().to(device).requires_grad_(True)

    captured: dict[str, torch.Tensor] = {}

    def _forward_hook(_module, _inputs, output: torch.Tensor) -> None:
        captured["activation"] = output
        # Register a hook on the activation tensor to grab its gradient during
        # the backward pass (module backward hooks are finicky across versions).
        output.register_hook(lambda grad: captured.__setitem__("gradient", grad))

    handle = target_layer.register_forward_hook(_forward_hook)
    try:
        logits = model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    finally:
        handle.remove()

    activation = captured["activation"].detach()[0]   # (C, h, w)
    gradient = captured["gradient"].detach()[0]        # (C, h, w)

    weights = gradient.mean(dim=(1, 2))                # (C,) channel importance
    cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))  # (h, w)

    cam = cam - cam.min()
    peak = cam.max()
    if peak > 0:
        cam = cam / peak

    h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
    cam = F.interpolate(
        cam[None, None], size=(h_in, w_in), mode="bilinear", align_corners=False
    )[0, 0]
    return cam.cpu().numpy(), class_idx


def overlay_heatmap(base_rgb: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> bytes:
    """Blend a Grad-CAM heatmap over the (display) image and return PNG bytes.

    ``base_rgb`` must be the image at the *same* spatial size the heatmap was
    computed at (i.e. after the model's resize/crop), so the overlay lines up
    pixel-for-pixel.
    """
    h, w = heatmap.shape
    base = base_rgb.convert("RGB")
    if base.size != (w, h):
        base = base.resize((w, h))

    coloured = _jet_colormap(heatmap) * 255.0          # (h, w, 3) in [0, 255]
    coloured = coloured.astype(np.float32)
    base_arr = np.asarray(base, dtype=np.float32)

    blended = (1.0 - alpha) * base_arr + alpha * coloured
    blended = blended.clip(0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(blended).save(buffer, format="PNG")
    return buffer.getvalue()
