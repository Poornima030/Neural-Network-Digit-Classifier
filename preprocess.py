"""Turn a user drawing into the same (1, 784) vector used during training."""

from __future__ import annotations

import numpy as np
from PIL import Image

MNIST_SIZE = 28
INNER_SIZE = 20
INK_THRESHOLD = 0.08


def _to_grayscale_on_black(image_data: np.ndarray) -> np.ndarray:
    """RGBA/RGB canvas pixels -> float grayscale in [0, 1] on a black background."""
    array = np.asarray(image_data, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA image array from the canvas.")

    rgb = array[..., :3]
    if array.shape[2] == 4:
        alpha = array[..., 3:4] / 255.0
        rgb = rgb * alpha

    gray = rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140
    if gray.max() > 1.5:
        gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0)


def is_empty_drawing(image_data: np.ndarray | None, json_data: dict | None = None) -> bool:
    """True when the user has not drawn enough ink to classify."""
    if json_data is not None:
        objects = json_data.get("objects") if isinstance(json_data, dict) else None
        if objects is not None and len(objects) == 0:
            return True
    if image_data is None:
        return True
    try:
        gray = _to_grayscale_on_black(image_data)
    except (ValueError, IndexError):
        return True
    return float(gray.max()) < INK_THRESHOLD


def _center_and_resize(gray: np.ndarray) -> np.ndarray:
    """Crop to the ink, fit it in a 20x20 box, then pad to 28x28 like MNIST."""
    ink = gray > INK_THRESHOLD
    if not np.any(ink):
        return np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.float64)

    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = gray[rmin : rmax + 1, cmin : cmax + 1]

    height, width = cropped.shape
    scale = INNER_SIZE / max(height, width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    resized = np.array(
        Image.fromarray((cropped * 255.0).astype(np.uint8), mode="L").resize(
            (new_w, new_h),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float64,
    ) / 255.0

    canvas = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.float64)
    top = (MNIST_SIZE - new_h) // 2
    left = (MNIST_SIZE - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def drawing_to_features(image_data: np.ndarray) -> np.ndarray:
    """Return a (1, 784) float array ready for NeuralNetwork.predict_proba."""
    gray = _to_grayscale_on_black(image_data)
    centered = _center_and_resize(gray)
    return centered.reshape(1, MNIST_SIZE * MNIST_SIZE)
