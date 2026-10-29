"""Download and preprocess the MNIST handwritten-digit dataset.

MNIST is read from the official IDX files (not Keras). Pixel values are scaled
to [0, 1] and images are flattened to 784 features so they match the network.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data" / "mnist"

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Google Cloud mirror of the original MNIST files (Yann LeCun's site is often down).
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049


def _download(filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    if path.exists():
        return path
    url = BASE_URL + filename
    urllib.request.urlretrieve(url, path)
    return path


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != IMAGE_MAGIC:
            raise ValueError(f"Unexpected image magic number {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
        return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != LABEL_MAGIC:
            raise ValueError(f"Unexpected label magic number {magic} in {path}")
        return np.frombuffer(handle.read(), dtype=np.uint8).copy()


def load_raw_mnist(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return original train/test arrays: images uint8 (N, 28, 28), labels uint8 (N,)."""
    paths = {key: _download(name, data_dir) for key, name in MNIST_FILES.items()}
    x_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    x_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])
    return x_train, y_train, x_test, y_test


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Scale pixels to [0, 1] and flatten 28x28 images to (N, 784)."""
    flat = images.reshape(images.shape[0], -1).astype(np.float64)
    return flat / 255.0


def train_val_test_split(
    seed: int = 42,
    val_size: int = 6000,
    data_dir: Path = DATA_DIR,
) -> dict[str, np.ndarray]:
    """Official 10k test set held out; 6k of the 60k training images used for validation."""
    x_train_full, y_train_full, x_test, y_test = load_raw_mnist(data_dir)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(x_train_full.shape[0])
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_train = preprocess_images(x_train_full[train_idx])
    y_train = y_train_full[train_idx].astype(np.int64)
    x_val = preprocess_images(x_train_full[val_idx])
    y_val = y_train_full[val_idx].astype(np.int64)
    x_test = preprocess_images(x_test)
    y_test = y_test.astype(np.int64)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }
