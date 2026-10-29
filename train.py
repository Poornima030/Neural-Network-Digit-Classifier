"""Train the NumPy MNIST network with mini-batch gradient descent."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import train_val_test_split
from model import NeuralNetwork

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "models" / "mnist_weights.npz"


def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = rng.permutation(x.shape[0])
    batches = []
    for start in range(0, x.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        batches.append((x[batch_idx], y[batch_idx]))
    return batches


def accuracy(network: NeuralNetwork, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(network.predict(x) == y))


def train(
    epochs: int = 12,
    learning_rate: float = 0.1,
    batch_size: int = 64,
    seed: int = 42,
    val_size: int = 6000,
    weights_path: Path = DEFAULT_WEIGHTS,
) -> dict:
    split = train_val_test_split(seed=seed, val_size=val_size)
    rng = np.random.default_rng(seed)
    network = NeuralNetwork(seed=seed)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        batches = iterate_minibatches(split["x_train"], split["y_train"], batch_size, rng)
        batch_losses = []
        for x_batch, y_batch in batches:
            batch_losses.append(network.train_batch(x_batch, y_batch, learning_rate))

        train_loss = float(np.mean(batch_losses))
        val_loss = network.loss(split["x_val"], split["y_val"])
        train_acc = accuracy(network, split["x_train"], split["y_train"])
        val_acc = accuracy(network, split["x_val"], split["y_val"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}"
        )

    network.save(weights_path)
    print(f"Saved weights to {weights_path}")
    return {"network": network, "history": history, "split": split}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the from-scratch MNIST network.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        weights_path=args.weights,
    )
