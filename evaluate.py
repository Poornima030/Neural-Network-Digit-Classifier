"""Evaluate saved NumPy weights on held-out MNIST data. Metrics are computed from predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import train_val_test_split
from model import NeuralNetwork
from train import DEFAULT_WEIGHTS, accuracy


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def per_class_metrics(matrix: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for digit in range(matrix.shape[0]):
        tp = matrix[digit, digit]
        fp = matrix[:, digit].sum() - tp
        fn = matrix[digit, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(matrix[digit, :].sum())
        rows.append(
            {
                "digit": digit,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
    return rows


def evaluate(
    weights_path: Path = DEFAULT_WEIGHTS,
    seed: int = 42,
    val_size: int = 6000,
) -> dict:
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"No saved weights at {weights_path}. Train first with: python train.py"
        )

    split = train_val_test_split(seed=seed, val_size=val_size)
    network = NeuralNetwork.load(weights_path)
    y_pred = network.predict(split["x_test"])
    y_true = split["y_test"]
    matrix = confusion_matrix(y_true, y_pred)
    class_rows = per_class_metrics(matrix)

    results = {
        "train_acc": accuracy(network, split["x_train"], split["y_train"]),
        "val_acc": accuracy(network, split["x_val"], split["y_val"]),
        "test_acc": float(np.mean(y_pred == y_true)),
        "confusion_matrix": matrix,
        "per_class": class_rows,
    }
    return results


def print_report(results: dict) -> None:
    print(f"Train accuracy: {results['train_acc']:.4f}")
    print(f"Validation accuracy: {results['val_acc']:.4f}")
    print(f"Test accuracy: {results['test_acc']:.4f}")
    print("\nConfusion matrix (rows = true digit, columns = predicted):")
    print(results["confusion_matrix"])
    print("\nPer-class metrics on the 10,000-image test set:")
    print(f"{'digit':>5}  {'precision':>10}  {'recall':>8}  {'f1':>8}  {'support':>8}")
    for row in results["per_class"]:
        print(
            f"{row['digit']:5d}  {row['precision']:10.4f}  {row['recall']:8.4f}  "
            f"{row['f1']:8.4f}  {row['support']:8d}"
        )


def plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_title("MNIST test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved MNIST weights.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG path for the confusion matrix.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = evaluate(weights_path=args.weights, seed=args.seed)
    print_report(report)
    if args.plot is not None:
        plot_confusion_matrix(report["confusion_matrix"], args.plot)
        print(f"Wrote {args.plot}")
