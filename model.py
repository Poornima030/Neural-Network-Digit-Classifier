"""From-scratch feedforward neural network using only NumPy.

Architecture
------------
    784 pixels  ->  128 ReLU  ->  64 ReLU  ->  10 softmax (digits 0-9)

Each layer computes Z = A_prev @ W + b, then an activation. Training uses
mini-batch gradient descent: the forward pass produces class probabilities,
cross-entropy measures the error, backpropagation applies the chain rule to
get dW and db, and weights are stepped downhill.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

INPUT_SIZE = 784
HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 64
OUTPUT_SIZE = 10


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU(z) = max(0, z). Hidden layers use this so the model can learn curves."""
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """d(ReLU)/dz is 1 for z > 0 and 0 otherwise."""
    return (z > 0).astype(np.float64)


def softmax(z: np.ndarray) -> np.ndarray:
    """Turn raw scores into probabilities that sum to 1 for each example.

    Subtracting the row max keeps exp() numerically stable.
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(labels: np.ndarray, num_classes: int = OUTPUT_SIZE) -> np.ndarray:
    """Convert integer labels of shape (N,) to a (N, 10) one-hot matrix."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    encoded = np.zeros((labels.size, num_classes), dtype=np.float64)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded


def cross_entropy(probabilities: np.ndarray, labels_one_hot: np.ndarray) -> float:
    """Mean categorical cross-entropy: -mean(sum(y * log(y_hat)))."""
    eps = 1e-12
    clipped = np.clip(probabilities, eps, 1.0)
    return float(-np.mean(np.sum(labels_one_hot * np.log(clipped), axis=1)))


class NeuralNetwork:
    """Three-layer MLP with ReLU hidden units and a softmax output."""

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        seed: int = 42,
    ) -> None:
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.params = self._init_params()

    def _he_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        """He initialization: variance 2/fan_in, a good default for ReLU."""
        std = np.sqrt(2.0 / fan_in)
        return self.rng.normal(0.0, std, size=(fan_in, fan_out))

    def _init_params(self) -> dict[str, np.ndarray]:
        return {
            "W1": self._he_weights(INPUT_SIZE, HIDDEN1_SIZE),
            "b1": np.zeros((1, HIDDEN1_SIZE), dtype=np.float64),
            "W2": self._he_weights(HIDDEN1_SIZE, HIDDEN2_SIZE),
            "b2": np.zeros((1, HIDDEN2_SIZE), dtype=np.float64),
            "W3": self._he_weights(HIDDEN2_SIZE, OUTPUT_SIZE),
            "b3": np.zeros((1, OUTPUT_SIZE), dtype=np.float64),
        }

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute class probabilities and cache values needed for backprop.

        Shapes
        ------
        x:  (N, 784)
        Z1: (N, 128)   A1 = ReLU(Z1)
        Z2: (N, 64)    A2 = ReLU(Z2)
        Z3: (N, 10)    A3 = softmax(Z3)
        """
        p = self.params
        z1 = x @ p["W1"] + p["b1"]
        a1 = relu(z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = relu(z2)
        z3 = a2 @ p["W3"] + p["b3"]
        a3 = softmax(z3)
        cache = {"X": x, "Z1": z1, "A1": a1, "Z2": z2, "A2": a2, "Z3": z3, "A3": a3}
        return a3, cache

    def loss(self, x: np.ndarray, labels: np.ndarray) -> float:
        probabilities, _ = self.forward(x)
        return cross_entropy(probabilities, one_hot(labels))

    def backward(self, cache: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, np.ndarray]:
        """Chain-rule gradients of mean cross-entropy w.r.t. every W and b.

        For softmax + cross-entropy, the output-layer gradient simplifies to
        (predicted_probabilities - one_hot_labels) / N.
        """
        p = self.params
        n = cache["X"].shape[0]
        y = one_hot(labels)

        dz3 = (cache["A3"] - y) / n
        dw3 = cache["A2"].T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ p["W3"].T
        dz2 = da2 * relu_derivative(cache["Z2"])
        dw2 = cache["A1"].T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ p["W2"].T
        dz1 = da1 * relu_derivative(cache["Z1"])
        dw1 = cache["X"].T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {
            "W1": dw1,
            "b1": db1,
            "W2": dw2,
            "b2": db2,
            "W3": dw3,
            "b3": db3,
        }

    def update(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        """Gradient descent: parameter <- parameter - learning_rate * gradient."""
        for name in self.params:
            self.params[name] = self.params[name] - learning_rate * grads[name]

    def train_batch(self, x: np.ndarray, labels: np.ndarray, learning_rate: float) -> float:
        probabilities, cache = self.forward(x)
        loss_value = cross_entropy(probabilities, one_hot(labels))
        grads = self.backward(cache, labels)
        self.update(grads, learning_rate)
        return loss_value

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probabilities, _ = self.forward(x)
        return probabilities

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            **self.params,
            architecture=np.array([INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE]),
        )

    @classmethod
    def load(cls, path: str | Path) -> NeuralNetwork:
        data = np.load(path)
        expected = np.array([INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE])
        if "architecture" in data and not np.array_equal(data["architecture"], expected):
            raise ValueError(
                f"Saved architecture {data['architecture']} does not match {expected}."
            )
        network = cls()
        network.params = {
            "W1": np.array(data["W1"], dtype=np.float64),
            "b1": np.array(data["b1"], dtype=np.float64),
            "W2": np.array(data["W2"], dtype=np.float64),
            "b2": np.array(data["b2"], dtype=np.float64),
            "W3": np.array(data["W3"], dtype=np.float64),
            "b3": np.array(data["b3"], dtype=np.float64),
        }
        return network
