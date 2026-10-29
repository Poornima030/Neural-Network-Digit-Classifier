"""Unit tests for the NumPy network. Run: python -m unittest discover -s tests"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from model import (
    HIDDEN1_SIZE,
    HIDDEN2_SIZE,
    INPUT_SIZE,
    OUTPUT_SIZE,
    NeuralNetwork,
    cross_entropy,
    one_hot,
    relu,
    softmax,
)
from preprocess import drawing_to_features, is_empty_drawing


class ActivationTests(unittest.TestCase):
    def test_relu_and_softmax(self) -> None:
        z = np.array([[-1.0, 0.0, 2.0]])
        np.testing.assert_array_equal(relu(z), np.array([[0.0, 0.0, 2.0]]))
        probs = softmax(np.array([[1.0, 1.0, 1.0]]))
        np.testing.assert_allclose(probs, np.full((1, 3), 1 / 3), atol=1e-12)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0)


class NetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        self.net = NeuralNetwork(seed=0)
        self.x = self.rng.random((4, INPUT_SIZE))
        self.y = np.array([0, 1, 2, 3])

    def test_forward_shapes(self) -> None:
        probs, cache = self.net.forward(self.x)
        self.assertEqual(probs.shape, (4, OUTPUT_SIZE))
        self.assertEqual(cache["A1"].shape, (4, HIDDEN1_SIZE))
        self.assertEqual(cache["A2"].shape, (4, HIDDEN2_SIZE))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-10)

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.npz"
            self.net.save(path)
            loaded = NeuralNetwork.load(path)
            original = self.net.predict_proba(self.x)
            restored = loaded.predict_proba(self.x)
            np.testing.assert_allclose(original, restored)

    def test_parameter_update_changes_weights(self) -> None:
        before = self.net.params["W1"].copy()
        self.net.train_batch(self.x, self.y, learning_rate=0.1)
        self.assertFalse(np.allclose(before, self.net.params["W1"]))

    def test_loss_decreases_on_tiny_dataset(self) -> None:
        x = self.x[:2]
        y = self.y[:2]
        start = self.net.loss(x, y)
        for _ in range(80):
            self.net.train_batch(x, y, learning_rate=0.3)
        end = self.net.loss(x, y)
        self.assertLess(end, start)


class GradientCheckTests(unittest.TestCase):
    def test_numerical_gradients_match_backprop(self) -> None:
        rng = np.random.default_rng(1)
        net = NeuralNetwork(seed=1)
        x = rng.random((3, INPUT_SIZE))
        y = np.array([1, 4, 7])
        _, cache = net.forward(x)
        analytic = net.backward(cache, y)

        epsilon = 1e-5
        # Check a random subset of weights so the test stays fast.
        for name in ("W1", "b2", "W3"):
            param = net.params[name]
            sampled = list(zip(*np.unravel_index(rng.choice(param.size, size=6, replace=False), param.shape)))
            for index in sampled:
                original = param[index]
                param[index] = original + epsilon
                plus = cross_entropy(net.forward(x)[0], one_hot(y))
                param[index] = original - epsilon
                minus = cross_entropy(net.forward(x)[0], one_hot(y))
                param[index] = original
                numerical = (plus - minus) / (2 * epsilon)
                self.assertAlmostEqual(
                    numerical,
                    analytic[name][index],
                    places=5,
                    msg=f"Mismatch at {name}{index}",
                )


class PreprocessTests(unittest.TestCase):
    def test_empty_canvas_detected(self) -> None:
        blank = np.zeros((280, 280, 4), dtype=np.uint8)
        blank[..., 3] = 255
        self.assertTrue(is_empty_drawing(blank, {"objects": []}))
        self.assertTrue(is_empty_drawing(None))

    def test_drawing_to_features_shape(self) -> None:
        image = np.zeros((280, 280, 4), dtype=np.uint8)
        image[80:200, 100:180, :3] = 255
        image[..., 3] = 255
        features = drawing_to_features(image)
        self.assertEqual(features.shape, (1, 784))
        self.assertGreater(features.max(), 0.1)
        self.assertLessEqual(features.max(), 1.0)
        self.assertFalse(is_empty_drawing(image, {"objects": [{"type": "path"}]}))


if __name__ == "__main__":
    unittest.main()
