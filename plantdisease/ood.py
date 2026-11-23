"""Out-of-distribution (OOD) detection: "is this even a tomato leaf photo?"

Three scores, all oriented so that *higher = more in-distribution*:

* **MSP**: maximum softmax probability (Hendrycks & Gimpel, 2017), the baseline.
* **Energy**: ``logsumexp(logits)`` (Liu et al., 2020); uses the raw logits,
  so it is not squashed by the softmax.
* **Mahalanobis**: negative distance of the penultimate features to the
  nearest class mean under a shared covariance (Lee et al., 2018).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FeatureExtractor:
    """Penultimate features and logits of a model ending in a softmax Dense layer."""

    def __init__(self, model):
        import tensorflow as tf

        dense = model.layers[-1]
        if not isinstance(dense, tf.keras.layers.Dense):
            raise ValueError("Expected the model to end with a Dense layer.")
        inputs = model.inputs[0] if len(model.inputs) == 1 else model.inputs
        self._features = tf.keras.Model(inputs, dense.input)
        self._kernel = np.asarray(dense.kernel)
        self._bias = np.asarray(dense.bias)

    def __call__(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(self._features(batch, training=False), dtype=np.float64)
        return features, features @ self._kernel + self._bias


def msp_score(probs: np.ndarray) -> np.ndarray:
    return probs.max(axis=1)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    peak = z.max(axis=1, keepdims=True)
    return temperature * (peak[:, 0] + np.log(np.exp(z - peak).sum(axis=1)))


class MahalanobisDetector:
    """Class-conditional Gaussians with a shared covariance on penultimate features."""

    def __init__(self, means: np.ndarray, precision: np.ndarray, threshold: float | None = None):
        self.means = means
        self.precision = precision
        self.threshold = threshold  # score below this is treated as OOD

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, shrinkage: float = 1e-3) -> "MahalanobisDetector":
        classes = np.unique(labels)
        means = np.stack([features[labels == c].mean(axis=0) for c in classes])
        centred = features - means[np.searchsorted(classes, labels)]
        covariance = centred.T @ centred / len(features)
        covariance += shrinkage * np.trace(covariance) / len(covariance) * np.eye(len(covariance))
        return cls(means, np.linalg.inv(covariance))

    def score(self, features: np.ndarray) -> np.ndarray:
        diffs = features[:, None, :] - self.means[None, :, :]  # (N, C, D)
        distances = np.einsum("ncd,de,nce->nc", diffs, self.precision, diffs)
        return -distances.min(axis=1)

    def calibrate_threshold(self, in_distribution_scores: np.ndarray, tpr: float = 0.95) -> float:
        """Accept ``tpr`` of in-distribution validation images."""
        self.threshold = float(np.quantile(in_distribution_scores, 1.0 - tpr))
        return self.threshold

    def save(self, path: Path) -> None:
        # float32 halves the file; distances are recomputed in float64 after loading.
        np.savez_compressed(path, means=self.means.astype(np.float32), precision=self.precision.astype(np.float32),
                            threshold=np.array(np.nan if self.threshold is None else self.threshold))

    @classmethod
    def load(cls, path: Path) -> "MahalanobisDetector":
        data = np.load(path)
        threshold = float(data["threshold"])
        return cls(data["means"].astype(np.float64), data["precision"].astype(np.float64),
                   None if np.isnan(threshold) else threshold)


def synthetic_ood_images(count: int, size: int = 128, seed: int = 0) -> np.ndarray:
    """Far-OOD images in [0, 1]: noise, flat colours, gradients and checkerboards."""
    rng = np.random.default_rng(seed)
    images = np.empty((count, size, size, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size] / size
    for i in range(count):
        kind = i % 5
        if kind == 0:
            img = rng.random((size, size, 3))
        elif kind == 1:
            img = np.clip(rng.normal(0.5, rng.uniform(0.05, 0.3), (size, size, 3)), 0, 1)
        elif kind == 2:
            img = np.broadcast_to(rng.random(3), (size, size, 3))
        elif kind == 3:
            angle = rng.uniform(0, np.pi)
            ramp = (np.cos(angle) * xx + np.sin(angle) * yy)
            ramp = (ramp - ramp.min()) / (np.ptp(ramp) + 1e-8)
            img = ramp[..., None] * rng.random(3) + (1 - ramp[..., None]) * rng.random(3)
        else:
            cells = rng.integers(2, 16)
            board = ((xx * cells).astype(int) + (yy * cells).astype(int)) % 2
            img = np.where(board[..., None] == 1, rng.random(3), rng.random(3))
        images[i] = img
    return images
