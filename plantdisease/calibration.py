"""Temperature scaling (Guo et al., 2017, "On Calibration of Modern Neural Networks").

A single scalar ``T`` divides the logits before the softmax. ``T > 1`` softens
over-confident predictions without changing which class wins, so accuracy is
unchanged while the confidence becomes more honest. ``T`` is fitted on the
validation split by minimising the negative log-likelihood.
"""

from __future__ import annotations

import math

import numpy as np

from plantdisease.metrics import negative_log_likelihood


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale softmax probabilities as if their logits had been divided by ``temperature``.

    ``log(p)`` equals the logits up to a per-row constant, which the softmax ignores.
    """
    if temperature == 1.0:
        return probs
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_temperature(probs: np.ndarray, y_true: np.ndarray, low: float = 0.05, high: float = 20.0) -> float:
    """Golden-section search for the NLL-minimising temperature (NLL is unimodal in log T)."""
    a, b = math.log(low), math.log(high)
    ratio = (math.sqrt(5) - 1) / 2

    def loss(log_t: float) -> float:
        return negative_log_likelihood(apply_temperature(probs, math.exp(log_t)), y_true)

    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = loss(d)
    return float(math.exp((a + b) / 2))
