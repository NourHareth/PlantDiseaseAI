"""Evaluation metrics implemented with NumPy only (no scikit-learn dependency)."""

from __future__ import annotations

from math import comb

import numpy as np


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Accuracy, macro F1, per-class precision/recall/F1 and the confusion matrix."""
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (np.asarray(y_true), np.asarray(y_pred)), 1)
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i, i]
        support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": support,
        }
    present = [m for m in per_class.values() if m["support"]]
    return {
        "accuracy": round(float((np.asarray(y_true) == np.asarray(y_pred)).mean()), 4),
        "macro_f1": round(float(np.mean([m["f1"] for m in present])), 4) if present else 0.0,
        "num_images": int(len(y_true)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def bootstrap_accuracy_ci(
    y_true: np.ndarray, y_pred: np.ndarray, *, n_resamples: int = 2000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for accuracy."""
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_resamples, len(correct)))
    samples = correct[idx].mean(axis=1)
    alpha = (1.0 - level) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact two-sided McNemar test: do two models, scored on the *same* images, differ?

    Only images on which the models disagree carry information. Under the null
    hypothesis (equal accuracy) each such image is equally likely to favour either
    model, so the count favouring A is Binomial(n_discordant, 0.5).
    """
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Both models must be scored on the same images")
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))  # exact integers, any n
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail / 2**n) if n else 1.0}


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values: control the chance of *any* false positive
    when several comparisons are tested together."""
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))  # monotone in the sorted order
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def reliability_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    """Per-bin mean confidence, accuracy and count (equal-width confidence bins)."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    conf_mean = np.zeros(n_bins)
    acc_mean = np.zeros(n_bins)
    counts = np.bincount(index, minlength=n_bins)
    for b in range(n_bins):
        mask = index == b
        if mask.any():
            conf_mean[b] = confidence[mask].mean()
            acc_mean[b] = correct[mask].mean()
    return conf_mean, acc_mean, counts


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average |accuracy - confidence| over bins, weighted by bin size (Guo et al., 2017)."""
    conf, acc, counts = reliability_bins(probs, y_true, n_bins)
    return float(np.sum(counts / counts.sum() * np.abs(acc - conf)))


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0))))


# ---------------------------------------------------------------------------
# Out-of-distribution detection
# ---------------------------------------------------------------------------
def auroc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """Area under the ROC curve for separating in-distribution (higher score) from OOD.

    Computed as the Mann-Whitney U statistic with average ranks for ties.
    """
    scores = np.concatenate([scores_in, scores_out])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):  # average ranks over ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_in, n_out = len(scores_in), len(scores_out)
    u = ranks[:n_in].sum() - n_in * (n_in + 1) / 2.0
    return float(u / (n_in * n_out))


def fpr_at_tpr(scores_in: np.ndarray, scores_out: np.ndarray, tpr: float = 0.95) -> float:
    """False-positive rate on OOD data when ``tpr`` of in-distribution data is accepted."""
    threshold = np.quantile(scores_in, 1.0 - tpr)
    return float((scores_out >= threshold).mean())
