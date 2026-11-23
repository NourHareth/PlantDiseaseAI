import numpy as np
import pytest

from plantdisease.calibration import apply_temperature, fit_temperature
from plantdisease.metrics import (
    auroc,
    bootstrap_accuracy_ci,
    expected_calibration_error,
    fpr_at_tpr,
    holm_adjust,
    mcnemar_exact,
    negative_log_likelihood,
    reliability_bins,
)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --- AUROC / FPR ------------------------------------------------------------------
def test_auroc_extremes_and_ties():
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert auroc(np.ones(5), np.ones(7)) == pytest.approx(0.5)


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 40).round(1), rng.normal(0, 1, 30).round(1)  # rounding creates ties
    pairwise = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert auroc(a, b) == pytest.approx(pairwise)


def test_fpr_at_tpr():
    scores_in = np.linspace(0, 1, 101)
    assert fpr_at_tpr(scores_in, np.full(10, -1.0)) == 0.0
    assert fpr_at_tpr(scores_in, np.full(10, 2.0)) == 1.0


# --- Calibration ------------------------------------------------------------------
def _overconfident_sample(true_temperature=2.0, n=6000, classes=5, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3, (n, classes))
    true_probs = _softmax(logits / true_temperature)
    labels = np.array([rng.choice(classes, p=p) for p in true_probs])
    return _softmax(logits), labels


def test_temperature_scaling_recovers_true_temperature():
    probs, labels = _overconfident_sample(true_temperature=2.0)
    temperature = fit_temperature(probs, labels)
    assert temperature == pytest.approx(2.0, rel=0.12)
    calibrated = apply_temperature(probs, temperature)
    assert expected_calibration_error(calibrated, labels) < expected_calibration_error(probs, labels)
    assert negative_log_likelihood(calibrated, labels) < negative_log_likelihood(probs, labels)


def test_temperature_keeps_the_ranking():
    probs, _ = _overconfident_sample(n=200)
    softened = apply_temperature(probs, 3.0)
    assert np.array_equal(softened.argmax(1), probs.argmax(1))
    assert np.allclose(softened.sum(1), 1.0)
    assert (softened.max(1) <= probs.max(1) + 1e-9).all()
    assert apply_temperature(probs, 1.0) is probs


def test_ece_is_zero_for_a_calibrated_predictor():
    rng = np.random.default_rng(1)
    confidence = rng.uniform(0.5, 1.0, 20000)
    correct = rng.random(20000) < confidence
    probs = np.stack([confidence, 1 - confidence], axis=1)
    labels = np.where(correct, 0, 1)
    assert expected_calibration_error(probs, labels) < 0.02
    conf, acc, counts = reliability_bins(probs, labels)
    assert counts.sum() == 20000 and len(conf) == len(acc) == 15


def test_mcnemar_exact_matches_the_binomial_formula():
    # 1 image only A gets right, 9 only B gets right, 40 both right: p = 2 * (C(10,0) + C(10,1)) / 2^10
    a = np.array([True] * 40 + [True] + [False] * 9)
    b = np.array([True] * 40 + [False] + [True] * 9)
    result = mcnemar_exact(a, b)
    assert (result["only_a"], result["only_b"]) == (1, 9)
    assert result["p_value"] == pytest.approx(22 / 1024)
    assert mcnemar_exact(b, a)["p_value"] == pytest.approx(result["p_value"])  # symmetric


def test_mcnemar_exact_edge_cases():
    same = np.array([True, False, True])
    assert mcnemar_exact(same, same) == {"only_a": 0, "only_b": 0, "p_value": 1.0}
    balanced = np.array([True, False] * 500), np.array([False, True] * 500)
    assert mcnemar_exact(*balanced)["p_value"] == pytest.approx(1.0)  # 1,000 discordant pairs, no overflow
    with pytest.raises(ValueError):
        mcnemar_exact(np.ones(3, bool), np.ones(4, bool))


def test_holm_adjustment():
    # Sorted p: 0.01 (x3), 0.02 (x2), 0.03 (x1) -> 0.03, 0.04, 0.04 (kept monotone), back in input order
    assert holm_adjust([0.03, 0.01, 0.02]).tolist() == pytest.approx([0.04, 0.03, 0.04])
    assert holm_adjust([0.5, 0.9]).tolist() == pytest.approx([1.0, 1.0])  # capped at 1
    assert holm_adjust([]).tolist() == []


def test_bootstrap_interval_contains_the_estimate():
    y = np.array([0, 1] * 100)
    pred = y.copy()
    pred[:30] = 1 - pred[:30]  # 85% accuracy
    low, high = bootstrap_accuracy_ci(y, pred)
    assert low < 0.85 < high
    assert 0.0 <= low and high <= 1.0
