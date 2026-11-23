"""Paired comparisons, the latency benchmark loop and the report table (no models needed)."""

import numpy as np
import pytest

from scripts import benchmark_speed, check_detector, compare_runs, make_report


def test_compare_counts_disagreements_and_corrects_for_multiple_tests():
    a = np.array([True] * 80 + [False] * 20)
    b = np.array([True] * 80 + [True] * 15 + [False] * 5)  # B fixes 15 of A's mistakes, breaks none
    correct = {("A", "s"): a, ("B", "s"): b, ("C", "s"): a.copy()}
    rows = compare_runs.compare(correct, [("A", "B", "s", "B vs A"), ("A", "C", "s", "C vs A")])
    first, second = rows
    assert (first["only_a"], first["only_b"], first["images"]) == (0, 15, 100)
    assert first["difference"] == pytest.approx(0.15)
    assert first["p_value"] == pytest.approx(2 / 2**15)
    assert first["p_holm"] == pytest.approx(2 * first["p_value"])  # two tests: smallest p doubled
    assert second["p_value"] == second["p_holm"] == 1.0  # identical predictions


def test_interleaved_timings_time_every_model_equally():
    calls = []
    timings = benchmark_speed.interleaved_timings(
        {"x": lambda v: calls.append(("x", v)), "y": lambda v: calls.append(("y", v))},
        {"x": 1, "y": 2}, rounds=3, calls=2, warmup=1,
    )
    assert {k: len(v) for k, v in timings.items()} == {"x": 6, "y": 6}
    timed = calls[2:]  # after one warm-up call each
    assert [c[0] for c in timed[:4]] == ["x", "x", "y", "y"]  # alternating rounds, not all x then all y


def test_detector_check_splits_accuracy_by_warning():
    correct = np.array([True, True, False, False, True])
    flagged = np.array([False, False, True, True, True])
    summary = check_detector.summarise(correct, flagged)
    assert summary["flagged_share"] == pytest.approx(0.6)
    assert summary["accuracy_accepted"] == 1.0 and summary["accuracy_flagged"] == pytest.approx(1 / 3, abs=1e-4)
    assert check_detector.summarise(correct, np.zeros(5, bool))["accuracy_flagged"] is None


def test_report_renders_comparisons_and_best_rule():
    row = {"name": "run", "backbone": "mobilenetv2", "strategy": "finetune", "image_size": 128,
           "augmentation": "standard", "background_prob": 0.0, "pv_accuracy": 0.9, "pv_ci95": [0.88, 0.92],
           "pv_ece": 0.02, "pv_ece_calibrated": 0.01, "params": 2_000_000, "latency_ms": 50.0}
    comparison = {"question": "Q", "a": "a", "b": "b", "split": "plantdoc_all", "images": 741,
                  "accuracy_a": 0.22, "accuracy_b": 0.27, "difference": 0.05, "only_a": 30, "only_b": 67,
                  "p_value": 0.0002, "p_holm": 0.002}
    text = make_report.markdown([row], [], [], [comparison], best="run")
    assert "Paired comparisons" in text
    assert "(+5.0 pts) | 30 / 67 | < 0.001 | **0.002** |" in text
    assert "Detailed figures use `run`" in text
