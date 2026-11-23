"""Evaluation -> compact results -> report, end to end on a tiny synthetic dataset."""

import json
import shutil

import numpy as np
import pytest
from PIL import Image

from helpers import PLANTVILLAGE_FOLDERS
from plantdisease import config
from plantdisease.evaluation import evaluate_run, stratified_sample
from scripts import make_report
from scripts.run_experiments import compact


def _image(path, rng, leaf=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((96, 96, 3), 140, np.uint8)
    if leaf:
        pixels[20:76, 20:76] = rng.integers(0, 256, (56, 56, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _mask(path, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((96, 96, 3), np.uint8)
    pixels[20:76, 20:76] = 200
    Image.fromarray(pixels).save(path)


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("data")
    pv_rows, pd_rows = [], []
    for label in PLANTVILLAGE_FOLDERS:
        for split, count in (("train", 6), ("val", 3), ("test", 3)):
            for i in range(count):
                rel = f"pv/{label}/{split}{i}.png"
                _image(root / rel, rng)
                _mask(root / f"pv/{label}/{split}{i}_mask.png", rng)
                pv_rows.append({"path": rel, "mask_path": f"pv/{label}/{split}{i}_mask.png",
                                "label": label, "split": split, "group": rel})
        for i in range(2):
            rel = f"pd/{label}/{i}.png"
            _image(root / rel, rng)
            pd_rows.append({"path": rel, "mask_path": "", "label": label, "split": "test", "group": rel})
    return root, pv_rows, pd_rows


def test_stratified_sample_keeps_every_class(tiny_data):
    _, pv_rows, _ = tiny_data
    sample = stratified_sample(pv_rows, 20)
    assert {r["label"] for r in sample} == set(PLANTVILLAGE_FOLDERS)


def test_full_evaluation_and_report(tiny_data, tmp_path):
    root, pv_rows, pd_rows = tiny_data
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in (config.MODEL_FILENAME, config.CLASS_NAMES_FILENAME, config.MODEL_INFO_FILENAME):
        shutil.copy2(config.MODELS_DIR / name, model_dir / name)

    results = evaluate_run(model_dir, pv_rows, root, pd_rows, root, ood_samples=10,
                           train_feature_samples=60, attention_samples=5, log=lambda *_: None)

    assert results["plantvillage_test"]["num_images"] == 30
    assert results["plantdoc_test"]["num_images"] == 20
    assert 0.05 <= results["temperature"] <= 20
    for ood_set in ("synthetic", "background_only"):
        for method in ("msp", "energy", "mahalanobis"):
            assert 0.0 <= results["ood"][ood_set][method]["auroc"] <= 1.0
    assert results["gradcam"]["images"] == 10  # stratified: at least one image per class
    assert results["gradcam"]["lift"] > 0
    assert results["speed"]["params"] == 2270794
    assert (model_dir / "ood_detector.npz").exists()
    info = json.loads((model_dir / config.MODEL_INFO_FILENAME).read_text())
    assert info["temperature"] == results["temperature"] and "field_accuracy" in info

    results_dir, reports_dir = tmp_path / "results", tmp_path / "reports"
    results_dir.mkdir()
    for name, seed in (("mnv2_finetune", 42), ("mnv2_finetune_s1", 1)):
        run = {"name": name, "group": "seeds", "backbone": "mobilenetv2", "strategy": "finetune",
               "seed": seed, "train_seconds": 60}
        (results_dir / f"{name}.json").write_text(json.dumps(compact(results, run)))
    assert make_report.main(["--results-dir", str(results_dir), "--reports-dir", str(reports_dir)]) == 0

    summary = json.loads((results_dir / "summary.json").read_text())
    assert len(summary["runs"]) == 2 and summary["seed_stats"][0]["runs"] == 2
    markdown = (reports_dir / "RESULTS.md").read_text(encoding="utf-8")
    assert "mnv2_finetune_s1" in markdown and "Seed variation" in markdown
    for figure in ("domain_gap", "reliability_best", "ood_best", "confusion_plantvillage_best", "confusion_plantdoc_best"):
        assert (reports_dir / "figures" / f"{figure}.png").stat().st_size > 5_000, figure
