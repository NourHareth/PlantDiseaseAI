"""End-to-end UI tests with Streamlit's headless AppTest runner."""

import json

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from helpers import png_bytes
from plantdisease import config
from plantdisease.ui import resources

TIMEOUT = 120


def diagnose_page():
    from plantdisease.ui.pages import diagnose

    diagnose.render()


def model_info_page():
    from plantdisease.ui.pages import model_info

    model_info.render()


def _texts(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_home_page_renders():
    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Supported classes"] == "10"
    assert metrics["Test accuracy"] == "89.62%"

    at.button(key="start_diagnosis").click().run()  # call-to-action switches page
    assert not at.exception
    assert any("Upload a photo" in i.value for i in at.info)


def test_diagnose_page_waits_for_an_image():
    at = AppTest.from_function(diagnose_page, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("Upload a photo" in i.value for i in at.info)
    assert not at.metric


def test_diagnose_sample_photo_end_to_end():
    at = AppTest.from_function(diagnose_page, default_timeout=TIMEOUT).run()
    at.button(key="sample_button").click().run()
    assert not at.exception
    assert "Late blight" in _texts(at)
    assert at.metric[0].label == "Calibrated confidence"
    assert len(at.tabs) == 3
    assert any("feature map" in c.value for c in at.caption)  # Grad-CAM rendered


def test_blank_image_is_withheld_until_override(monkeypatch):
    black = png_bytes(Image.new("RGB", (200, 200)))
    monkeypatch.setattr(resources, "sample_image_bytes", lambda: black)

    at = AppTest.from_function(diagnose_page, default_timeout=TIMEOUT).run()
    at.button(key="sample_button").click().run()
    assert not at.exception
    assert any("almost completely black" in e.value for e in at.error)
    assert not at.metric  # no confidence shown for a meaningless result

    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert at.metric[0].label == "Calibrated confidence"


def test_model_info_page_renders():
    at = AppTest.from_function(model_info_page, default_timeout=TIMEOUT).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Test images"] == "2,726"
    assert len(at.dataframe) == 1
    assert len(at.dataframe[0].value) == 10
    text = _texts(at)
    assert "T = 0.82" in text and "Field photos (PlantDoc): 22.81%" in text
    assert "measured at 23% accuracy" in text  # limitations quote the field result


def experiments_page():
    from plantdisease.ui.pages import experiments

    experiments.render()


def _point(auroc):
    return {"auroc": auroc, "fpr_at_95_tpr": 0.1}


def _fake_results(directory):
    directory.mkdir(parents=True, exist_ok=True)
    common = {"backbone": "mobilenetv2", "strategy": "finetune", "image_size": 128, "augmentation": "standard",
              "background_prob": 0.0, "pv_ece": 0.05, "pv_ece_calibrated": 0.01, "gradcam_lift": 1.3,
              "params": 2_270_794, "latency_ms": 20.0, "train_minutes": 12.0,
              "ood_synthetic_mahalanobis": 0.99, "ood_background_only_mahalanobis": 0.97}
    runs = [dict(common, name="baseline_v1_deployed", group="baseline", pv_accuracy=0.9, pd_accuracy=0.2),
            dict(common, name="mnv2_finetune", group="strategy", pv_accuracy=0.99, pd_accuracy=0.4)]
    comparison = {"question": "Fine-tuning vs. frozen", "a": "baseline_v1_deployed", "b": "mnv2_finetune",
                  "split": "plantdoc_all", "images": 741, "accuracy_a": 0.2, "accuracy_b": 0.4, "difference": 0.2,
                  "only_a": 10, "only_b": 158, "p_value": 1e-30, "p_holm": 1e-30}
    (directory / "summary.json").write_text(json.dumps(
        {"runs": runs, "best_run": "mnv2_finetune", "seed_stats": [], "comparisons": [comparison], "tflite": []}))
    bins = {"confidence": [0.5, 0.9], "accuracy": [0.4, 0.8], "count": [10, 90]}
    ood = {s: {m: _point(0.9) for m in ("msp", "energy", "mahalanobis")} for s in ("synthetic", "background_only")}
    (directory / "mnv2_finetune.json").write_text(json.dumps(
        {"temperature": 1.5, "reliability": {"raw": bins, "calibrated": bins}, "ood": ood}))


def test_experiments_page_without_results(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "none")
    at = AppTest.from_function(experiments_page, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("No experiment results yet" in i.value for i in at.info)


def test_experiments_page_with_results(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    _fake_results(tmp_path / "results")
    at = AppTest.from_function(experiments_page, default_timeout=TIMEOUT).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Field-photo accuracy"] == "40.0%"
    assert len(at.tabs) == 5
    assert len(at.dataframe) == 2  # all runs + paired tests (no seed families in the fake data)
    assert at.dataframe[1].value["Change (points)"].iloc[0] == pytest.approx(20.0)


def test_unfamiliar_photo_is_flagged(monkeypatch):
    from plantdisease.ood import MahalanobisDetector

    strict = MahalanobisDetector(np.zeros((10, 1280)), np.eye(1280), threshold=float("inf"))
    monkeypatch.setattr(resources, "get_ood_detector", lambda: strict)
    at = AppTest.from_function(diagnose_page, default_timeout=TIMEOUT).run()
    at.button(key="sample_button").click().run()
    assert not at.exception
    assert any("unlike the images the model was trained on" in w.value for w in at.warning)
    assert any("Treat as a hint only" in w.value for w in at.warning)  # assessment capped at "low"


def test_cached_values_are_plain_data():
    # st.cache_data pickles its values; project classes break that after a hot reload.
    probabilities, issues = resources._analyze_cached(resources.sample_image_bytes())
    assert type(probabilities).__module__ == "numpy"
    assert all(type(row) is tuple and all(type(v) is str for v in row) for row in issues)
    assert type(resources.explain(resources.sample_image_bytes(), 0)).__module__ == "numpy"


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Cached results must not leak between tests that patch resources."""
    yield
    resources._analyze_cached.clear()
    resources.explain.clear()
    resources.sample_image_bytes.clear()
