"""The experiment runner's main loop on a tiny dataset: train, evaluate, resume, report."""

import json
import shutil

import numpy as np
import pytest
from PIL import Image

from helpers import PLANTVILLAGE_FOLDERS
from plantdisease import config
from plantdisease.pipeline import write_manifest
from scripts import benchmark_speed, compare_runs, make_report, run_experiments


@pytest.mark.slow
def test_runner_trains_evaluates_resumes_and_reports(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    pv_rows, pd_rows = [], []
    for label in PLANTVILLAGE_FOLDERS[:3]:
        for split, count in (("train", 6), ("val", 3), ("test", 3)):
            for i in range(count):
                rel = f"{label}/{split}{i}.png"
                (raw / "plantvillage" / label).mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)).save(raw / "plantvillage" / rel)
                mask = np.zeros((48, 48, 3), np.uint8)
                mask[12:36, 12:36] = 255
                Image.fromarray(mask).save(raw / "plantvillage" / f"{label}/{split}{i}_m.png")
                pv_rows.append({"path": rel, "mask_path": f"{label}/{split}{i}_m.png", "label": label,
                                "split": split, "group": rel})
        for split in ("train", "test"):
            rel = f"{label}/{split}.png"
            (raw / "plantdoc" / label).mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(raw / "plantdoc" / rel)
            pd_rows.append({"path": rel, "mask_path": "", "label": label, "split": split, "group": rel})
    write_manifest(tmp_path / "pv.csv", pv_rows)
    write_manifest(tmp_path / "pd.csv", pd_rows)
    monkeypatch.setattr(config, "RAW_DATA_DIR", raw)
    monkeypatch.setattr(config, "PLANTVILLAGE_MANIFEST", tmp_path / "pv.csv")
    monkeypatch.setattr(config, "PLANTDOC_MANIFEST", tmp_path / "pd.csv")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")

    tiny = {"image_size": 32, "epochs": 1, "fine_tune_epochs": 1, "batch_size": 6, "weights": None}
    suite = {"runs": [
        dict(tiny, name="tiny_bg_mixed", group="field", backbone="mobilenetv2", strategy="finetune",
             background_prob=0.5, extra_train_sources=["plantdoc"]),
        dict(tiny, name="tiny_frozen", group="strategy", backbone="mobilenetv2", strategy="frozen"),
    ]}
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    args = ["--suite", str(tmp_path / "suite.json"), "--runs-dir", str(tmp_path / "runs"),
            "--results-dir", str(tmp_path / "results")]

    assert run_experiments.main(args) == 0
    for name in ("tiny_bg_mixed", "tiny_frozen"):
        result = json.loads((tmp_path / "results" / f"{name}.json").read_text())
        assert result["run"]["epochs_run"] >= 1 and "plantdoc_test" in result
        assert "per_class" not in result["plantvillage_test"]  # compact form
    assert (tmp_path / "results" / "summary.json").exists()
    assert (tmp_path / "reports" / "RESULTS.md").exists()

    # Paired tests and the latency benchmark feed back into the report.
    runs_args = ["--runs-dir", str(tmp_path / "runs")]
    assert compare_runs.main(runs_args + ["--pair", "tiny_frozen", "tiny_bg_mixed", "plantdoc_test",
                                          "--output", str(tmp_path / "results" / "comparisons.json")]) == 0
    comparison = json.loads((tmp_path / "results" / "comparisons.json").read_text())["comparisons"][0]
    assert comparison["images"] == 3 and 0 <= comparison["p_holm"] <= 1
    assert benchmark_speed.main(runs_args + ["--results-dir", str(tmp_path / "results"), "--rounds", "1", "--calls", "1"]) == 0
    assert "latency_method" in json.loads((tmp_path / "results" / "tiny_frozen.json").read_text())["speed"]
    assert make_report.main(["--results-dir", str(tmp_path / "results"), "--reports-dir", str(tmp_path / "reports")]) == 0
    assert "Paired comparisons" in (tmp_path / "reports" / "RESULTS.md").read_text(encoding="utf-8")

    # Second call resumes: nothing is retrained.
    shutil.rmtree(tmp_path / "runs" / "tiny_frozen")
    assert run_experiments.main(args + ["--only", "tiny_frozen"]) == 0
    assert not (tmp_path / "runs" / "tiny_frozen").exists()
