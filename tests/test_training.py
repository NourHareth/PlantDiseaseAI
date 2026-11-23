"""Training core: every strategy and data option on a tiny dataset (random weights)."""

import io
import json
import zipfile

import h5py
import numpy as np
import pytest
from PIL import Image

from plantdisease import config
from plantdisease.inference import PlantDiseaseClassifier
from plantdisease.training import TrainConfig, train

LABELS = ["Tomato___Late_blight", "Tomato___healthy"]


def _dataset(root, source, with_masks, rng):
    rows = []
    for label in LABELS:
        for split, count in (("train", 6), ("val", 2)):
            for i in range(count):
                rel = f"{label}/{split}{i}.png"
                (root / label).mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(root / rel)
                row = {"path": rel, "label": label, "split": split, "source": source, "mask_path": ""}
                if with_masks:
                    mask = np.zeros((40, 40, 3), np.uint8)
                    mask[10:30, 10:30] = 255
                    Image.fromarray(mask).save(root / f"{label}/{split}{i}_mask.png")
                    row["mask_path"] = f"{label}/{split}{i}_mask.png"
                rows.append(row)
    return rows


@pytest.mark.slow
@pytest.mark.parametrize(
    "options",
    [
        {"strategy": "finetune", "background_prob": 0.5, "label_smoothing": 0.1, "augmentation": "strong"},
        {"strategy": "scratch", "augmentation": "none"},
    ],
    ids=["finetune-bg-smoothing-mixed", "scratch"],
)
def test_training_core(tmp_path, options):
    rng = np.random.default_rng(0)
    pv = _dataset(tmp_path / "pv", "plantvillage", True, rng)
    pd = _dataset(tmp_path / "pd", "plantdoc", False, rng)  # field photos have no masks
    rows = pv + ([r for r in pd if r["split"] == "train"] if options["strategy"] == "finetune" else [])
    cfg = TrainConfig(name="tiny", image_size=32, epochs=1, fine_tune_epochs=1, batch_size=4, weights=None, **options)

    out = train(cfg, [r for r in rows if r["split"] == "train"], [r for r in pv if r["split"] == "val"],
                {"plantvillage": tmp_path / "pv", "plantdoc": tmp_path / "pd"}, tmp_path / "run", verbose=0)

    classifier = PlantDiseaseClassifier.load(out)
    assert list(classifier.class_names) == LABELS
    info = json.loads((out / config.MODEL_INFO_FILENAME).read_text())
    history = json.loads((out / "history.json").read_text())
    assert history["train_seconds"] > 0
    if options["strategy"] == "scratch":
        assert info["pretrained_weights"].startswith("None")
        assert classifier.describe()["backbone_trainable"] is True
    else:
        assert history["fine_tune_start"] == 1 and len(history["history"]["loss"]) == 2
    assert json.loads((out / "train_config.json").read_text())["strategy"] == options["strategy"]

    # The saved file holds the inference model only: no optimizer state.
    with zipfile.ZipFile(out / config.MODEL_FILENAME) as archive:
        with h5py.File(io.BytesIO(archive.read("model.weights.h5"))) as weights:
            assert "optimizer" not in weights
