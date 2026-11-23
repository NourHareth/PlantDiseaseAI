"""Command-line scripts: evaluation (incl. the class-order check), Grad-CAM export,
model inspection and a tiny end-to-end training run."""

import json

import numpy as np
import pytest
from PIL import Image

from helpers import PLANTVILLAGE_FOLDERS
from plantdisease import config
from plantdisease.inference import PlantDiseaseClassifier, load_class_names, preprocess
from scripts import evaluate, explain, inspect_model, train

# The order the app's class_names.json was written in (case-insensitive A-Z),
# which differs from the order Keras assigns (case-sensitive A-Z).
HAND_WRITTEN_ORDER = [
    "Bacterial spot", "Early blight", "healthy", "Late blight", "Leaf Mold", "Septoria leaf spot",
    "Spider mites Two-spotted spider mite", "Target Spot", "Tomato mosaic virus", "Tomato Yellow Leaf Curl Virus",
]


# --- Evaluation -------------------------------------------------------------
def test_classification_report_numbers():
    report = evaluate.classification_report(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), ["a", "b"])
    assert report["accuracy"] == 0.75
    assert report["per_class"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert report["confusion_matrix"] == [[1, 1], [0, 2]]


def test_keras_order_uses_case_sensitive_folder_sort():
    folders = sorted(PLANTVILLAGE_FOLDERS)
    mapping = {f: i for i, f in enumerate(folders)}  # identity mapping for the test
    order = evaluate.keras_sorted_order(folders, folders, mapping)
    assert order[-1] == "Tomato___healthy"  # lower-case sorts after upper-case


class ColourCodedModel:
    """Stand-in for a model trained with Keras' folder order.

    Each test image is a flat grey whose level encodes its folder's position in
    case-sensitive sorted order; the fake model 'recognises' it perfectly and
    answers with that Keras index.
    """

    def __init__(self, model_dir):
        self.class_names = tuple(load_class_names(model_dir / config.CLASS_NAMES_FILENAME))
        self.image_size = 16

    @classmethod
    def load(cls, model_dir):
        return cls(model_dir)

    def preprocess(self, image):
        return preprocess(image, self.image_size)

    def predict_batch(self, batch):
        index = np.rint(batch[:, 0, 0, 0] * 255 / 20).astype(int)
        return np.eye(len(self.class_names))[index]


def _colour_coded_dataset(root):
    for k, folder in enumerate(sorted(HAND_WRITTEN_ORDER)):  # Keras order
        (root / folder).mkdir(parents=True)
        for j in range(2):
            Image.new("RGB", (32, 32), (20 * k,) * 3).save(root / folder / f"{j}.png")


def test_evaluate_detects_and_fixes_wrong_class_order(tmp_path, monkeypatch, capsys):
    data_dir, model_dir, reports = tmp_path / "test", tmp_path / "model", tmp_path / "reports"
    _colour_coded_dataset(data_dir)
    model_dir.mkdir()
    (model_dir / config.CLASS_NAMES_FILENAME).write_text(json.dumps(HAND_WRITTEN_ORDER))
    monkeypatch.setattr(evaluate, "PlantDiseaseClassifier", ColourCodedModel)

    args = ["--data-dir", str(data_dir), "--model-dir", str(model_dir), "--reports-dir", str(reports)]
    assert evaluate.main(args + ["--fix-class-order"]) == 0
    assert "WRONG ORDER" in capsys.readouterr().out
    metrics = json.loads((reports / "metrics.json").read_text())
    assert metrics["class_order_check"]["accuracy_with_keras_sorted_order"] == 1.0
    assert metrics["accuracy"] < 0.5
    assert (reports / "figures" / "confusion_matrix.png").exists()

    fixed = load_class_names(model_dir / config.CLASS_NAMES_FILENAME)
    assert fixed == sorted(HAND_WRITTEN_ORDER)

    assert evaluate.main(args) == 0  # with the corrected file everything is right
    assert json.loads((reports / "metrics.json").read_text())["accuracy"] == 1.0
    assert "check passed" in capsys.readouterr().out


# --- Explain / inspect ----------------------------------------------------------
def test_explain_writes_a_figure(tmp_path):
    output = tmp_path / "cam.png"
    assert explain.main([str(config.SAMPLE_IMAGE_PATH), "--output", str(output)]) == 0
    assert output.stat().st_size > 10_000


def test_inspect_model_runs(capsys):
    assert inspect_model.main([]) == 0
    assert "All checks passed" in capsys.readouterr().out


# --- Training (tiny smoke run: random weights, 2 epochs) ------------------------
@pytest.mark.slow
def test_training_run_produces_a_usable_model(tmp_path):
    rng = np.random.default_rng(0)
    folders = ["Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___healthy"]
    for split, count in (("train", 6), ("val", 3), ("test", 3)):
        for folder in folders:
            directory = tmp_path / "data" / split / folder
            directory.mkdir(parents=True)
            for i in range(count):
                pixels = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(directory / f"{i}.png")

    out = tmp_path / "run"
    code = train.main([
        "--data-dir", str(tmp_path / "data"), "--output-dir", str(out), "--weights", "none",
        "--image-size", "64", "--batch-size", "6", "--epochs", "1", "--fine-tune-epochs", "1",
    ])
    assert code == 0
    for name in (config.MODEL_FILENAME, config.CLASS_NAMES_FILENAME, config.MODEL_INFO_FILENAME,
                 "history.json", "training_curves.png"):
        assert (out / name).exists(), name

    # Class names are saved in exactly the order Keras used.
    assert load_class_names(out / config.CLASS_NAMES_FILENAME) == sorted(folders)
    info = json.loads((out / config.MODEL_INFO_FILENAME).read_text())
    assert info["evaluation"]["split"] == "test" and info["evaluation"]["num_images"] == 9

    # The app's code path works with the freshly trained model.
    from plantdisease.gradcam import GradCAM

    classifier = PlantDiseaseClassifier.load(out)
    image = Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8))
    prediction = classifier.predict(image)
    assert prediction.probabilities.sum() == pytest.approx(1.0, abs=1e-4)
    heatmap = GradCAM(classifier.model).heatmap(classifier.preprocess(image), prediction.top_index)
    assert heatmap.ndim == 2
