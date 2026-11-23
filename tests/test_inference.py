import json

import numpy as np
import pytest
from PIL import Image

from helpers import jpeg_bytes, png_bytes
from plantdisease import config
from plantdisease.inference import (
    ImageLoadError,
    ModelConfigError,
    PlantDiseaseClassifier,
    Prediction,
    load_class_names,
    load_image,
    preprocess,
)


# --- Image loading ----------------------------------------------------------
def test_transparent_png_is_flattened_on_white():
    image = Image.new("RGBA", (40, 30), (255, 0, 0, 0))  # fully transparent red
    loaded = load_image(png_bytes(image))
    assert loaded.mode == "RGB"
    assert loaded.getpixel((5, 5)) == (255, 255, 255)


def test_greyscale_becomes_rgb():
    loaded = load_image(png_bytes(Image.new("L", (20, 20), 128)))
    assert loaded.mode == "RGB" and loaded.getpixel((0, 0)) == (128, 128, 128)


def test_exif_rotation_is_applied():
    image = Image.new("RGB", (60, 20), (0, 128, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"
    loaded = load_image(jpeg_bytes(image, exif=exif.tobytes()))
    assert loaded.size == (20, 60)


def test_garbage_bytes_raise_clear_error():
    with pytest.raises(ImageLoadError, match="JPG, PNG or WEBP"):
        load_image(b"definitely not an image")


def test_oversized_image_is_rejected_with_resize_advice(monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ImageLoadError, match="too large.*resize") as caught:
        load_image(png_bytes(Image.new("RGB", (20, 20))))
    assert "JPG" not in str(caught.value)  # the format is fine; only the size is the problem


def test_preprocess_contract():
    batch = preprocess(Image.new("RGB", (300, 200), (255, 0, 0)), 128)
    assert batch.shape == (1, 128, 128, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= 0.0 and batch.max() <= 1.0
    assert batch[0, 0, 0, 0] == pytest.approx(1.0)


# --- Artifacts --------------------------------------------------------------
def test_class_names_accepts_index_dict(tmp_path):
    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"1": "b", "0": "a", "10": "k"}))
    assert load_class_names(path) == ["a", "b", "k"]


def test_duplicate_class_names_rejected(tmp_path):
    path = tmp_path / "classes.json"
    path.write_text(json.dumps(["a", "a"]))
    with pytest.raises(ModelConfigError):
        load_class_names(path)


def test_class_count_mismatch_is_caught():
    class FakeModel:
        output_shape = (None, 5)
        input_shape = (None, 128, 128, 3)

    with pytest.raises(ModelConfigError, match="5 outputs"):
        PlantDiseaseClassifier(FakeModel(), ["a", "b"])


def test_missing_model_dir_is_reported(tmp_path):
    with pytest.raises(ModelConfigError, match="Missing model artifact"):
        PlantDiseaseClassifier.load(tmp_path)


# --- Real model -------------------------------------------------------------
def test_model_matches_its_artifacts(classifier):
    assert classifier.num_classes == 10
    assert classifier.image_size == 128
    assert classifier.info.evaluation is not None


def test_model_expects_unit_range_input(classifier):
    # The app feeds [0, 1]; the model's own Rescaling layer must map that to [-1, 1].
    # If someone retrains with a different Rescaling, the app's /255 would be wrong.
    rescaling = classifier.model.get_layer("rescaling")
    assert rescaling.scale == pytest.approx(2.0)
    assert rescaling.offset == pytest.approx(-1.0)


def test_prediction_is_a_distribution(classifier, sample_image):
    prediction = classifier.predict(sample_image)
    assert prediction.probabilities.shape == (10,)
    assert prediction.probabilities.sum() == pytest.approx(1.0, abs=1e-4)
    ranked = prediction.ranked()
    assert [p for _, _, p in ranked] == sorted(prediction.probabilities, reverse=True)
    assert ranked[0][1] == prediction.top_label


def test_sample_prediction_is_stable(classifier, sample_image):
    # Regression guard: catches accidental changes to preprocessing or class order.
    prediction = classifier.predict(sample_image)
    assert prediction.top_label == "Late blight"
    assert prediction.confidence > 0.9


def test_describe_reads_the_real_model(classifier):
    facts = classifier.describe()
    assert facts["backbone_name"].startswith("mobilenetv2")
    assert facts["backbone_trainable"] is False
    assert 0 < facts["trainable_params"] < facts["total_params"]


# --- Prediction maths ---------------------------------------------------------
def test_prediction_statistics():
    certain = Prediction(np.array([1.0, 0.0, 0.0, 0.0]), ("a", "b", "c", "d"))
    uniform = Prediction(np.full(4, 0.25), ("a", "b", "c", "d"))
    assert certain.margin == pytest.approx(1.0)
    assert certain.normalized_entropy == pytest.approx(0.0, abs=1e-6)
    assert uniform.margin == pytest.approx(0.0)
    assert uniform.normalized_entropy == pytest.approx(1.0)
