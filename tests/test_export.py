"""TensorFlow Lite export (slow: each conversion takes several seconds)."""

import numpy as np
import pytest

from scripts.export_tflite import convert, inference_model, run_tflite


@pytest.mark.slow
def test_tflite_variants_agree_with_keras(classifier, sample_image):
    model = inference_model(classifier.model)
    assert "data_augmentation" not in [layer.name for layer in model.layers]
    batch = classifier.preprocess(sample_image)
    expected = classifier.predict_batch(batch)

    float_probs, latency = run_tflite(convert(model, "float32", None), batch)
    assert np.allclose(float_probs, expected, atol=1e-4)
    assert latency > 0

    calibration = np.random.default_rng(0).random((8, 128, 128, 3), dtype=np.float32)
    int8_probs, _ = run_tflite(convert(model, "int8", calibration), batch)
    assert int8_probs.shape == expected.shape
    assert np.allclose(int8_probs.sum(), 1.0, atol=0.05)
