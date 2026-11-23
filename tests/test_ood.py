import numpy as np
import pytest

from plantdisease.ood import FeatureExtractor, MahalanobisDetector, energy_score, synthetic_ood_images


def test_mahalanobis_separates_far_points(tmp_path):
    rng = np.random.default_rng(0)
    features = np.concatenate([rng.normal(0, 1, (300, 8)), rng.normal(6, 1, (300, 8))])
    labels = np.repeat([0, 1], 300)
    detector = MahalanobisDetector.fit(features, labels)

    in_scores = detector.score(rng.normal(0, 1, (100, 8)))
    out_scores = detector.score(rng.normal(-15, 1, (100, 8)))
    assert in_scores.min() > out_scores.max()

    threshold = detector.calibrate_threshold(detector.score(features), tpr=0.95)
    assert np.mean(detector.score(features) >= threshold) == pytest.approx(0.95, abs=0.01)

    path = tmp_path / "detector.npz"
    detector.save(path)
    loaded = MahalanobisDetector.load(path)
    assert loaded.threshold == pytest.approx(threshold)
    assert np.allclose(loaded.score(features[:5]), detector.score(features[:5]))


def test_energy_prefers_confident_logits():
    peaked = np.array([[10.0, 0.0, 0.0]])
    flat = np.array([[0.0, 0.0, 0.0]])
    assert energy_score(peaked)[0] > energy_score(flat)[0]


def test_synthetic_ood_images_are_valid():
    images = synthetic_ood_images(12, size=32)
    assert images.shape == (12, 32, 32, 3) and images.dtype == np.float32
    assert images.min() >= 0.0 and images.max() <= 1.0


def test_feature_extractor_reproduces_the_model(classifier, sample_image):
    extractor = FeatureExtractor(classifier.model)
    batch = classifier.preprocess(sample_image)
    features, logits = extractor(batch)
    assert features.shape == (1, 1280)
    z = logits - logits.max()
    assert np.allclose(np.exp(z) / np.exp(z).sum(), classifier.predict_batch(batch), atol=1e-4)


def test_predict_with_features_matches_predict(classifier, sample_image):
    prediction, features = classifier.predict_with_features(sample_image)
    assert features.shape == (1280,)
    assert np.allclose(prediction.probabilities, classifier.predict(sample_image).probabilities, atol=1e-4)
