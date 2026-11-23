"""Cached access to the model and other heavy resources.

``st.cache_resource`` keeps one model in memory for every user session;
``st.cache_data`` memoises per-image results, so interacting with the page
(switching tabs, toggling options) never re-runs the network on the same photo.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from plantdisease import config
from plantdisease.gradcam import GradCAM
from plantdisease.inference import PlantDiseaseClassifier, Prediction, load_image
from plantdisease.knowledge import KnowledgeBase
from plantdisease.ood import MahalanobisDetector
from plantdisease.quality import Issue, check_image, familiarity_issue


@st.cache_resource(show_spinner="Loading the model...")
def get_classifier() -> PlantDiseaseClassifier:
    return PlantDiseaseClassifier.load()


@st.cache_resource
def get_knowledge() -> KnowledgeBase:
    return KnowledgeBase.from_json()


@st.cache_resource(show_spinner=False)
def get_gradcam() -> GradCAM:
    return GradCAM(get_classifier().model)


@st.cache_resource(show_spinner=False)
def get_ood_detector() -> MahalanobisDetector | None:
    """The optional "unfamiliar image" detector stored next to the model."""
    path = config.MODELS_DIR / config.OOD_DETECTOR_FILENAME
    return MahalanobisDetector.load(path) if path.exists() else None


@st.cache_data(max_entries=32, show_spinner=False)
def _analyze_cached(image_bytes: bytes) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    # Only plain types are cached: Streamlit pickles cached values, and instances of
    # project classes stop being picklable when Streamlit hot-reloads edited modules.
    image = load_image(image_bytes)
    classifier, detector = get_classifier(), get_ood_detector()
    issues = check_image(image)
    if detector is None:
        prediction = classifier.predict(image)
    else:
        prediction, features = classifier.predict_with_features(image)
        if not any(issue.is_blocking for issue in issues):
            issues += familiarity_issue(float(detector.score(features[None])[0]), detector.threshold)
    return prediction.probabilities, [(i.severity, i.code, i.message) for i in issues]


def analyze(image_bytes: bytes) -> tuple[Prediction, list[Issue]]:
    """Prediction and input checks for an encoded image (memoised per image)."""
    probabilities, issues = _analyze_cached(image_bytes)
    prediction = Prediction(probabilities=probabilities, labels=get_classifier().class_names)
    return prediction, [Issue(*row) for row in issues]


@st.cache_data(max_entries=16, show_spinner=False)
def explain(image_bytes: bytes, class_index: int) -> np.ndarray:
    """Grad-CAM heatmap (values 0..1) for ``class_index``."""
    classifier = get_classifier()
    batch = classifier.preprocess(load_image(image_bytes))
    return get_gradcam().heatmap(batch, class_index)


@st.cache_data
def sample_image_bytes() -> bytes:
    return config.SAMPLE_IMAGE_PATH.read_bytes()


@st.cache_data
def stylesheet() -> str:
    return config.STYLES_PATH.read_text(encoding="utf-8")
