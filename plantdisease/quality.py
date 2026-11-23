"""Input sanity checks and an honest reliability assessment for each prediction.

A softmax classifier always answers with one of its classes, even for a black
square or random noise (this model says "Late blight, 99.5%" for an all-black
image). These checks catch inputs that are clearly *not* leaf photos before
the prediction is shown as if it were meaningful.

They are deliberately simple heuristics, not a learned out-of-distribution
detector: they stop obvious garbage, they cannot prove an image is a tomato leaf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from plantdisease import config
from plantdisease.inference import Prediction

_ANALYSIS_SIZE = 128

# Thresholds (calibrated on real leaf photos vs. synthetic non-photos)
_MIN_STD = 4.0  # grey-level std below this = blank / single colour
_MAX_NOISE_RATIO = 0.6  # neighbour difference / std above this = noise, not a photo
_MIN_PLANT_COVERAGE = 0.15  # share of pixels with leaf-like colours (warning below this)
_NO_PLANT_COVERAGE = 0.03  # essentially no leaf-like colours at all (blocking)
_TOO_DARK_MEAN = 15.0  # too dark to analyse at all (blocking)
_DARK_MEAN = 30.0  # dark, but usable (warning)
_BRIGHT_MEAN = 235.0
_MIN_SIDE = 96


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" blocks the result until the user overrides, "warning" is shown alongside
    code: str
    message: str

    @property
    def is_blocking(self) -> bool:
        return self.severity == "error"


@dataclass(frozen=True)
class Assessment:
    level: str  # "high" | "moderate" | "low" | "unreliable"
    title: str
    message: str


def _plant_coverage(rgb: Image.Image) -> float:
    """Fraction of pixels whose colour is plausible for leaf tissue or lesions.

    Hue 15-170 degrees covers brown lesions, yellowing and every shade of green;
    the saturation/value floors drop grey, white and near-black pixels.
    """
    hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
    hue_deg = hsv[..., 0] * (360.0 / 255.0)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    plant = (hue_deg >= 15) & (hue_deg <= 170) & (sat >= 0.15) & (val >= 0.08)
    return float(plant.mean())


def image_statistics(image: Image.Image) -> dict[str, float]:
    small = image.convert("RGB").resize((_ANALYSIS_SIZE, _ANALYSIS_SIZE), Image.Resampling.BILINEAR)
    grey = np.asarray(small.convert("L"), dtype=np.float32)
    std = float(grey.std())
    neighbour_diff = float(
        (np.abs(np.diff(grey, axis=0)).mean() + np.abs(np.diff(grey, axis=1)).mean()) / 2
    )
    return {
        "mean": float(grey.mean()),
        "std": std,
        "noise_ratio": neighbour_diff / std if std > 0 else 0.0,
        "plant_coverage": _plant_coverage(small),
    }


def check_image(image: Image.Image) -> list[Issue]:
    """Return problems with ``image`` as model input, most severe first."""
    stats = image_statistics(image)
    issues: list[Issue] = []

    if stats["std"] < _MIN_STD:
        if stats["mean"] < _DARK_MEAN:
            message = "The image is almost completely black."
        elif stats["mean"] > _BRIGHT_MEAN:
            message = "The image is almost completely white."
        else:
            message = "The image is almost a single flat colour."
        issues.append(Issue("error", "uniform", message))
    elif stats["noise_ratio"] > _MAX_NOISE_RATIO:
        issues.append(Issue("error", "noise", "The image looks like random noise, not a photo."))
    elif stats["mean"] < _TOO_DARK_MEAN:
        issues.append(Issue("error", "too_dark", "The image is too dark to analyse. Try better lighting."))
    elif stats["plant_coverage"] < _NO_PLANT_COVERAGE:
        if stats["mean"] < _DARK_MEAN:  # in dim JPEGs the colours are lost, so darkness is the real cause
            issues.append(Issue("error", "too_dark", "The image is too dark to make out a leaf. Try better lighting."))
        else:
            issues.append(Issue("error", "no_plant", "No leaf-like colours were found in the image."))
    else:
        if stats["plant_coverage"] < _MIN_PLANT_COVERAGE:
            issues.append(
                Issue(
                    "warning",
                    "low_plant_coverage",
                    "Very little of the image has leaf-like colours. Is a leaf clearly in frame?",
                )
            )
        if stats["mean"] < _DARK_MEAN:
            issues.append(Issue("warning", "too_dark", "The image is very dark."))
        elif stats["mean"] > _BRIGHT_MEAN:
            issues.append(Issue("warning", "too_bright", "The image is very bright or overexposed."))

    if min(image.size) < _MIN_SIDE:
        issues.append(
            Issue(
                "warning",
                "low_resolution",
                f"The image is small ({image.width} x {image.height} px); details may be lost.",
            )
        )
    return sorted(issues, key=lambda issue: issue.severity != "error")


UNFAMILIAR = Issue(
    "warning",
    "unfamiliar",
    "This photo looks unlike the images the model was trained on (lab photos of single leaves), "
    "so its answer is less reliable than usual.",
)


def familiarity_issue(score: float, threshold: float | None) -> list[Issue]:
    """Warning when the feature-space OOD score falls below the calibrated threshold."""
    return [UNFAMILIAR] if threshold is not None and score < threshold else []


def assess(prediction: Prediction, issues: list[Issue]) -> Assessment:
    """Translate confidence and input checks into plain-language guidance."""
    if any(issue.is_blocking for issue in issues):
        return Assessment(
            "unreliable",
            "Not a meaningful result",
            "This image does not look like a leaf photo, so the model's answer is not meaningful. "
            "It always picks one of its 10 classes, whatever it is shown.",
        )

    confidence = prediction.confidence
    if confidence >= config.HIGH_CONFIDENCE:
        level = "high"
    elif confidence >= config.MODERATE_CONFIDENCE:
        level = "moderate"
    else:
        level = "low"
    if issues and level == "high":
        level = "moderate"  # image problems cap how much the score can be trusted
    if any(issue.code == "unfamiliar" for issue in issues):
        level = "low"  # outside the training distribution, confidence scores are unreliable

    messages = {
        "high": (
            "Strong match",
            "The model strongly favours this class. A high score is not a guarantee: "
            "compare the symptoms below with your leaf.",
        ),
        "moderate": (
            "Possible match",
            "The model leans towards this class but is not certain. Check the runner-up "
            "classes, or try a sharper, well-lit photo of a single leaf.",
        ),
        "low": (
            "Uncertain",
            "Several classes are close. Treat this as a hint only and try a clearer photo "
            "of a single leaf against a plain background.",
        ),
    }
    title, message = messages[level]
    if level == "low" and any(issue.code == "unfamiliar" for issue in issues):
        title, message = (
            "Treat as a hint only",
            "The photo differs from the model's training images, where accuracy drops sharply. "
            "Retake it as a close-up of one leaf against a plain background for a more reliable answer.",
        )
    return Assessment(level, title, message)
