import numpy as np
import pytest
from PIL import Image

from plantdisease.inference import Prediction
from plantdisease.quality import Issue, assess, check_image

LABELS = tuple("abcdefghij")


def _prediction(top: float) -> Prediction:
    rest = (1.0 - top) / 9
    return Prediction(np.array([top] + [rest] * 9), LABELS)


@pytest.mark.parametrize(
    "name, code",
    [("black", "uniform"), ("white", "uniform"), ("green", "uniform"), ("noise", "noise"), ("grey_gradient", "no_plant")],
)
def test_non_photos_are_blocked(synthetic_images, name, code):
    issues = check_image(synthetic_images[name])
    assert issues and issues[0].code == code and issues[0].is_blocking


def test_real_leaf_photo_passes(sample_image):
    assert check_image(sample_image) == []


def test_small_image_gets_a_warning(sample_image):
    issues = check_image(sample_image.resize((64, 64)))
    assert [i.code for i in issues] == ["low_resolution"]
    assert not issues[0].is_blocking


def test_dark_photo_gets_a_warning(sample_image):
    dark = Image.eval(sample_image, lambda v: v // 5)  # underexposed but usable
    issues = check_image(dark)
    assert [i.code for i in issues] == ["too_dark"]
    assert not issues[0].is_blocking


def test_nearly_black_photo_is_blocked_as_too_dark(sample_image):
    very_dark = Image.eval(sample_image, lambda v: v // 10)
    issues = check_image(very_dark)
    assert issues[0].code == "too_dark" and issues[0].is_blocking


def test_dim_image_without_visible_colours_is_called_too_dark_not_leafless():
    # Mean ~20: dark but above the hard limit. JPEG destroys colour at this level, so
    # "no leaf-like colours" would blame the wrong thing; the message must name the lighting.
    dim = np.tile(np.linspace(5, 35, 200, dtype=np.float32)[None, :, None], (200, 1, 3)).astype(np.uint8)
    issues = check_image(Image.fromarray(dim))
    assert issues[0].code == "too_dark" and issues[0].is_blocking
    assert "too dark" in issues[0].message


def test_flat_image_message_names_the_colour(synthetic_images):
    assert "black" in check_image(synthetic_images["black"])[0].message
    assert "white" in check_image(synthetic_images["white"])[0].message
    assert "flat colour" in check_image(synthetic_images["green"])[0].message


def test_blocking_issue_makes_result_unreliable():
    issue = Issue("error", "uniform", "flat")
    assert assess(_prediction(0.99), [issue]).level == "unreliable"


@pytest.mark.parametrize("top, level", [(0.95, "high"), (0.7, "moderate"), (0.4, "low")])
def test_confidence_bands(top, level):
    assert assess(_prediction(top), []).level == level


def test_warnings_cap_confidence_at_moderate():
    warning = Issue("warning", "too_dark", "dark")
    assert assess(_prediction(0.99), [warning]).level == "moderate"
