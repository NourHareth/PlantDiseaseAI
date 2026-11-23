import numpy as np
import pytest

from plantdisease import gradcam


@pytest.fixture(scope="module")
def cam(classifier):
    return gradcam.GradCAM(classifier.model)


def test_heatmap_shape_and_range(cam, classifier, sample_image):
    prediction = classifier.predict(sample_image)
    heatmap = cam.heatmap(classifier.preprocess(sample_image), prediction.top_index)
    assert heatmap.shape == (8, 8)  # block_13_expand_relu at 128 px input
    assert heatmap.min() >= 0.0
    assert heatmap.max() == pytest.approx(1.0)


def test_heatmap_depends_on_the_class(cam, classifier, sample_image):
    batch = classifier.preprocess(sample_image)
    maps = [cam.heatmap(batch, i) for i in range(3)]
    assert not np.allclose(maps[0], maps[1]) or not np.allclose(maps[1], maps[2])


def test_render_matches_photo_size(sample_image):
    heatmap = np.zeros((8, 8), dtype=np.float32)
    heatmap[2, 3] = 1.0
    heat_image, overlay = gradcam.render(sample_image, heatmap)
    assert heat_image.size == sample_image.size
    assert overlay.size == sample_image.size


def test_zero_heatmap_leaves_photo_untouched(sample_image):
    _, overlay = gradcam.render(sample_image, np.zeros((8, 8), dtype=np.float32))
    assert np.array_equal(np.asarray(overlay), np.asarray(sample_image))


def test_unknown_layer_is_reported(classifier):
    with pytest.raises(ValueError, match="no_such_layer"):
        gradcam.GradCAM(classifier.model, "no_such_layer")


def test_legend_gradient_is_css():
    css = gradcam.legend_gradient()
    assert css.startswith("linear-gradient(90deg, #") and css.endswith(")")
