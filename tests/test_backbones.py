import numpy as np
import pytest
import tensorflow as tf

from plantdisease.backbones import BACKBONES, SCRATCH_BN_MOMENTUM, build_model, get_backbone, unfreeze_top
from plantdisease.gradcam import GradCAM


@pytest.mark.parametrize("key", list(BACKBONES))
def test_every_backbone_keeps_the_app_contract(key):
    model, backbone = build_model(key, 3, 64, weights=None)
    spec = get_backbone(key)
    assert model.output_shape == (None, 3)
    rescaling = model.get_layer("rescaling")
    assert (rescaling.scale, rescaling.offset) == (spec.scale, spec.offset)
    assert not backbone.trainable

    probs = model(np.random.rand(2, 64, 64, 3).astype("float32"), training=False).numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    cam = GradCAM(model)  # picks a suitable layer automatically
    heatmap = cam.heatmap(np.random.rand(1, 64, 64, 3).astype("float32"), 0)
    assert heatmap.ndim == 2 and heatmap.shape[0] >= 2


def test_unfreeze_top_keeps_batchnorm_frozen():
    _, backbone = build_model("mobilenetv2", 3, 64, weights=None)
    start = unfreeze_top(backbone, 0.35)
    assert 0 < start < len(backbone.layers)
    for i, layer in enumerate(backbone.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization) or i < start:
            assert not layer.trainable
    assert any(layer.trainable for layer in backbone.layers[start:])


def test_scratch_training_uses_fast_batchnorm_statistics():
    def momenta(backbone):
        return {l.momentum for l in backbone.layers if isinstance(l, tf.keras.layers.BatchNormalization)}

    _, transfer = build_model("mobilenetv2", 3, 64, weights=None)
    _, scratch = build_model("mobilenetv2", 3, 64, weights=None, train_batchnorm=True)
    assert momenta(transfer) == {0.999}  # Keras default kept: ImageNet statistics are not updated anyway
    assert momenta(scratch) == {SCRATCH_BN_MOMENTUM}
    assert scratch.trainable


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="Unknown backbone"):
        get_backbone("alexnet")
