"""Grad-CAM visual explanations (Selvaraju et al., 2017).

The model is expected to be ``[preprocessing layers] -> backbone -> [head layers]
-> Dense``, which is what ``scripts/train.py`` produces. Gradients are taken
with respect to the class *logit* (the Dense output before softmax): at 99%
confidence the softmax is saturated, which shrinks and distorts gradients.
"""

from __future__ import annotations

import matplotlib
import matplotlib.colors
import numpy as np
from PIL import Image

from plantdisease import config

COLORMAP = "inferno"  # perceptually uniform: dark = low attention, bright = high


def default_layer(backbone, min_size: int = 8) -> str:
    """The deepest activation layer whose feature map is still at least ``min_size`` wide.

    Deeper layers are more class-specific; at 128 px input the final block is
    only 4 x 4, so the last 8 x 8 activation is a better compromise. Works for
    MobileNetV2 (block_13_expand_relu), EfficientNetB0 (block6a_expand_activation)
    and ResNet50V2 (conv4_block6_1_relu).
    """
    candidates = []
    for layer in backbone.layers:
        shape = getattr(layer, "output", None)
        shape = getattr(shape, "shape", None)
        if shape is None or len(shape) != 4 or (shape[1] or 0) < min_size:
            continue
        if layer.name.endswith(("relu", "activation", "_out")):
            candidates.append(layer.name)
    if not candidates:
        raise ValueError("Could not find a suitable Grad-CAM layer.")
    return candidates[-1]


class GradCAM:
    def __init__(self, model, layer_name: str | None = None):
        import tensorflow as tf

        layers = [layer for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]
        nested = [(i, layer) for i, layer in enumerate(layers) if isinstance(layer, tf.keras.Model)]
        if not nested:
            raise ValueError("Expected a nested backbone model.")
        backbone_index, backbone = max(nested, key=lambda item: len(item[1].layers))
        if layer_name is None:
            layer_name = (
                config.GRADCAM_LAYER if _has_layer(backbone, config.GRADCAM_LAYER) else default_layer(backbone)
            )
        elif not _has_layer(backbone, layer_name):
            raise ValueError(f"The backbone has no layer named {layer_name!r}.")
        head = layers[backbone_index + 1 :]
        if not head or not isinstance(head[-1], tf.keras.layers.Dense):
            raise ValueError("Expected the model to end with a Dense classification layer.")

        self._tf = tf
        self._pre_layers = layers[:backbone_index]
        self._head_layers = head[:-1]
        self._classifier = head[-1]
        self.layer_name = layer_name
        self._grad_model = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=[backbone.get_layer(layer_name).output, backbone.output],
        )

    def heatmap(self, batch: np.ndarray, class_index: int) -> np.ndarray:
        """Heatmap in [0, 1] for one preprocessed image (``batch`` shape ``(1, S, S, 3)``)."""
        tf = self._tf
        x = tf.convert_to_tensor(batch, dtype=tf.float32)
        for layer in self._pre_layers:  # augmentation is a no-op with training=False
            x = layer(x, training=False)

        with tf.GradientTape() as tape:
            feature_map, features = self._grad_model(x, training=False)
            for layer in self._head_layers:
                features = layer(features, training=False)
            logits = tf.matmul(features, self._classifier.kernel) + self._classifier.bias
            score = logits[:, class_index]

        gradients = tape.gradient(score, feature_map)
        channel_weights = tf.reduce_mean(gradients, axis=(1, 2))  # global-average the gradients
        cam = tf.nn.relu(tf.reduce_sum(feature_map[0] * channel_weights[0], axis=-1))
        peak = float(tf.reduce_max(cam))
        if peak <= 0.0:
            return np.zeros(cam.shape, dtype=np.float32)
        return (cam / peak).numpy().astype(np.float32)


def _has_layer(model, name: str) -> bool:
    try:
        model.get_layer(name)
        return True
    except ValueError:
        return False


def colorize(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Upsample a heatmap to ``size`` (width, height); returns (values 0..1, RGB uint8)."""
    small = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255))
    values = np.asarray(small.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    rgb = (matplotlib.colormaps[COLORMAP](values)[..., :3] * 255).astype(np.uint8)
    return values, rgb


def render(image: Image.Image, heatmap: np.ndarray, max_alpha: float = 0.65) -> tuple[Image.Image, Image.Image]:
    """Return ``(heatmap image, overlay)`` at the original image's resolution.

    The overlay's opacity follows the heat, so low-attention regions keep the
    original photo untouched instead of being tinted.
    """
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    values, rgb = colorize(heatmap, image.size)
    alpha = (np.sqrt(values) * max_alpha)[..., np.newaxis]  # sqrt keeps mid-level heat visible
    blended = base * (1.0 - alpha) + rgb.astype(np.float32) * alpha
    return Image.fromarray(rgb), Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def legend_gradient(stops: int = 7) -> str:
    """CSS ``linear-gradient`` matching :data:`COLORMAP`, for an on-page scale legend."""
    cmap = matplotlib.colormaps[COLORMAP]
    colors = [matplotlib.colors.to_hex(cmap(i / (stops - 1))) for i in range(stops)]
    return f"linear-gradient(90deg, {', '.join(colors)})"
