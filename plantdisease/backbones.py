"""Backbone registry and model construction shared by training and experiments.

Every model built here keeps the app's input contract: RGB in ``[0, 1]``,
mapped by a ``Rescaling`` layer to whatever range the backbone expects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    display_name: str
    constructor: str  # attribute of keras.applications
    scale: float  # Rescaling from [0, 1] to the backbone's expected input range
    offset: float
    fine_tune_fraction: float = 0.35  # share of backbone layers (from the top) unfrozen when fine-tuning


BACKBONES: dict[str, BackboneSpec] = {
    # MobileNetV2 / ResNet50V2 expect [-1, 1]
    "mobilenetv2": BackboneSpec("mobilenetv2", "MobileNetV2", "MobileNetV2", 2.0, -1.0),
    "resnet50v2": BackboneSpec("resnet50v2", "ResNet50V2", "ResNet50V2", 2.0, -1.0, 0.30),
    # Keras' EfficientNet normalises internally and expects [0, 255]
    "efficientnetb0": BackboneSpec("efficientnetb0", "EfficientNetB0", "EfficientNetB0", 255.0, 0.0, 0.30),
}

STRATEGIES = ("frozen", "finetune", "scratch")
SCRATCH_BN_MOMENTUM = 0.9


def get_backbone(key: str) -> BackboneSpec:
    try:
        return BACKBONES[key]
    except KeyError:
        raise ValueError(f"Unknown backbone {key!r}; choose from {', '.join(BACKBONES)}") from None


def build_augmentation(strength: str = "standard"):
    import tensorflow as tf
    from tensorflow.keras import layers

    if strength == "none":
        # Keras 3 cannot build an empty Sequential; keep the layer name for export/inspection.
        return layers.Identity(name="data_augmentation")
    if strength == "standard":
        steps = [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ]
    elif strength == "strong":  # used for the field-robustness experiments
        steps = [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom((-0.3, 0.1)),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
        ]
    else:
        raise ValueError(f"Unknown augmentation strength {strength!r}")
    return tf.keras.Sequential(steps, name="data_augmentation")


def build_model(
    backbone_key: str,
    num_classes: int,
    image_size: int,
    *,
    weights: str | None = "imagenet",
    dropout: float = 0.3,
    augmentation: str = "standard",
    train_batchnorm: bool = False,
):
    """Return ``(model, backbone)``; the backbone starts frozen.

    ``train_batchnorm`` must be True when training from scratch: BatchNorm then
    learns its statistics. For transfer learning it stays False so the ImageNet
    statistics are kept, also while fine-tuning (Keras transfer-learning guide).
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    spec = get_backbone(backbone_key)
    backbone = getattr(tf.keras.applications, spec.constructor)(
        input_shape=(image_size, image_size, 3), include_top=False, weights=weights
    )
    backbone.trainable = train_batchnorm
    if train_batchnorm:
        # Keras applications use BatchNorm momentum 0.999: the moving statistics then average
        # over ~1,000 steps. From random weights the activations change much faster than that,
        # so the moving statistics used at evaluation time lag far behind the batch statistics
        # used in training, and validation collapses to one class. 0.9 averages ~10 steps.
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.momentum = SCRATCH_BN_MOMENTUM

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = build_augmentation(augmentation)(inputs)
    x = layers.Rescaling(spec.scale, offset=spec.offset, name="rescaling")(x)
    x = backbone(x) if train_batchnorm else backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dense")(x)
    return tf.keras.Model(inputs, outputs, name=f"plant_disease_{spec.key}"), backbone


def unfreeze_top(backbone, fraction: float) -> int:
    """Unfreeze the top ``fraction`` of backbone layers (BatchNorm stays frozen).

    Returns the index of the first trainable layer.
    """
    import tensorflow as tf

    start = int(len(backbone.layers) * (1.0 - fraction))
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        frozen = i < start or isinstance(layer, tf.keras.layers.BatchNormalization)
        layer.trainable = not frozen
    return start
