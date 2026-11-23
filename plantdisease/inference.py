"""Model loading, image preprocessing and prediction.

The model contract (shared with ``scripts/train.py``):

* input: RGB float32 batch, shape ``(N, S, S, 3)``, values in ``[0, 1]``
  (the model's own ``Rescaling`` layer maps this to MobileNetV2's ``[-1, 1]``);
* output: softmax probabilities, one per entry of ``class_names.json``,
  in the same order.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from plantdisease import config

ImageSource = Union[bytes, bytearray, str, Path, BinaryIO]


class ImageLoadError(ValueError):
    """The uploaded file could not be read as an image."""


class ModelConfigError(RuntimeError):
    """Model artifacts are missing or inconsistent with each other."""


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def load_image(source: ImageSource) -> Image.Image:
    """Open an image as RGB, honouring EXIF rotation and flattening transparency.

    Raises :class:`ImageLoadError` for unreadable, corrupt or oversized files.
    """
    fp = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        with Image.open(fp) as img:
            if img.width * img.height > config.MAX_IMAGE_PIXELS:
                raise ImageLoadError(
                    f"The image is too large ({img.width} × {img.height} pixels; the limit is "
                    f"{config.MAX_IMAGE_PIXELS / 1e6:.0f} megapixels). Please resize it and try again."
                )
            img = ImageOps.exif_transpose(img)  # phone photos are often stored rotated
            has_alpha = img.mode in ("RGBA", "LA") or (
                img.mode == "P" and "transparency" in img.info
            )
            if has_alpha:
                rgba = img.convert("RGBA")
                flat = Image.new("RGB", rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.getchannel("A"))
                return flat
            return img.convert("RGB")
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageLoadError("The file is not a readable image. Please upload a JPG, PNG or WEBP photo.") from exc


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """Resize to ``size`` x ``size`` and scale to a ``(1, size, size, 3)`` float batch in [0, 1]."""
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array[np.newaxis, ...]


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Evaluation:
    split: str
    accuracy: float
    num_images: int
    source: str = ""


@dataclass(frozen=True)
class ModelInfo:
    """Facts about the trained model, read from ``models/model_info.json``."""

    name: str = "PlantDisease AI classifier"
    backbone: str = "Unknown"
    architecture: str = "Unknown"
    pretrained_weights: str = "Unknown"
    training: str = "Unknown"
    dataset: str = "Unknown"
    evaluation: Evaluation | None = None
    temperature: float = 1.0  # fitted by temperature scaling; 1.0 = uncalibrated
    field_accuracy: float | None = None  # accuracy on PlantDoc field photos, if measured
    field_images: int | None = None  # how many field photos that accuracy is based on

    @classmethod
    def from_json(cls, path: Path) -> "ModelInfo":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        evaluation = raw.get("evaluation")
        return cls(
            name=raw.get("name", cls.name),
            backbone=raw.get("backbone", cls.backbone),
            architecture=raw.get("architecture", cls.architecture),
            pretrained_weights=raw.get("pretrained_weights", cls.pretrained_weights),
            training=raw.get("training", cls.training),
            dataset=raw.get("dataset", cls.dataset),
            evaluation=Evaluation(**evaluation) if evaluation else None,
            temperature=float(raw.get("temperature", 1.0)),
            field_accuracy=raw.get("field_accuracy"),
            field_images=raw.get("field_images"),
        )

    @property
    def is_calibrated(self) -> bool:
        return self.temperature != 1.0


def load_class_names(path: Path) -> list[str]:
    names = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(names, dict):  # tolerate {"0": "...", "1": "..."} style files
        names = [names[k] for k in sorted(names, key=int)]
    if not names or not all(isinstance(n, str) for n in names):
        raise ModelConfigError(f"{path} must contain a non-empty list of class names.")
    if len(set(names)) != len(names):
        raise ModelConfigError(f"{path} contains duplicate class names.")
    return names


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Prediction:
    probabilities: np.ndarray
    labels: tuple[str, ...] = field(repr=False)

    @property
    def top_index(self) -> int:
        return int(np.argmax(self.probabilities))

    @property
    def top_label(self) -> str:
        return self.labels[self.top_index]

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.top_index])

    @property
    def margin(self) -> float:
        """Gap between the best and second-best class (0..1)."""
        top_two = np.sort(self.probabilities)[-2:]
        return float(top_two[1] - top_two[0]) if len(top_two) == 2 else 1.0

    @property
    def normalized_entropy(self) -> float:
        """0 = all mass on one class, 1 = uniform over every class."""
        p = np.clip(self.probabilities, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum() / math.log(len(p)))

    def ranked(self, k: int | None = None) -> list[tuple[int, str, float]]:
        """``(index, label, probability)`` from most to least likely."""
        order = np.argsort(self.probabilities)[::-1][:k]
        return [(int(i), self.labels[i], float(self.probabilities[i])) for i in order]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


class PlantDiseaseClassifier:
    """A trained Keras model plus the class names that give its outputs meaning."""

    def __init__(self, model, class_names: list[str], info: ModelInfo | None = None):
        n_outputs = int(model.output_shape[-1])
        if n_outputs != len(class_names):
            raise ModelConfigError(
                f"The model has {n_outputs} outputs but class_names.json lists "
                f"{len(class_names)} classes. They must come from the same training run."
            )
        self.model = model
        self.class_names = tuple(class_names)
        self.info = info or ModelInfo()
        self.image_size = int(model.input_shape[1])

    @classmethod
    def load(cls, model_dir: Path = config.MODELS_DIR) -> "PlantDiseaseClassifier":
        model_dir = Path(model_dir)
        model_path = model_dir / config.MODEL_FILENAME
        classes_path = model_dir / config.CLASS_NAMES_FILENAME
        for path in (model_path, classes_path):
            if not path.exists():
                raise ModelConfigError(f"Missing model artifact: {path}")

        import tensorflow as tf  # imported lazily: it is slow and not needed for UI-only code

        model = tf.keras.models.load_model(model_path, compile=False)
        return cls(
            model,
            load_class_names(classes_path),
            ModelInfo.from_json(model_dir / config.MODEL_INFO_FILENAME),
        )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        return preprocess(image, self.image_size)

    def predict_batch(self, batch: np.ndarray) -> np.ndarray:
        """Raw (uncalibrated) probabilities for a preprocessed batch, shape ``(N, num_classes)``."""
        outputs = np.asarray(self.model(batch, training=False), dtype=np.float64)
        if not np.allclose(outputs.sum(axis=-1), 1.0, atol=1e-3):
            outputs = _softmax(outputs)  # model exported without a softmax layer
        return outputs

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        from plantdisease.calibration import apply_temperature

        return apply_temperature(probabilities, self.info.temperature)

    def predict(self, image: Image.Image) -> Prediction:
        """Prediction for one image, with temperature scaling applied when available."""
        probabilities = self.calibrate(self.predict_batch(self.preprocess(image)))[0]
        return Prediction(probabilities=probabilities, labels=self.class_names)

    def predict_with_features(self, image: Image.Image) -> tuple[Prediction, np.ndarray]:
        """Calibrated prediction plus the penultimate feature vector (for OOD detection)."""
        if not hasattr(self, "_extractor"):
            from plantdisease.ood import FeatureExtractor

            self._extractor = FeatureExtractor(self.model)
        features, logits = self._extractor(self.preprocess(image))
        probabilities = self.calibrate(_softmax(logits))[0]
        return Prediction(probabilities=probabilities, labels=self.class_names), features[0]

    def describe(self) -> dict:
        """Architecture facts read from the loaded model itself (not hard-coded)."""
        import tensorflow as tf

        backbone = next(
            (
                layer
                for layer in self.model.layers
                if isinstance(layer, tf.keras.Model) and len(layer.layers) > 20
            ),
            None,
        )
        trainable = int(sum(np.prod(w.shape) for w in self.model.trainable_weights))
        return {
            "input_size": self.image_size,
            "num_classes": self.num_classes,
            "total_params": int(self.model.count_params()),
            "trainable_params": trainable,
            "backbone_name": backbone.name if backbone is not None else "n/a",
            "backbone_trainable": bool(backbone.trainable) if backbone is not None else None,
        }
