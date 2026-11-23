"""Training core shared by ``scripts/train.py`` and ``scripts/run_experiments.py``."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from plantdisease import config
from plantdisease.backbones import STRATEGIES, build_model, get_backbone, unfreeze_top
from plantdisease.pipeline import class_names_for, make_dataset

# Two-series line chart colours (categorical slots 1 and 2 of the validated palette)
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"


@dataclass
class TrainConfig:
    name: str = "run"
    backbone: str = "mobilenetv2"
    strategy: str = "finetune"  # frozen | finetune | scratch
    image_size: int = 128
    augmentation: str = "standard"  # none | standard | strong
    background_prob: float = 0.0  # share of training images whose background is replaced
    epochs: int = 8  # stage 1 (head only); for "scratch" the whole run
    fine_tune_epochs: int = 12
    learning_rate: float = 1e-3
    # 1e-5 was still improving after 8 epochs (92% val); 1e-4 converges within the budget.
    fine_tune_learning_rate: float = 1e-4
    batch_size: int = 32
    dropout: float = 0.3
    label_smoothing: float = 0.0
    patience: int = 3
    seed: int = 42
    weights: str | None = "imagenet"
    dataset: str = "PlantVillage tomato (leakage-aware 70/15/15 split)"
    extra_train_sources: list[str] = field(default_factory=list)  # e.g. ["plantdoc"]

    def __post_init__(self):
        get_backbone(self.backbone)
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.strategy == "scratch":
            self.weights = None

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _compile(model, learning_rate: float, label_smoothing: float) -> None:
    import tensorflow as tf

    loss = (
        tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        if label_smoothing
        else tf.keras.losses.SparseCategoricalCrossentropy()
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss, metrics=["accuracy"])


def _check_saved_model(model, path: Path) -> None:
    """The file on disk must predict exactly what the trained model predicts.

    Compared after a reload on purpose: some layers (EfficientNet's Normalization)
    only refresh derived state when loaded, so an in-memory clone can differ while
    the saved file is correct.
    """
    import numpy as np
    import tensorflow as tf

    reloaded = tf.keras.models.load_model(path, compile=False)
    x = np.random.default_rng(0).random((2, *model.input_shape[1:]), dtype=np.float32)
    difference = float(np.abs(model(x, training=False) - reloaded(x, training=False)).max())
    if difference > 1e-5:
        raise RuntimeError(f"Saved model at {path} differs from the trained model (max |diff| {difference:.2e})")


def _one_hot(ds, num_classes: int):
    import tensorflow as tf

    return ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))


def plot_history(history: dict, fine_tune_start: int | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    for ax, metric, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], color=TRAIN_COLOR, linewidth=2, label="Training")
        ax.plot(epochs, history[f"val_{metric}"], color=VAL_COLOR, linewidth=2, label="Validation")
        if fine_tune_start:
            ax.axvline(fine_tune_start + 0.5, color="#9ca3af", linewidth=1, linestyle="--")
            ax.annotate("fine-tuning", (fine_tune_start + 0.6, ax.get_ylim()[1]), va="top", fontsize=8, color="#6b7280")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train(
    cfg: TrainConfig,
    train_rows: list[dict],
    val_rows: list[dict],
    roots: dict[str, Path] | Path,
    output_dir: Path,
    *,
    verbose: int = 2,
) -> Path:
    """Train one model and save everything the app and the evaluation need.

    ``roots`` maps a row's ``source`` (default ``"plantvillage"``) to its image folder,
    so rows from several datasets can be mixed.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = get_backbone(cfg.backbone)
    class_names = class_names_for(train_rows)
    num_classes = len(class_names)

    def dataset(rows, training):
        by_source: dict[str, list[dict]] = {}
        for row in rows:
            by_source.setdefault(row.get("source", "plantvillage"), []).append(row)
        parts = []
        for source, source_rows in by_source.items():
            root = roots[source] if isinstance(roots, dict) else roots
            parts.append(
                make_dataset(
                    source_rows, root, class_names, cfg.image_size, cfg.batch_size,
                    shuffle=training, seed=cfg.seed,
                    background_prob=cfg.background_prob if training and any(r.get("mask_path") for r in source_rows) else 0.0,
                ).unbatch()
            )
        ds = parts[0]
        for extra in parts[1:]:
            ds = ds.concatenate(extra)
        if training and len(parts) > 1:
            ds = ds.shuffle(4096, seed=cfg.seed)
        ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        return _one_hot(ds, num_classes) if cfg.label_smoothing else ds

    train_ds, val_ds = dataset(train_rows, True), dataset(val_rows, False)
    model, backbone = build_model(
        cfg.backbone, num_classes, cfg.image_size,
        weights=cfg.weights, dropout=cfg.dropout, augmentation=cfg.augmentation,
        train_batchnorm=cfg.strategy == "scratch",
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True)

    started = time.time()
    if cfg.strategy == "scratch":
        backbone.trainable = True
    _compile(model, cfg.learning_rate, cfg.label_smoothing)
    # shuffle=False: the tf.data pipeline already shuffles (Keras ignores the flag for datasets and warns)
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=[early_stop],
                        shuffle=False, verbose=verbose).history
    fine_tune_start, unfrozen_from = None, None

    if cfg.strategy == "finetune" and cfg.fine_tune_epochs > 0:
        unfrozen_from = unfreeze_top(backbone, spec.fine_tune_fraction)
        _compile(model, cfg.fine_tune_learning_rate, cfg.label_smoothing)
        fine_tune_start = len(history["loss"])
        stage2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=fine_tune_start + cfg.fine_tune_epochs, initial_epoch=fine_tune_start,
            callbacks=[early_stop], shuffle=False, verbose=verbose,
        ).history
        for key, values in stage2.items():
            history.setdefault(key, []).extend(values)
    train_seconds = time.time() - started

    # Save without optimizer state (~2.5x smaller). A new Model over the same layers is
    # uncompiled; clone_model is not enough, because Keras 3 recompiles the clone with the
    # original optimizer.
    inference = tf.keras.Model(model.input, model.output, name=model.name)  # singular: keep the 1-tensor signature
    inference.save(output_dir / config.MODEL_FILENAME)
    _check_saved_model(model, output_dir / config.MODEL_FILENAME)
    (output_dir / config.CLASS_NAMES_FILENAME).write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    trainable = int(sum(int(tf.size(w)) for w in model.trainable_weights))
    strategy_text = {
        "frozen": "Transfer learning, frozen backbone (feature extraction)",
        "finetune": f"Transfer learning, top {spec.fine_tune_fraction:.0%} of the backbone fine-tuned (from layer {unfrozen_from})",
        "scratch": "Trained from random initialisation",
    }[cfg.strategy]
    model_info = {
        "name": f"PlantDisease AI · {cfg.name}",
        "backbone": spec.display_name,
        "architecture": f"{spec.display_name} backbone + global average pooling + dropout ({cfg.dropout}) + dense softmax",
        "pretrained_weights": "ImageNet" if cfg.weights else "None (random initialisation)",
        "training": f"{strategy_text}; {len(history['loss'])} epochs; seed {cfg.seed}",
        "dataset": cfg.dataset + (f" + {', '.join(cfg.extra_train_sources)} train" if cfg.extra_train_sources else ""),
    }
    (output_dir / config.MODEL_INFO_FILENAME).write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps({
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "fine_tune_start": fine_tune_start,
        "train_seconds": round(train_seconds, 1),
        "trainable_params_final": trainable,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    plot_history(history, fine_tune_start, output_dir / "training_curves.png")
    return output_dir
