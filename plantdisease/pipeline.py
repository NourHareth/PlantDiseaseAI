"""``tf.data`` input pipelines built from split manifests.

Manifests are CSV files (``data/splits/*.csv``) with the columns
``path, mask_path, label, split, group``; paths are relative to the dataset's
raw folder. Keeping the split in a versioned file (instead of re-splitting on
every run) makes every experiment use exactly the same images.

Background replacement
----------------------
PlantVillage photos all share a plain lab background, which a network can
exploit as a shortcut. With ``background_prob > 0`` the leaf is cut out with
its PlantVillage segmentation mask and pasted onto a random synthetic
background (colour noise in foliage/soil/skin/grey tones plus texture), forcing
the model to rely on the leaf itself.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Foliage greens, soil browns, skin tones, greys, sky (RGB in [0, 1])
BACKGROUND_PALETTE = np.array(
    [
        [0.18, 0.35, 0.12], [0.30, 0.50, 0.20], [0.12, 0.22, 0.10], [0.45, 0.60, 0.30],
        [0.35, 0.25, 0.15], [0.50, 0.38, 0.25], [0.25, 0.18, 0.12], [0.62, 0.52, 0.40],
        [0.85, 0.65, 0.55], [0.70, 0.50, 0.40], [0.50, 0.50, 0.50], [0.20, 0.20, 0.22],
        [0.85, 0.85, 0.82], [0.55, 0.70, 0.85],
    ],
    dtype=np.float32,
)


def read_manifest(path: Path, split: str | None = None) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "mask_path", "label", "split", "group"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def class_names_for(rows: list[dict[str, str]]) -> list[str]:
    """Sorted case-sensitively, exactly like Keras sorts class folders."""
    return sorted({r["label"] for r in rows})


def _decode(path, image_size: int):
    import tensorflow as tf

    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image = tf.image.resize(tf.cast(image, tf.float32) / 255.0, (image_size, image_size), antialias=True)
    return tf.clip_by_value(image, 0.0, 1.0)


def _leaf_mask(mask_path, image_size: int):
    """Soft leaf mask from a PlantVillage 'segmented' image (black background)."""
    import tensorflow as tf

    segmented = _decode(mask_path, image_size)
    mask = tf.cast(tf.reduce_max(segmented, axis=-1, keepdims=True) > 0.08, tf.float32)
    return tf.nn.avg_pool2d(mask[None], 5, 1, "SAME")[0]  # feather the edge


def random_background(image_size: int):
    import tensorflow as tf

    resolution = tf.random.uniform([], 2, 10, dtype=tf.int32)
    noise = tf.random.uniform(tf.stack([1, resolution, resolution, 3]))
    noise = tf.image.resize(noise, (image_size, image_size), method="bicubic")[0]
    colour = tf.gather(tf.constant(BACKGROUND_PALETTE), tf.random.uniform([], 0, len(BACKGROUND_PALETTE), tf.int32))
    mix = tf.random.uniform([], 0.35, 0.85)
    background = mix * colour + (1.0 - mix) * noise
    background += tf.random.normal((image_size, image_size, 3), stddev=tf.random.uniform([], 0.0, 0.05))
    return tf.clip_by_value(background, 0.0, 1.0)


def replace_background(image, mask, image_size: int):
    return mask * image + (1.0 - mask) * random_background(image_size)


def make_dataset(
    rows: list[dict[str, str]],
    root: Path,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
    background_prob: float = 0.0,
):
    """Batches of ``(image in [0, 1], integer label)``."""
    import tensorflow as tf

    index = {name: i for i, name in enumerate(class_names)}
    paths = [str(Path(root) / r["path"]) for r in rows]
    masks = [str(Path(root) / r["mask_path"]) if r.get("mask_path") else "" for r in rows]
    labels = [index[r["label"]] for r in rows]
    if background_prob > 0 and not any(masks):
        raise ValueError("Background replacement needs mask_path values (none were given).")

    def load(path, mask_path, label):
        image = _decode(path, image_size)
        if background_prob > 0:  # rows without a mask are never replaced
            image = tf.cond(
                tf.logical_and(tf.random.uniform([]) < background_prob, tf.strings.length(mask_path) > 0),
                lambda: replace_background(image, _leaf_mask(mask_path, image_size), image_size),
                lambda: image,
            )
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, masks, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
