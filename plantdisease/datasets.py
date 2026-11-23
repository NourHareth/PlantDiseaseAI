"""Helpers for image-folder datasets (one sub-folder per class).

Expected layouts::

    dataset/                       dataset/
      train/<class>/*.jpg            <class>/*.jpg     (split automatically)
      val/<class>/*.jpg
      test/<class>/*.jpg           (optional)
"""

from __future__ import annotations

from pathlib import Path

from plantdisease.knowledge import normalize_label

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
_VAL_NAMES = ("val", "valid", "validation")


def find_splits(data_dir: Path) -> dict[str, Path]:
    """Return the split directories that exist (``train``/``val``/``test``)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    splits: dict[str, Path] = {}
    if (data_dir / "train").is_dir():
        splits["train"] = data_dir / "train"
        for name in _VAL_NAMES:
            if (data_dir / name).is_dir():
                splits["val"] = data_dir / name
                break
    if (data_dir / "test").is_dir():
        splits["test"] = data_dir / "test"
    return splits


def class_folders(directory: Path) -> list[str]:
    """Class sub-folder names, sorted exactly like Keras sorts them (case-sensitive)."""
    return sorted(p.name for p in Path(directory).iterdir() if p.is_dir())


def list_images(directory: Path) -> list[tuple[Path, str]]:
    """``(image path, class folder name)`` for every image below ``directory``."""
    items = []
    for folder in class_folders(directory):
        for path in sorted((Path(directory) / folder).rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                items.append((path, folder))
    return items


def map_folders_to_classes(folders: list[str], class_names: list[str]) -> dict[str, int]:
    """Map each dataset folder to the model's class index by normalised name."""
    index = {normalize_label(name): i for i, name in enumerate(class_names)}
    mapping, unknown = {}, []
    for folder in folders:
        key = normalize_label(folder)
        if key in index:
            mapping[folder] = index[key]
        else:
            unknown.append(folder)
    if unknown:
        raise ValueError(
            "These dataset folders do not match any class in class_names.json: "
            + ", ".join(unknown)
        )
    return mapping
