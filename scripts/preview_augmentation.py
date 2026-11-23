"""Save a figure showing the background-replacement augmentation on real images.

    python -m scripts.preview_augmentation

Rows: original photo, leaf mask, three random replaced backgrounds.
Output: ``reports/figures/background_replacement.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.preview_augmentation)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/preview_augmentation.py)

from plantdisease import config


def find_pairs(root: Path, count: int) -> list[tuple[Path, Path]]:
    pairs = []
    for class_dir in sorted((root / "raw" / "color").iterdir()):
        for image in sorted(class_dir.iterdir())[:50]:
            mask = root / "raw" / "segmented" / class_dir.name / f"{image.stem}_final_masked.jpg"
            if mask.exists():
                pairs.append((image, mask))
                break
        if len(pairs) == count:
            break
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=config.RAW_DATA_DIR / "plantvillage")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=config.FIGURES_DIR / "background_replacement.png")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    from plantdisease.knowledge import KnowledgeBase
    from plantdisease.pipeline import _decode, _leaf_mask, replace_background

    knowledge = KnowledgeBase.from_json(config.DISEASE_INFO_PATH)
    tf.keras.utils.set_random_seed(7)
    pairs = find_pairs(args.root, args.count)
    if not pairs:
        raise SystemExit(f"No image/mask pairs found under {args.root}")
    titles = ["Original", "Leaf mask", "Replaced 1", "Replaced 2", "Replaced 3"]
    fig, axes = plt.subplots(len(pairs), 5, figsize=(11, 2.3 * len(pairs)), dpi=150, layout="constrained")
    axes = axes.reshape(len(pairs), 5)
    for row, (image_path, mask_path) in enumerate(pairs):
        image = _decode(str(image_path), args.size)
        mask = _leaf_mask(str(mask_path), args.size)
        panels = [image, tf.repeat(mask, 3, axis=-1)] + [replace_background(image, mask, args.size) for _ in range(3)]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel.numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
        axes[row, 0].annotate(knowledge.display_name(image_path.parent.name),
                              (0, 0.5), xycoords="axes fraction", xytext=(-6, 0), textcoords="offset points",
                              ha="right", va="center", fontsize=8, rotation=90)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
