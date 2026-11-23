"""Measure the model on a labelled image folder and check the class order.

Images go through exactly the same loading and preprocessing as the app, so the
numbers describe what users actually get.

    python -m scripts.evaluate --data-dir datasets/tomato/test
    python -m scripts.evaluate --data-dir datasets/tomato/test --update-model-info

Outputs ``reports/metrics.json`` and ``reports/figures/confusion_matrix.png``.

Class-order check: Keras' ``image_dataset_from_directory`` numbers classes in
case-sensitive alphabetical order of the folder names. If ``class_names.json``
was written by hand in a different order, the app shows the wrong label for
most classes while the training accuracy still looks fine. This script scores
both orders and tells you which one the model actually uses
(``--fix-class-order`` rewrites the file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.evaluate)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/evaluate.py)

import numpy as np

from plantdisease import config
from plantdisease.datasets import class_folders, list_images, map_folders_to_classes
from plantdisease.inference import ImageLoadError, PlantDiseaseClassifier, load_image
from plantdisease.knowledge import normalize_label
from plantdisease.metrics import classification_report

BATCH_SIZE = 64
# Sequential single-hue ramp (light -> dark) for the confusion matrix
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]


def predict_folder(classifier: PlantDiseaseClassifier, items: list[tuple[Path, str]]) -> tuple[np.ndarray, list[str]]:
    """Probabilities for every readable image, plus each image's folder name."""
    probabilities, folders, batch, batch_folders = [], [], [], []
    skipped = 0

    def flush() -> None:
        if batch:
            probabilities.append(classifier.predict_batch(np.concatenate(batch)))
            folders.extend(batch_folders)
            batch.clear()
            batch_folders.clear()

    for i, (path, folder) in enumerate(items, start=1):
        try:
            batch.append(classifier.preprocess(load_image(path)))
            batch_folders.append(folder)
        except ImageLoadError:
            skipped += 1
        if len(batch) == BATCH_SIZE:
            flush()
        if i % 500 == 0:
            print(f"  {i}/{len(items)} images")
    flush()
    if skipped:
        print(f"  skipped {skipped} unreadable files")
    return np.concatenate(probabilities), folders


def keras_sorted_order(folders: list[str], class_names: list[str], folder_to_index: dict[str, int]) -> list[str]:
    """``class_names`` re-ordered the way Keras would have numbered the classes.

    Keras sorts the *folder names* case-sensitively, so the dataset's own folder
    names are used when every class has a folder; otherwise the labels themselves.
    """
    if len(folders) != len(class_names):
        return sorted(class_names)
    folder_by_key = {normalize_label(f): f for f in folders}
    ordered_folders = sorted(folder_by_key[normalize_label(c)] for c in class_names)
    return [class_names[folder_to_index[folder]] for folder in ordered_folders]


def plot_confusion_matrix(matrix: np.ndarray, labels: list[str], accuracy: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals > 0)

    fig, ax = plt.subplots(figsize=(10, 8.5), dpi=150)
    image = ax.imshow(normalised, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix (row-normalised) · accuracy {accuracy:.2%}", fontsize=12, pad=12)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                ax.text(
                    j, i, f"{normalised[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if normalised[i, j] > 0.55 else "#1f2937",
                )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, format=lambda v, _: f"{v:.0%}")
    colorbar.outline.set_visible(False)
    colorbar.set_label("Share of the true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with one sub-folder per class.")
    parser.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--split-name", default="test", help="Name recorded in the metrics (default: test).")
    parser.add_argument("--update-model-info", action="store_true", help="Write the result into model_info.json.")
    parser.add_argument("--fix-class-order", action="store_true", help="Rewrite class_names.json if the check finds a better order.")
    args = parser.parse_args(argv)

    classifier = PlantDiseaseClassifier.load(args.model_dir)
    class_names = list(classifier.class_names)
    folders = class_folders(args.data_dir)
    folder_to_index = map_folders_to_classes(folders, class_names)
    items = list_images(args.data_dir)
    if not items:
        print(f"No images found in {args.data_dir}", file=sys.stderr)
        return 1

    print(f"Evaluating {len(items)} images from {args.data_dir} ...")
    probabilities, image_folders = predict_folder(classifier, items)
    y_true = np.array([folder_to_index[f] for f in image_folders])
    y_pred = probabilities.argmax(axis=1)
    report = classification_report(y_true, y_pred, class_names)

    # --- Class-order check -------------------------------------------------
    # Interpretation B: output i means the i-th class in Keras' sorted order.
    keras_order = keras_sorted_order(folders, class_names, folder_to_index)
    remap = np.array([class_names.index(name) for name in keras_order])
    keras_accuracy = float((remap[y_pred] == y_true).mean())
    report["class_order_check"] = {
        "accuracy_with_class_names_json": report["accuracy"],
        "accuracy_with_keras_sorted_order": round(keras_accuracy, 4),
        "keras_sorted_order": keras_order,
    }

    print(f"\nAccuracy: {report['accuracy']:.2%}   macro F1: {report['macro_f1']:.3f}   images: {report['num_images']}")
    print(f"{'class':40s} {'prec':>6s} {'recall':>6s} {'f1':>6s} {'n':>6s}")
    for label, m in report["per_class"].items():
        print(f"{label[:40]:40s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:6d}")

    order_is_wrong = keras_order != class_names and keras_accuracy > report["accuracy"] + 0.05
    if order_is_wrong:
        print(
            "\nWARNING: class_names.json appears to be in the WRONG ORDER.\n"
            f"  accuracy with class_names.json order : {report['accuracy']:.2%}\n"
            f"  accuracy with Keras' sorted order    : {keras_accuracy:.2%}\n"
            f"  correct order is probably: {keras_order}\n"
            "  Re-run with --fix-class-order to rewrite models/class_names.json."
        )
    else:
        print(f"\nClass order check passed (Keras sorted order would score {keras_accuracy:.2%}).")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_confusion_matrix(
        np.array(report["confusion_matrix"]), class_names, report["accuracy"],
        args.reports_dir / "figures" / "confusion_matrix.png",
    )
    print(f"Saved {args.reports_dir / 'metrics.json'} and {args.reports_dir / 'figures' / 'confusion_matrix.png'}")

    if order_is_wrong and args.fix_class_order:
        path = args.model_dir / config.CLASS_NAMES_FILENAME
        path.write_text(json.dumps(keras_order, indent=2), encoding="utf-8")
        print(f"Rewrote {path}. Run the evaluation again to confirm.")
        return 0

    if args.update_model_info and not order_is_wrong:
        info_path = args.model_dir / config.MODEL_INFO_FILENAME
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        info["evaluation"] = {
            "split": args.split_name,
            "accuracy": report["accuracy"],
            "num_images": report["num_images"],
            "source": f"Measured with scripts/evaluate.py on {args.data_dir.name}/.",
        }
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"Updated {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
