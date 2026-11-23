"""Train one tomato leaf classifier.

    python -m scripts.train --manifest                       # the versioned PlantVillage split (recommended)
    python -m scripts.train --data-dir datasets/tomato       # any class-folder dataset
    python -m scripts.train --manifest --backbone efficientnetb0 --strategy finetune --background-prob 0.6

Strategies: ``frozen`` (train the head only), ``finetune`` (head first, then the
top of the backbone with a small learning rate, BatchNorm kept frozen) and
``scratch`` (random initialisation). Backbones: mobilenetv2, efficientnetb0,
resnet50v2. ``--background-prob`` pastes leaves onto random backgrounds using
the PlantVillage segmentation masks (manifest mode only).

Output goes to ``<workdir>/runs/<name>/``: the model, ``class_names.json`` (in
the order Keras used), ``model_info.json``, history and training curves. With
``--evaluate`` the full evaluation (field photos, calibration, OOD, Grad-CAM)
runs afterwards. The app keeps using ``models/`` until you copy a run there.

For a whole study use ``scripts/run_experiments.py`` instead.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.train)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/train.py)

import numpy as np

from plantdisease import config
from plantdisease.backbones import BACKBONES, STRATEGIES
from plantdisease.datasets import find_splits, list_images
from plantdisease.pipeline import read_manifest
from plantdisease.training import TrainConfig, train


def rows_from_folders(data_dir: Path, seed: int) -> list[dict]:
    """Manifest-style rows for a class-folder dataset (auto 80/20 split if needed)."""
    data_dir = Path(data_dir)
    splits = find_splits(data_dir)
    rows = []
    if "train" in splits:
        for split, directory in splits.items():
            for path, folder in list_images(directory):
                rows.append({"path": path.relative_to(data_dir).as_posix(), "label": folder, "split": split})
        if "val" not in splits:
            raise SystemExit("A train/ folder needs a val/ (or valid/, validation/) folder next to it.")
    else:
        items = list_images(data_dir)
        random.Random(seed).shuffle(items)
        n_val = max(1, round(0.2 * len(items)))
        for i, (path, folder) in enumerate(items):
            rows.append({"path": path.relative_to(data_dir).as_posix(), "label": folder,
                         "split": "val" if i < n_val else "train"})
    return rows


def evaluate_simple(model_dir: Path, rows: list[dict], root: Path, split: str) -> dict:
    """Accuracy on a held-out split, recorded in model_info.json."""
    from plantdisease.evaluation import labels_for, predict_rows
    from plantdisease.inference import PlantDiseaseClassifier

    classifier = PlantDiseaseClassifier.load(model_dir)
    probs = predict_rows(classifier, rows, root)
    y = labels_for(rows, classifier.class_names)
    accuracy = float((probs.argmax(1) == y).mean())
    info_path = model_dir / config.MODEL_INFO_FILENAME
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["evaluation"] = {
        "split": split,
        "accuracy": round(accuracy, 4),
        "num_images": len(rows),
        "source": f"Measured by scripts/train.py at {datetime.now():%Y-%m-%d %H:%M}.",
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info["evaluation"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", nargs="?", const=config.PLANTVILLAGE_MANIFEST, type=Path,
                        help="Split manifest CSV (default: data/splits/plantvillage_tomato.csv).")
    source.add_argument("--data-dir", type=Path, help="Class-folder dataset (train/val[/test] or flat).")
    parser.add_argument("--images-root", type=Path, default=config.RAW_DATA_DIR / "plantvillage",
                        help="Folder the manifest paths are relative to.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <workdir>/runs/<name>/")
    parser.add_argument("--backbone", choices=list(BACKBONES), default="mobilenetv2")
    parser.add_argument("--strategy", choices=STRATEGIES, default="finetune")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--augmentation", choices=["none", "standard", "strong"], default="standard")
    parser.add_argument("--background-prob", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8, help="Stage 1 (or the whole run for scratch).")
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true", help="Run the full evaluation afterwards (manifest mode).")
    args = parser.parse_args(argv)

    if args.manifest:
        rows, root = read_manifest(args.manifest), args.images_root
        dataset_name = "PlantVillage tomato (leakage-aware 70/15/15 split)"
    else:
        rows, root = rows_from_folders(args.data_dir, args.seed), args.data_dir
        dataset_name = f"Image folders in {args.data_dir.name}/"
        if args.background_prob:
            parser.error("--background-prob needs masks, i.e. --manifest mode.")

    name = args.name or f"{args.backbone}_{args.strategy}_{datetime.now():%Y%m%d-%H%M%S}"
    cfg = TrainConfig(
        name=name, backbone=args.backbone, strategy=args.strategy, image_size=args.image_size,
        augmentation=args.augmentation, background_prob=args.background_prob, epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs, learning_rate=args.learning_rate,
        fine_tune_learning_rate=args.fine_tune_learning_rate, batch_size=args.batch_size,
        dropout=args.dropout, label_smoothing=args.label_smoothing, patience=args.patience,
        seed=args.seed, weights=None if args.weights == "none" else "imagenet", dataset=dataset_name,
    )
    output_dir = args.output_dir or config.RUNS_DIR / name
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    print(f"Training {name}: {len(by_split['train'])} train / {len(by_split['val'])} val images")
    train(cfg, by_split["train"], by_split["val"], root, output_dir)

    if args.evaluate and args.manifest:
        from plantdisease.evaluation import evaluate_run

        pd_rows = read_manifest(config.PLANTDOC_MANIFEST) if config.PLANTDOC_MANIFEST.exists() else None
        results = evaluate_run(output_dir, rows, root, pd_rows, config.RAW_DATA_DIR / "plantdoc")
        print(json.dumps({k: results[k]["accuracy"] for k in ("plantvillage_test", "plantdoc_test") if k in results}))
    else:
        split = "test" if by_split["test"] else "val"
        result = evaluate_simple(output_dir, by_split[split], root, split)
        print(f"{split} accuracy: {result['accuracy']:.2%} on {result['num_images']} images")
    print(f"Saved run to {output_dir}. The app uses models/; copy the run there to deploy it.")
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    raise SystemExit(main())
