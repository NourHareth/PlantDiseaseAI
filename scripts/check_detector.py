"""How the app's "unfamiliar photo" warning behaves on lab and field photos.

    python -m scripts.check_detector
    python -m scripts.check_detector --model-dir <workdir>/runs/mnv2_finetune

Uses the model and Mahalanobis detector exactly as the app loads them, and
reports, for PlantVillage test photos and PlantDoc field photos: how many are
flagged as unfamiliar, and the accuracy on the photos that are accepted versus
flagged. If the warning is useful, accepted photos should be the more accurate ones.
Results go to ``reports/results/detector_check.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.check_detector)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/check_detector.py)

import numpy as np

from plantdisease import config
from plantdisease.pipeline import read_manifest


def summarise(correct: np.ndarray, flagged: np.ndarray) -> dict:
    def acc(mask):
        return round(float(correct[mask].mean()), 4) if mask.any() else None

    return {
        "images": int(len(correct)),
        "flagged_share": round(float(flagged.mean()), 4),
        "accuracy_all": acc(np.ones_like(flagged)),
        "accuracy_accepted": acc(~flagged),
        "accuracy_flagged": acc(flagged),
        "accepted": int((~flagged).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "detector_check.json")
    args = parser.parse_args(argv)

    from plantdisease.evaluation import BATCH, labels_for, load_batch
    from plantdisease.inference import PlantDiseaseClassifier
    from plantdisease.ood import FeatureExtractor, MahalanobisDetector

    detector_path = args.model_dir / config.OOD_DETECTOR_FILENAME
    if not detector_path.exists():
        print(f"No detector at {detector_path}")
        return 1
    classifier = PlantDiseaseClassifier.load(args.model_dir)
    detector = MahalanobisDetector.load(detector_path)
    extractor = FeatureExtractor(classifier.model)

    sets = {
        "plantvillage_test": ([r for r in read_manifest(config.PLANTVILLAGE_MANIFEST) if r["split"] == "test"],
                              config.RAW_DATA_DIR / "plantvillage"),
        "plantdoc_all": (read_manifest(config.PLANTDOC_MANIFEST), config.RAW_DATA_DIR / "plantdoc"),
    }
    report = {"model_dir": str(args.model_dir), "threshold": detector.threshold}
    for name, (rows, root) in sets.items():
        pred, scores = [], []
        for start in range(0, len(rows), BATCH):
            batch = load_batch(classifier, rows[start:start + BATCH], root)
            pred.append(classifier.predict_batch(batch).argmax(axis=1))
            scores.append(detector.score(extractor(batch)[0]))
        correct = np.concatenate(pred) == labels_for(rows, classifier.class_names)
        report[name] = summarise(correct, np.concatenate(scores) < detector.threshold)
        print(f"{name:18s} {report[name]}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
