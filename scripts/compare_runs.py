"""Paired significance tests between experiment runs (exact McNemar + Holm).

    python -m scripts.compare_runs
    python -m scripts.compare_runs --pair mnv2_finetune mnv2_finetune_224 plantdoc_all

Repeating a run with other seeds shows how much *training* randomness moves a
score. These tests answer the other half: could the difference between two
trained models be an accident of *which test images* were drawn? Both models are
scored on exactly the same images, so only the images on which they disagree
count, which makes the test much sharper than comparing two confidence intervals.

All tests are corrected together with Holm-Bonferroni. Results go to
``reports/results/comparisons.json``, which ``make_report`` turns into a table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.compare_runs)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/compare_runs.py)

import numpy as np

from plantdisease import config
from plantdisease.metrics import holm_adjust, mcnemar_exact
from plantdisease.pipeline import read_manifest

SPLITS = ("plantvillage_test", "plantdoc_all", "plantdoc_test")

# (A, B, test set, question). Each pair changes one thing; field questions use field photos.
COMPARISONS = [
    ("mnv2_frozen", "mnv2_finetune", "plantvillage_test", "Fine-tuning vs. a frozen backbone"),
    ("mnv2_scratch", "mnv2_finetune", "plantvillage_test", "ImageNet pre-training vs. random initialisation"),
    ("mnv2_finetune", "effb0_finetune", "plantvillage_test", "EfficientNetB0 vs. MobileNetV2 (lab)"),
    ("mnv2_finetune", "mnv2_finetune_noaug", "plantvillage_test", "No augmentation vs. standard (lab)"),
    ("mnv2_finetune", "mnv2_finetune_strong", "plantdoc_all", "Strong vs. standard augmentation (field)"),
    ("mnv2_finetune_strong", "mnv2_finetune_bg", "plantdoc_all", "Adding background replacement (field)"),
    ("mnv2_finetune_strong", "mnv2_finetune_bg", "plantvillage_test", "Adding background replacement (lab)"),
    ("mnv2_finetune", "mnv2_finetune_224", "plantdoc_all", "224 px vs. 128 px input (field)"),
    ("mnv2_finetune_bg", "mnv2_finetune_bg_mixed", "plantdoc_test", "Adding 677 field photos to training (field)"),
    ("baseline_v1_deployed", "mnv2_finetune_strong", "plantdoc_all", "Strong-augmentation model vs. deployed model (field)"),
]


def split_rows(split: str, pv_rows: list[dict], pd_rows: list[dict] | None) -> tuple[list[dict], Path]:
    if split == "plantvillage_test":
        return [r for r in pv_rows if r["split"] == "test"], config.RAW_DATA_DIR / "plantvillage"
    if not pd_rows:
        raise ValueError("PlantDoc manifest missing")
    rows = pd_rows if split == "plantdoc_all" else [r for r in pd_rows if r["split"] == "test"]
    return rows, config.RAW_DATA_DIR / "plantdoc"


def compare(correct: dict[tuple[str, str], np.ndarray], comparisons) -> list[dict]:
    """``correct[(run, split)]`` is a boolean array per test image; returns one row per comparison."""
    out = []
    for a, b, split, question in comparisons:
        ca, cb = correct[(a, split)], correct[(b, split)]
        test = mcnemar_exact(ca, cb)
        out.append({
            "question": question, "a": a, "b": b, "split": split, "images": int(len(ca)),
            "accuracy_a": round(float(ca.mean()), 4), "accuracy_b": round(float(cb.mean()), 4),
            "difference": round(float(cb.mean() - ca.mean()), 4),
            "only_a": test["only_a"], "only_b": test["only_b"], "p_value": test["p_value"],
        })
    for row, adjusted in zip(out, holm_adjust([r["p_value"] for r in out])):
        row["p_holm"] = float(adjusted)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "comparisons.json")
    parser.add_argument("--pair", nargs=3, action="append", metavar=("RUN_A", "RUN_B", "SPLIT"),
                        help=f"Compare two runs on one split ({', '.join(SPLITS)}); repeatable. Default: the built-in list.")
    args = parser.parse_args(argv)

    from plantdisease.evaluation import labels_for, predict_rows
    from plantdisease.inference import PlantDiseaseClassifier

    comparisons = [(a, b, s, f"{b} vs. {a}") for a, b, s in args.pair] if args.pair else COMPARISONS
    for _, _, split, _ in comparisons:
        if split not in SPLITS:
            parser.error(f"unknown split {split!r}; choose from {', '.join(SPLITS)}")
    available = [c for c in comparisons if all((args.runs_dir / n / config.MODEL_FILENAME).exists() for n in c[:2])]
    for skipped in sorted(set(comparisons) - set(available)):
        print(f"Skipping {skipped[0]} vs {skipped[1]}: run not found in {args.runs_dir}")
    if not available:
        return 1

    pv_rows = read_manifest(config.PLANTVILLAGE_MANIFEST)
    pd_rows = read_manifest(config.PLANTDOC_MANIFEST) if config.PLANTDOC_MANIFEST.exists() else None
    needed: dict[str, set[str]] = {}
    for a, b, split, _ in available:
        needed.setdefault(a, set()).add(split)
        needed.setdefault(b, set()).add(split)

    correct = {}
    for run, splits in needed.items():
        classifier = PlantDiseaseClassifier.load(args.runs_dir / run)
        for split in sorted(splits):
            rows, root = split_rows(split, pv_rows, pd_rows)
            pred = predict_rows(classifier, rows, root).argmax(axis=1)
            correct[(run, split)] = pred == labels_for(rows, classifier.class_names)
            print(f"{run:28s} {split:18s} {correct[(run, split)].mean():.2%} of {len(rows)}", flush=True)

    rows = compare(correct, available)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"comparisons": rows}, indent=2), encoding="utf-8")
    print()
    for r in rows:
        print(f"{r['question']:55s} {r['accuracy_a']:.1%} -> {r['accuracy_b']:.1%} "
              f"(disagree {r['only_a']}/{r['only_b']}) p={r['p_value']:.3g} holm={r['p_holm']:.3g}")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
