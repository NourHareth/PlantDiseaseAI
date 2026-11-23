"""Run the experiment suite: train + fully evaluate every run, resumably.

    python -m scripts.run_experiments                      # whole suite
    python -m scripts.run_experiments --only mnv2_finetune mnv2_frozen
    python -m scripts.run_experiments --evaluate-baseline  # also score models/ (the deployed model)

Each run is trained into ``<workdir>/runs/<name>/`` and evaluated with
``plantdisease.evaluation`` (PlantVillage test, PlantDoc field photos,
calibration, OOD detection, Grad-CAM leaf attention, speed). A compact copy of
every result lands in ``reports/results/<name>.json`` (versioned in git).
Finished runs are skipped, so the suite can be stopped and restarted.
Afterwards build the report with ``python -m scripts.make_report``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.run_experiments)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/run_experiments.py)

from plantdisease import config
from plantdisease.pipeline import read_manifest

SUITE_PATH = config.PROJECT_ROOT / "experiments" / "suite.json"
BASELINE_NAME = "baseline_v1_deployed"


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def compact(results: dict, run: dict) -> dict:
    """Small summary for reports/results (no confusion matrices of every split)."""
    keep = {k: results[k] for k in ("temperature", "val_accuracy", "ood", "gradcam", "speed", "class_names") if k in results}
    for split in ("plantvillage_test", "plantdoc_all", "plantdoc_test"):
        if split in results:
            keep[split] = {k: v for k, v in results[split].items() if k != "per_class"}
            keep[split]["per_class_f1"] = {c: m["f1"] for c, m in results[split]["per_class"].items()}
    keep["reliability"] = results.get("reliability")
    keep["run"] = run
    return keep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", type=Path, default=SUITE_PATH)
    parser.add_argument("--only", nargs="*", default=None, help="Run just these experiment names.")
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--evaluate-baseline", action="store_true", help="Evaluate the deployed model in models/ too.")
    parser.add_argument("--retrain", action="store_true", help="Ignore finished runs and train again.")
    parser.add_argument("--no-report", action="store_true", help="Skip rebuilding the report at the end.")
    args = parser.parse_args(argv)

    from plantdisease.evaluation import evaluate_run
    from plantdisease.training import TrainConfig, train

    pv_root = config.RAW_DATA_DIR / "plantvillage"
    pd_root = config.RAW_DATA_DIR / "plantdoc"
    pv_rows = read_manifest(config.PLANTVILLAGE_MANIFEST)
    pd_rows = read_manifest(config.PLANTDOC_MANIFEST) if config.PLANTDOC_MANIFEST.exists() else None
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_baseline:
        target = args.results_dir / f"{BASELINE_NAME}.json"
        if args.retrain or not target.exists():
            log(f"Evaluating the deployed model ({config.MODELS_DIR})")
            baseline_dir = args.runs_dir / BASELINE_NAME
            baseline_dir.mkdir(parents=True, exist_ok=True)
            for name in (config.MODEL_FILENAME, config.CLASS_NAMES_FILENAME, config.MODEL_INFO_FILENAME):
                shutil.copy2(config.MODELS_DIR / name, baseline_dir / name)
            results = evaluate_run(baseline_dir, pv_rows, pv_root, pd_rows, pd_root, log=log)
            run = {"name": BASELINE_NAME, "group": "baseline", "backbone": "mobilenetv2", "strategy": "frozen",
                   "note": "Original deployed model; its training split is unknown, so PlantVillage test images "
                           "may have been seen in training (PlantDoc numbers are unaffected)."}
            target.write_text(json.dumps(compact(results, run), indent=2), encoding="utf-8")

    suite = json.loads(args.suite.read_text(encoding="utf-8"))["runs"]
    if args.only:
        unknown = set(args.only) - {r["name"] for r in suite}
        if unknown:
            parser.error(f"unknown runs: {', '.join(sorted(unknown))}")
        suite = [r for r in suite if r["name"] in args.only]

    failures = 0
    for i, run in enumerate(suite, start=1):
        name = run["name"]
        target = args.results_dir / f"{name}.json"
        if target.exists() and not args.retrain:
            log(f"({i}/{len(suite)}) {name}: done already, skipping")
            continue
        run_dir = args.runs_dir / name
        try:
            cfg = TrainConfig.from_dict(run)
            train_rows = [r for r in pv_rows if r["split"] == "train"]
            roots = {"plantvillage": pv_root, "plantdoc": pd_root}
            if "plantdoc" in cfg.extra_train_sources and pd_rows:
                train_rows = train_rows + [dict(r, source="plantdoc") for r in pd_rows if r["split"] == "train"]
            started = time.time()
            if args.retrain or not (run_dir / config.MODEL_FILENAME).exists():
                log(f"({i}/{len(suite)}) {name}: training {json.dumps(run)}")
                train(cfg, train_rows, [r for r in pv_rows if r["split"] == "val"], roots, run_dir)
            log(f"({i}/{len(suite)}) {name}: evaluating")
            results = evaluate_run(run_dir, pv_rows, pv_root, pd_rows, pd_root, log=log,
                                   plantdoc_trained="plantdoc" in cfg.extra_train_sources)
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            run_info = dict(run, train_seconds=history["train_seconds"],
                            epochs_run=len(history["history"]["loss"]),
                            trainable_params=history["trainable_params_final"])
            target.write_text(json.dumps(compact(results, run_info), indent=2), encoding="utf-8")
            pv = results["plantvillage_test"]["accuracy"]
            field_all = results.get("plantdoc_all", {}).get("accuracy")
            field_test = results.get("plantdoc_test", {}).get("accuracy")
            log(f"({i}/{len(suite)}) {name}: PlantVillage {pv:.2%}"
                + (f" | PlantDoc all {field_all:.2%}" if field_all is not None else "")
                + (f" | PlantDoc test {field_test:.2%}" if field_test is not None else "")
                + f" | {time.time() - started:.0f}s")
        except Exception:
            failures += 1
            log(f"({i}/{len(suite)}) {name}: FAILED")
            traceback.print_exc()
    log(f"Suite finished with {failures} failure(s).")
    if not args.no_report and any(args.results_dir.glob("*.json")):
        from scripts import make_report

        log("Building the report")
        make_report.main(["--results-dir", str(args.results_dir)])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
