"""Full evaluation of a trained model: accuracy, field generalisation,
calibration, OOD detection, explanation quality and speed.

Images are loaded and preprocessed exactly as in the app (``load_image`` +
``preprocess``), so the numbers describe what a user actually gets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from plantdisease import config
from plantdisease.calibration import apply_temperature, fit_temperature
from plantdisease.inference import PlantDiseaseClassifier, load_image
from plantdisease.knowledge import normalize_label
from plantdisease.metrics import (
    auroc,
    bootstrap_accuracy_ci,
    classification_report,
    expected_calibration_error,
    fpr_at_tpr,
    negative_log_likelihood,
    reliability_bins,
)
from plantdisease.ood import FeatureExtractor, MahalanobisDetector, energy_score, msp_score, synthetic_ood_images

BATCH = 64


def load_batch(classifier: PlantDiseaseClassifier, rows: list[dict], root: Path) -> np.ndarray:
    return np.concatenate([classifier.preprocess(load_image(Path(root) / r["path"])) for r in rows])


def predict_rows(
    classifier: PlantDiseaseClassifier, rows: list[dict], root: Path, extractor: FeatureExtractor | None = None
):
    """Probabilities (and optionally features/logits) for manifest rows, in order."""
    probs, feats, logits = [], [], []
    for start in range(0, len(rows), BATCH):
        batch = load_batch(classifier, rows[start : start + BATCH], root)
        probs.append(classifier.predict_batch(batch))
        if extractor is not None:
            f, z = extractor(batch)
            feats.append(f)
            logits.append(z)
    probs = np.concatenate(probs)
    if extractor is None:
        return probs
    return probs, np.concatenate(feats), np.concatenate(logits)


def labels_for(rows: list[dict], class_names: tuple[str, ...]) -> np.ndarray:
    index = {normalize_label(name): i for i, name in enumerate(class_names)}
    return np.array([index[normalize_label(r["label"])] for r in rows])


def stratified_sample(rows: list[dict], count: int, seed: int = 0) -> list[dict]:
    """About ``count`` rows with the class proportions of ``rows`` (at least one per class)."""
    if len(rows) <= count:
        return rows
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    picked = []
    for label_rows in by_label.values():
        k = max(1, round(count * len(label_rows) / len(rows)))
        picked.extend(label_rows[i] for i in rng.choice(len(label_rows), size=min(k, len(label_rows)), replace=False))
    return picked


def background_only_images(rows: list[dict], root: Path, size: int, seed: int = 0) -> np.ndarray:
    """PlantVillage photos with the leaf removed (filled with the photo's own background
    colour plus noise): images of the right *setting* that contain no leaf at all."""
    rng = np.random.default_rng(seed)
    out = []
    for row in rows:
        photo = np.asarray(load_image(Path(root) / row["path"]).resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        segmented = np.asarray(load_image(Path(root) / row["mask_path"]).resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        leaf = segmented.max(axis=-1) > 0.08
        background = photo[~leaf].mean(axis=0) if (~leaf).any() else np.full(3, 0.5)
        filled = photo.copy()
        filled[leaf] = np.clip(background + rng.normal(0, 0.03, (leaf.sum(), 3)), 0, 1)
        out.append(filled)
    return np.stack(out).astype(np.float32)


def leaf_attention(classifier: PlantDiseaseClassifier, rows: list[dict], root: Path) -> dict:
    """Share of Grad-CAM mass inside the leaf mask, vs. the leaf's share of the image."""
    from plantdisease.gradcam import GradCAM

    cam = GradCAM(classifier.model)
    on_leaf, leaf_area = [], []
    for row in rows:
        image = load_image(Path(root) / row["path"])
        batch = classifier.preprocess(image)
        heatmap = cam.heatmap(batch, int(classifier.predict_batch(batch)[0].argmax()))
        mask_img = load_image(Path(root) / row["mask_path"]).resize(heatmap.shape[::-1], Image.Resampling.BILINEAR)
        mask = np.asarray(mask_img, dtype=np.float32).max(axis=-1) / 255 > 0.08
        if heatmap.sum() > 0:
            on_leaf.append(float((heatmap * mask).sum() / heatmap.sum()))
            leaf_area.append(float(mask.mean()))
    if not on_leaf:  # every heatmap was empty (e.g. an untrained model)
        return {"images": 0, "layer": cam.layer_name, "attention_on_leaf": None, "leaf_area_fraction": None, "lift": None}
    return {
        "images": len(on_leaf),
        "layer": cam.layer_name,
        "attention_on_leaf": round(float(np.mean(on_leaf)), 4),
        "leaf_area_fraction": round(float(np.mean(leaf_area)), 4),
        "lift": round(float(np.mean(on_leaf) / np.mean(leaf_area)), 3),  # >1: attends to the leaf more than chance
    }


def speed(classifier: PlantDiseaseClassifier, model_path: Path, repeats: int = 30) -> dict:
    x = np.random.default_rng(0).random((1, classifier.image_size, classifier.image_size, 3), dtype=np.float32)
    for _ in range(3):
        classifier.predict_batch(x)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        classifier.predict_batch(x)
        times.append((time.perf_counter() - t) * 1000)
    return {
        "latency_ms_median": round(float(np.median(times)), 2),
        "params": int(classifier.model.count_params()),
        "model_mb": round(model_path.stat().st_size / 1e6, 2),
    }


def _split_report(probs, y, class_names, temperature) -> dict:
    report = classification_report(y, probs.argmax(axis=1), list(class_names))
    low, high = bootstrap_accuracy_ci(y, probs.argmax(axis=1))
    calibrated = apply_temperature(probs, temperature)
    report.update({
        "accuracy_ci95": [round(low, 4), round(high, 4)],
        "ece": round(expected_calibration_error(probs, y), 4),
        "ece_calibrated": round(expected_calibration_error(calibrated, y), 4),
        "nll": round(negative_log_likelihood(probs, y), 4),
        "nll_calibrated": round(negative_log_likelihood(calibrated, y), 4),
        "mean_confidence": round(float(probs.max(axis=1).mean()), 4),
    })
    return report


def evaluate_run(
    model_dir: Path,
    pv_rows: list[dict],
    pv_root: Path,
    plantdoc_rows: list[dict] | None = None,
    plantdoc_root: Path | None = None,
    *,
    plantdoc_trained: bool = False,
    ood_samples: int = 500,
    train_feature_samples: int = 3000,
    attention_samples: int = 300,
    save_detector: bool = True,
    log=print,
) -> dict:
    classifier = PlantDiseaseClassifier.load(model_dir)
    classes = classifier.class_names
    extractor = FeatureExtractor(classifier.model)
    split = {name: [r for r in pv_rows if r["split"] == name] for name in ("train", "val", "test")}
    results: dict = {"model_dir": str(model_dir), "class_names": list(classes)}

    log("  validation set -> temperature")
    val_probs = predict_rows(classifier, split["val"], pv_root)
    y_val = labels_for(split["val"], classes)
    temperature = fit_temperature(val_probs, y_val)
    results["temperature"] = round(temperature, 4)
    results["val_accuracy"] = round(float((val_probs.argmax(1) == y_val).mean()), 4)

    log("  PlantVillage test")
    test_probs, test_feats, test_logits = predict_rows(classifier, split["test"], pv_root, extractor)
    y_test = labels_for(split["test"], classes)
    results["plantvillage_test"] = _split_report(test_probs, y_test, classes, temperature)
    conf, acc, counts = reliability_bins(test_probs, y_test)
    conf_t, acc_t, counts_t = reliability_bins(apply_temperature(test_probs, temperature), y_test)
    results["reliability"] = {
        "raw": {"confidence": conf.round(4).tolist(), "accuracy": acc.round(4).tolist(), "count": counts.tolist()},
        "calibrated": {"confidence": conf_t.round(4).tolist(), "accuracy": acc_t.round(4).tolist(), "count": counts_t.tolist()},
    }

    if plantdoc_rows:
        # PlantDoc's test split is small (64 tomato images). Models that never trained on
        # PlantDoc can fairly be scored on all of it, which gives a much tighter estimate.
        log("  PlantDoc (field photos)")
        field_probs = predict_rows(classifier, plantdoc_rows, plantdoc_root)
        y_field = labels_for(plantdoc_rows, classes)
        is_test = np.array([r["split"] == "test" for r in plantdoc_rows])
        results["plantdoc_test"] = _split_report(field_probs[is_test], y_field[is_test], classes, temperature)
        if not plantdoc_trained:
            results["plantdoc_all"] = _split_report(field_probs, y_field, classes, temperature)

    log("  out-of-distribution detection")
    train_sample = stratified_sample(split["train"], train_feature_samples)
    _, train_feats, _ = predict_rows(classifier, train_sample, pv_root, extractor)
    detector = MahalanobisDetector.fit(train_feats, labels_for(train_sample, classes))
    _, val_feats, _ = predict_rows(classifier, stratified_sample(split["val"], 1000), pv_root, extractor)
    detector.calibrate_threshold(detector.score(val_feats))
    if save_detector:
        detector.save(model_dir / "ood_detector.npz")

    test_with_masks = [r for r in split["test"] if r.get("mask_path")]
    ood_sets = {
        "synthetic": synthetic_ood_images(ood_samples, classifier.image_size),
        "background_only": background_only_images(
            stratified_sample(test_with_masks, ood_samples, seed=1), pv_root, classifier.image_size
        ),
    }
    in_scores = {
        "msp": msp_score(test_probs),
        "energy": energy_score(test_logits),
        "mahalanobis": detector.score(test_feats),
    }
    results["ood"] = {}
    for set_name, images in ood_sets.items():
        probs_out = np.concatenate([classifier.predict_batch(images[i : i + BATCH]) for i in range(0, len(images), BATCH)])
        feats_out, logits_out = zip(*(extractor(images[i : i + BATCH]) for i in range(0, len(images), BATCH)))
        out_scores = {
            "msp": msp_score(probs_out),
            "energy": energy_score(np.concatenate(logits_out)),
            "mahalanobis": detector.score(np.concatenate(feats_out)),
        }
        results["ood"][set_name] = {
            method: {
                "auroc": round(auroc(in_scores[method], out_scores[method]), 4),
                "fpr_at_95_tpr": round(fpr_at_tpr(in_scores[method], out_scores[method]), 4),
            }
            for method in in_scores
        }
        results["ood"][set_name]["mean_msp"] = round(float(out_scores["msp"].mean()), 4)

    log("  Grad-CAM leaf attention")
    if test_with_masks:
        results["gradcam"] = leaf_attention(classifier, stratified_sample(test_with_masks, attention_samples, seed=2), pv_root)

    results["speed"] = speed(classifier, model_dir / config.MODEL_FILENAME)

    # Record calibration + evaluation in the run's model_info.json (shown in the app)
    info_path = model_dir / config.MODEL_INFO_FILENAME
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info["temperature"] = results["temperature"]
    info["evaluation"] = {
        "split": "test",
        "accuracy": results["plantvillage_test"]["accuracy"],
        "num_images": results["plantvillage_test"]["num_images"],
        "source": "Measured with plantdisease.evaluation on the leakage-aware PlantVillage test split.",
    }
    field = results.get("plantdoc_all") or results.get("plantdoc_test")
    if field:
        info["field_accuracy"] = field["accuracy"]
        info["field_images"] = field["num_images"]
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    (model_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
