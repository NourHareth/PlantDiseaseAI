"""Export a model to TensorFlow Lite and measure size, speed and accuracy.

    python -m scripts.export_tflite --model-dir models
    python -m scripts.export_tflite --model-dir <workdir>/runs/mnv2_finetune --eval-images 1000

Three variants are produced:

* ``float32``: plain conversion;
* ``dynamic``: weights quantised to int8, activations float (no data needed);
* ``int8``: full integer quantisation calibrated on 200 training images.

Training-only layers (augmentation, dropout) are stripped before export.
Results go to ``reports/results/tflite_<name>.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.export_tflite)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/export_tflite.py)

import numpy as np

from plantdisease import config
from plantdisease.evaluation import labels_for, load_batch, stratified_sample
from plantdisease.inference import PlantDiseaseClassifier
from plantdisease.pipeline import read_manifest


def inference_model(model):
    """Same weights, without augmentation and dropout layers."""
    import tensorflow as tf

    layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    keep = [l for l in layers if l.name != "data_augmentation" and not isinstance(l, tf.keras.layers.Dropout)]
    inputs = tf.keras.Input(shape=model.input_shape[1:], name="image")
    x = inputs
    for layer in keep:
        x = layer(x, training=False) if isinstance(layer, tf.keras.Model) else layer(x)
    return tf.keras.Model(inputs, x, name="inference")


def convert(model, variant: str, calibration: np.ndarray | None) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if variant in ("dynamic", "int8"):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if variant == "int8":
        def representative():
            for image in calibration:
                yield [image[None].astype(np.float32)]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32  # keep the [0, 1] float contract at the boundary
        converter.inference_output_type = tf.float32
    with contextlib.redirect_stdout(io.StringIO()):  # the converter prints its whole SavedModel signature
        return converter.convert()


def run_tflite(model_bytes: bytes, images: np.ndarray) -> tuple[np.ndarray, float]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    probs, times = [], []
    for image in images:
        interpreter.set_tensor(inp["index"], image[None].astype(np.float32))
        start = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - start) * 1000)
        probs.append(interpreter.get_tensor(out["index"])[0])
    return np.stack(probs), float(np.median(times))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--name", default=None, help="Name used for output files (default: folder name).")
    parser.add_argument("--eval-images", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=None, help="Where .tflite files go (default: the model dir).")
    args = parser.parse_args(argv)

    name = args.name or args.model_dir.resolve().name
    out_dir = args.out_dir or args.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    classifier = PlantDiseaseClassifier.load(args.model_dir)
    rows = read_manifest(config.PLANTVILLAGE_MANIFEST)
    root = config.RAW_DATA_DIR / "plantvillage"
    test_rows = stratified_sample([r for r in rows if r["split"] == "test"], args.eval_images, seed=3)
    calib_rows = stratified_sample([r for r in rows if r["split"] == "train"], 200, seed=4)
    test_images = load_batch(classifier, test_rows, root)
    y = labels_for(test_rows, classifier.class_names)
    calibration = load_batch(classifier, calib_rows, root)

    keras_probs = np.concatenate([classifier.predict_batch(test_images[i:i + 64]) for i in range(0, len(test_images), 64)])
    keras_pred = keras_probs.argmax(1)
    report = {
        "model": name,
        "eval_images": len(test_rows),
        "keras": {
            "accuracy": round(float((keras_pred == y).mean()), 4),
            "model_mb": round((args.model_dir / config.MODEL_FILENAME).stat().st_size / 1e6, 2),
        },
    }
    model = inference_model(classifier.model)
    for variant in ("float32", "dynamic", "int8"):
        print(f"Converting {variant} ...", flush=True)
        model_bytes = convert(model, variant, calibration)
        path = out_dir / f"plant_disease_{variant}.tflite"
        path.write_bytes(model_bytes)
        probs, latency = run_tflite(model_bytes, test_images)
        pred = probs.argmax(1)
        report[variant] = {
            "accuracy": round(float((pred == y).mean()), 4),
            "agreement_with_keras": round(float((pred == keras_pred).mean()), 4),
            "model_mb": round(len(model_bytes) / 1e6, 2),
            "latency_ms_median": round(latency, 2),
        }
        print(f"  {variant}: {report[variant]}", flush=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.RESULTS_DIR / f"tflite_{name}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
