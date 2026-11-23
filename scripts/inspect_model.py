"""Print the model's structure and check that all artifacts are consistent.

    python -m scripts.inspect_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.inspect_model)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/inspect_model.py)

from plantdisease import config
from plantdisease.gradcam import GradCAM
from plantdisease.inference import PlantDiseaseClassifier
from plantdisease.knowledge import KnowledgeBase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    args = parser.parse_args(argv)

    classifier = PlantDiseaseClassifier.load(args.model_dir)  # fails loudly on class-count mismatch
    model = classifier.model
    facts = classifier.describe()

    print("=== Layers ===")
    for layer in model.layers:
        print(f"  {layer.__class__.__name__:24s} {layer.name:28s} trainable={layer.trainable}")
    print("\n=== Summary ===")
    for key, value in facts.items():
        print(f"  {key:20s} {value}")
    print(f"  {'output_shape':20s} {model.output_shape}")

    print("\n=== Classes (model output order) ===")
    knowledge = KnowledgeBase.from_json()
    for i, name in enumerate(classifier.class_names):
        print(f"  {i:2d}  {name:40s} -> {knowledge.display_name(name)}")
    missing = knowledge.missing(list(classifier.class_names))
    print("  disease guide:", "complete" if not missing else f"MISSING for {missing}")

    print("\n=== Grad-CAM ===")
    cam = GradCAM(model)
    print(f"  layer {cam.layer_name}: ready")

    info = classifier.info
    print("\n=== model_info.json ===")
    print(f"  {info.architecture}")
    if info.evaluation:
        e = info.evaluation
        print(f"  {e.split} accuracy {e.accuracy:.2%} on {e.num_images} images ({e.source})")
    if info.field_accuracy is not None:
        print(f"  field (PlantDoc) accuracy {info.field_accuracy:.2%}")
    print(f"  calibration: {'temperature %.3f' % info.temperature if info.is_calibrated else 'none'}")
    detector = args.model_dir / config.OOD_DETECTOR_FILENAME
    print(f"  unfamiliar-image detector: {'present' if detector.exists() else 'absent'}")
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
