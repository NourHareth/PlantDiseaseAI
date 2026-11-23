"""Save a Grad-CAM figure (photo, attention map, overlay) for one image.

    python -m scripts.explain assets/samples/field_tomato_leaf.png
    python -m scripts.explain my_leaf.jpg --class "Early blight" --output reports/figures/leaf.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.explain)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/explain.py)

from plantdisease import config, gradcam
from plantdisease.inference import PlantDiseaseClassifier, load_image
from plantdisease.knowledge import KnowledgeBase, normalize_label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Default: reports/figures/gradcam_<image>.png")
    parser.add_argument("--class", dest="target", default=None, help="Explain this class instead of the prediction.")
    parser.add_argument("--layer", default=None, help="Backbone layer (default: chosen automatically).")
    parser.add_argument("--model-dir", type=Path, default=config.MODELS_DIR)
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    classifier = PlantDiseaseClassifier.load(args.model_dir)
    knowledge = KnowledgeBase.from_json()
    image = load_image(args.image)
    prediction = classifier.predict(image)

    if args.target:
        keys = [normalize_label(name) for name in classifier.class_names]
        if normalize_label(args.target) not in keys:
            parser.error(f"Unknown class {args.target!r}. Choose from: {', '.join(classifier.class_names)}")
        class_index = keys.index(normalize_label(args.target))
    else:
        class_index = prediction.top_index
    target_name = knowledge.display_name(classifier.class_names[class_index])

    cam = gradcam.GradCAM(classifier.model, args.layer)
    heatmap = cam.heatmap(classifier.preprocess(image), class_index)
    heat_image, overlay = gradcam.render(image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=150, layout="constrained")
    predicted = f"{knowledge.display_name(prediction.top_label)} ({prediction.confidence:.1%})"
    panels = [
        (image, f"Photo · predicted {predicted}"),
        (heat_image, f"Grad-CAM for '{target_name}' · {heatmap.shape[0]}×{heatmap.shape[1]} map"),
        (overlay, "Overlay"),
    ]
    for ax, (picture, title) in zip(axes, panels):
        ax.imshow(picture)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=gradcam.COLORMAP), ax=axes[1],
        orientation="horizontal", fraction=0.05, pad=0.02, ticks=[0, 1],
    )
    colorbar.ax.set_xticklabels(["low influence", "high influence"], fontsize=9)
    colorbar.outline.set_visible(False)

    output = args.output or config.FIGURES_DIR / f"gradcam_{args.image.stem}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction: {prediction.top_label} ({prediction.confidence:.2%})")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
