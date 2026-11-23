"""How the model was built, how well it does, and where it falls short."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from plantdisease.ui import components, resources

def limitations(info, detector_installed: bool) -> list[str]:
    field = ""
    if info.field_accuracy is not None:
        images = f" ({info.field_images:,} PlantDoc images)" if info.field_images else ""
        field = f" On real field photos it was measured at {info.field_accuracy:.0%} accuracy{images}."
    return [
        "Trained on lab photos of single leaves on plain backgrounds. Field photos with hands, soil "
        "or several leaves look different, and accuracy on them is far lower than the test score." + field,
        "The model always answers with one of its 10 classes. The app adds input checks for blank, "
        "noisy or dark images"
        + (" and a detector for photos unlike the training images" if detector_installed else "")
        + ", but none of these can prove that a photo shows a tomato leaf.",
        "Confidence is calibrated on held-out lab photos only. On field photos the model is far more "
        "confident than it is accurate."
        if info.is_calibrated else
        "Confidence scores are not calibrated: 99% does not mean the prediction is right 99% of the time.",
        "Only tomato leaves are supported. Other crops, fruit and stems are out of scope.",
    ]


def render() -> None:
    classifier = resources.get_classifier()
    knowledge = resources.get_knowledge()
    info = classifier.info
    facts = classifier.describe()

    components.hero(
        "About the model 🧠",
        "How PlantDisease AI was built, how well it performs, and where it falls short.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test accuracy", components.format_accuracy(classifier))
    col2.metric(
        "Test images",
        f"{info.evaluation.num_images:,}" if info.evaluation else "Not measured",
    )
    col3.metric("Classes", facts["num_classes"])
    col4.metric("Parameters", f"{facts['total_params'] / 1e6:.2f} M")

    arch_col, data_col = st.columns(2)
    with arch_col:
        backbone_state = "frozen" if facts["backbone_trainable"] is False else "fine-tuned"
        components.card(
            "Architecture",
            [
                info.architecture,
                f"Pre-trained weights: {info.pretrained_weights}. Backbone {backbone_state}.",
                f"Trainable parameters: {facts['trainable_params']:,} of {facts['total_params']:,}.",
                f"Input: {facts['input_size']} × {facts['input_size']} RGB, scaled to [0, 1].",
            ],
        )
    with data_col:
        paragraphs = [f"Dataset: {info.dataset}", f"Training: {info.training}"]
        if info.evaluation:
            paragraphs.append(
                f"Evaluated on {info.evaluation.num_images:,} {info.evaluation.split} images: "
                f"{info.evaluation.accuracy:.2%} accuracy."
            )
            if info.evaluation.source:
                paragraphs.append(info.evaluation.source)
        if info.field_accuracy is not None:
            paragraphs.append(f"Field photos (PlantDoc): {info.field_accuracy:.2%} accuracy.")
        paragraphs.append(
            f"Confidence calibrated with temperature scaling (T = {info.temperature:.2f})."
            if info.is_calibrated else "Confidence scores are not calibrated."
        )
        paragraphs.append(
            "Unfamiliar-image detector: active (Mahalanobis distance in feature space)."
            if resources.get_ood_detector() is not None
            else "Unfamiliar-image detector: not installed for this model."
        )
        components.card("Data & evaluation", paragraphs)

    st.markdown("## Supported classes")
    rows = []
    for index, label in enumerate(classifier.class_names):
        entry = knowledge.get(label)
        rows.append(
            {
                "#": index,
                "Class": entry.display_name,
                "Type": entry.category.capitalize(),
                "Cause": entry.cause,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("## Known limitations")
    components.card("Read before trusting a prediction",
                    bullets=limitations(info, resources.get_ood_detector() is not None))

    st.warning(
        "PlantDisease AI is an academic demonstration tool. Predictions should not replace "
        "professional agricultural inspection or laboratory diagnosis.",
        icon=":material/gavel:",
    )
