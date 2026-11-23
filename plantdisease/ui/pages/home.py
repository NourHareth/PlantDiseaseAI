"""Landing page."""

from __future__ import annotations

import streamlit as st

from plantdisease.ui import components, navigation, resources


def render() -> None:
    classifier = resources.get_classifier()

    components.hero(
        "Plant health, powered by AI 🌱",
        "Detect tomato leaf diseases from a photo. Upload a picture of a single leaf to get "
        "a prediction, see which parts of the image the model relied on, and read practical next steps.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Supported classes", classifier.num_classes)
    col2.metric("Test accuracy", components.format_accuracy(classifier))
    col3.metric("Parameters", f"{classifier.model.count_params() / 1e6:.2f} M",
                help=f"{classifier.info.backbone} backbone plus the classification head.")
    col4.metric("Input size", f"{classifier.image_size} px")

    st.markdown("## How it works")
    step1, step2, step3 = st.columns(3)
    with step1:
        components.card(
            "1 · Upload",
            ["Take a clear, well-lit photo of one tomato leaf and upload it."],
            equal_height=True,
        )
    with step2:
        components.card(
            "2 · Analyze",
            ["The image is checked, resized and classified by a MobileNetV2 network into one of the supported classes."],
            equal_height=True,
        )
    with step3:
        components.card(
            "3 · Understand",
            ["Review the confidence, the runner-up classes, a Grad-CAM attention map and the disease guide."],
            equal_height=True,
        )

    if st.button("Start a diagnosis", type="primary", icon=":material/arrow_forward:", key="start_diagnosis"):
        st.switch_page(navigation.get_pages()["diagnose"])
    st.caption(
        "PlantDisease AI is an academic demonstration. Predictions do not replace inspection "
        "by an agronomist or a laboratory test."
    )
