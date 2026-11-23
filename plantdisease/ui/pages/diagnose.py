"""Upload a leaf photo and get a prediction, an attention map and a disease guide."""

from __future__ import annotations

import hashlib

import streamlit as st
from PIL import Image

from plantdisease.gradcam import render as render_gradcam
from plantdisease.inference import ImageLoadError, Prediction, load_image
from plantdisease.knowledge import DiseaseInfo, KnowledgeBase
from plantdisease.quality import Assessment, Issue, assess
from plantdisease.ui import charts, components, resources

SAMPLE_NAME = "Sample field photo"

_ASSESSMENT_STYLE = {
    "high": (st.success, ":material/verified:"),
    "moderate": (st.info, ":material/info:"),
    "low": (st.warning, ":material/help:"),
}


# ---------------------------------------------------------------------------
# Which image is active: the last thing the user did (upload or sample) wins.
# ---------------------------------------------------------------------------
def _use_upload() -> None:
    st.session_state["image_source"] = "upload"


def _use_sample() -> None:
    st.session_state["image_source"] = "sample"


def _selected_image(uploaded) -> tuple[str, bytes] | None:
    if st.session_state.get("image_source") == "sample":
        return SAMPLE_NAME, resources.sample_image_bytes()
    if uploaded is not None:
        return uploaded.name, uploaded.getvalue()
    return None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def render() -> None:
    components.hero(
        "Leaf disease scanner 🔬",
        "Upload a clear photo of a single tomato leaf. The model returns the most likely class, "
        "how sure it is, and which parts of the image it relied on.",
    )
    knowledge = resources.get_knowledge()

    upload_col, tips_col = st.columns([3, 2], gap="large")
    with upload_col:
        uploaded = st.file_uploader(
            "Leaf photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="leaf_upload",
            on_change=_use_upload,
            help="JPG, PNG or WEBP. One leaf per photo works best.",
        )
        st.button(
            "Try the sample photo",
            key="sample_button",
            icon=":material/photo_library:",
            on_click=_use_sample,
        )
    with tips_col:
        components.card(
            "For the best result",
            bullets=[
                "Photograph one leaf and let it fill most of the frame",
                "Use even daylight and avoid blur and harsh shadows",
                "Prefer a plain background: the model was trained on lab photos",
                "Show the side of the leaf where symptoms are visible",
            ],
        )

    selected = _selected_image(uploaded)
    if selected is None:
        st.info("Upload a photo or try the sample to start.", icon=":material/upload:")
        return
    name, data = selected

    try:
        image = load_image(data)
        with st.spinner("Analyzing the leaf..."):
            prediction, issues = resources.analyze(data)
    except ImageLoadError as exc:
        st.error(str(exc), icon=":material/broken_image:")  # each message carries its own advice
        return

    st.divider()
    assessment = assess(prediction, issues)
    info = knowledge.get(prediction.top_label)
    image_key = hashlib.sha1(data).hexdigest()[:12]

    image_col, result_col = st.columns([1, 1.1], gap="large")
    with image_col:
        st.image(image, caption=f"{name} · {image.width} × {image.height} px", width="stretch")
    with result_col:
        shown = _render_result(prediction, assessment, issues, info, knowledge, image_key)
    if not shown:
        return

    st.markdown("")
    guide_tab, cam_tab, probs_tab = st.tabs(
        [":material/menu_book: Disease guide", ":material/visibility: Model attention", ":material/bar_chart: All classes"]
    )
    with guide_tab:
        _render_guide(info)
    with cam_tab:
        _render_gradcam(image, data, prediction, info)
    with probs_tab:
        _render_probabilities(prediction, knowledge)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _render_result(
    prediction: Prediction,
    assessment: Assessment,
    issues: list[Issue],
    info: DiseaseInfo,
    knowledge: KnowledgeBase,
    image_key: str,
) -> bool:
    """Draw the result summary. Returns False when the result is withheld."""
    components.eyebrow("Analysis result")

    if assessment.level == "unreliable":
        for issue in issues:
            if issue.is_blocking:
                st.error(issue.message, icon=":material/block:")
        st.markdown(assessment.message)
        if not st.toggle("Show the model's answer anyway", key=f"show_anyway_{image_key}"):
            return False

    components.result_name(info.display_name)
    components.category_badge(info.category)
    calibrated = resources.get_classifier().info.is_calibrated
    st.metric(
        "Calibrated confidence" if calibrated else "Model confidence",
        charts.format_probability(prediction.confidence),
        help="Temperature-scaled so that, on held-out lab photos, X% confidence means about X% accuracy."
        if calibrated else "The model's raw softmax score. It has not been calibrated.",
    )
    st.progress(prediction.confidence)

    if assessment.level in _ASSESSMENT_STYLE:
        show, icon = _ASSESSMENT_STYLE[assessment.level]
        show(f"**{assessment.title}.** {assessment.message}", icon=icon)
    for issue in issues:
        if not issue.is_blocking:
            st.warning(issue.message, icon=":material/warning:")

    runners_up = ", ".join(
        f"{knowledge.display_name(label)} ({charts.format_probability(probability)})"
        for _, label, probability in prediction.ranked(3)[1:]
    )
    st.caption(
        f"Next most likely: {runners_up}. Confidence is the model's own score for its top class: "
        "it is not the probability that the diagnosis is correct, and it says nothing about severity."
    )
    return True


def _render_guide(info: DiseaseInfo) -> None:
    about_col, symptoms_col, actions_col = st.columns(3)
    with about_col:
        components.card("About", [info.description, f"Cause: {info.cause}"])
    with symptoms_col:
        components.card("Typical symptoms", bullets=info.symptoms)
    with actions_col:
        components.card("What you can do", bullets=info.actions)
    if info.category in ("disease", "pest"):
        st.caption(
            "General guidance only. Confirm the diagnosis with a local extension service or "
            "agronomist before applying any treatment."
        )


def _render_gradcam(image: Image.Image, data: bytes, prediction: Prediction, info: DiseaseInfo) -> None:
    st.markdown(
        f"Grad-CAM highlights the regions that pushed the model towards **{info.display_name}**. "
        "It shows where the model looked, not where the disease is. If the bright areas sit on the "
        "background or on fingers instead of the lesions, the model is using the wrong cues and "
        "the prediction deserves less trust."
    )
    try:
        heatmap = resources.explain(data, prediction.top_index)
    except Exception as exc:  # the explanation is optional; never break the page over it
        st.warning("The attention map could not be generated for this image.", icon=":material/warning:")
        st.caption(f"Technical detail: {exc}")
        return

    heat_image, overlay = render_gradcam(image, heatmap)
    overlay_col, heat_col = st.columns(2, gap="medium")
    overlay_col.image(overlay, caption="Attention overlaid on the photo", width="stretch")
    heat_col.image(heat_image, caption="Attention map", width="stretch")
    components.heat_legend()
    rows, cols = heatmap.shape
    st.caption(
        f"Computed on a {rows} × {cols} feature map (layer `{resources.get_gradcam().layer_name}`) "
        "and upscaled, so region boundaries are approximate."
    )


def _render_probabilities(prediction: Prediction, knowledge: KnowledgeBase) -> None:
    table = charts.probability_table(prediction, knowledge)
    st.altair_chart(charts.probability_chart(table), width="stretch")
    with st.expander("Show as a table"):
        st.dataframe(
            table,
            hide_index=True,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability", format="percent", min_value=0.0, max_value=1.0
                )
            },
        )
