"""Small, reusable building blocks for the pages.

Every piece of text inserted into raw HTML goes through :func:`html.escape`.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from plantdisease import gradcam
from plantdisease.inference import PlantDiseaseClassifier
from plantdisease.ui import resources

CATEGORY_BADGES = {
    "disease": ("Disease", ":material/microbiology:", "red"),  # a pathogen, not the COVID glyph
    "pest": ("Pest", ":material/pest_control:", "orange"),
    "healthy": ("Healthy", ":material/check_circle:", "green"),
    "unknown": ("Not documented", ":material/help:", "gray"),
}


def inject_css() -> None:
    st.html(f"<style>{resources.stylesheet()}</style>")


def hero(title: str, subtitle: str) -> None:
    # Glue the last word to the one before it (usually an emoji) so it never wraps onto a line alone.
    head, _, last = title.rpartition(" ")
    title_html = f"{escape(head)}&nbsp;{escape(last)}" if head else escape(title)
    st.markdown(
        f'<div class="pd-hero"><h1>{title_html}</h1><p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def card(
    title: str,
    paragraphs: list[str] | tuple[str, ...] = (),
    bullets: list[str] | tuple[str, ...] = (),
    *,
    equal_height: bool = False,
) -> None:
    """A titled card. ``equal_height`` gives cards in one row the same minimum height."""
    body = "".join(f"<p>{escape(p)}</p>" for p in paragraphs)
    if bullets:
        body += "<ul>" + "".join(f"<li>{escape(b)}</li>" for b in bullets) + "</ul>"
    css_class = "pd-card pd-card-equal" if equal_height else "pd-card"
    st.markdown(
        f'<div class="{css_class}"><h3>{escape(title)}</h3>{body}</div>',
        unsafe_allow_html=True,
    )


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="pd-eyebrow">{escape(text)}</div>', unsafe_allow_html=True)


def result_name(text: str) -> None:
    st.markdown(f'<div class="pd-result-name">{escape(text)}</div>', unsafe_allow_html=True)


def category_badge(category: str) -> None:
    label, icon, color = CATEGORY_BADGES.get(category, CATEGORY_BADGES["unknown"])
    st.badge(label, icon=icon, color=color)


def heat_legend() -> None:
    st.markdown(
        '<div class="pd-legend">'
        f'<div class="pd-legend-bar" style="background: {gradcam.legend_gradient()};"></div>'
        '<div class="pd-legend-labels"><span>Low influence</span><span>High influence</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def format_accuracy(classifier: PlantDiseaseClassifier) -> str:
    evaluation = classifier.info.evaluation
    return f"{evaluation.accuracy:.2%}" if evaluation else "Not measured"


def sidebar_status(classifier: PlantDiseaseClassifier) -> None:
    with st.sidebar:
        st.divider()
        st.markdown("**Model**")
        st.markdown(
            f":green[:material/check_circle:] Ready  \n"
            f"{escape(classifier.info.backbone)} · {classifier.num_classes} classes · "
            f"{classifier.image_size} px input"
        )
        st.caption("Academic deep learning project. Not a substitute for expert diagnosis.")


def footer() -> None:
    st.divider()
    st.markdown(
        '<div class="pd-footer">PlantDisease AI · Academic deep learning project · '
        "MobileNetV2 transfer learning</div>",
        unsafe_allow_html=True,
    )
