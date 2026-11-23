"""PlantDisease AI: Streamlit entry point.

Run with:  streamlit run app.py

This file only wires the app together. The model code lives in
``plantdisease/``, page layouts in ``plantdisease/ui/pages/``, disease content
in ``data/disease_info.json`` and styling in ``assets/styles.css``.
"""

import streamlit as st

from plantdisease import config
from plantdisease.ui import components, navigation, resources

st.set_page_config(
    page_title="PlantDisease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, collapsed on phones (where it would cover the page)
)
components.inject_css()
st.logo(str(config.LOGO_PATH), size="large")

try:
    classifier = resources.get_classifier()
    knowledge = resources.get_knowledge()
except Exception as exc:  # show a readable message instead of a stack trace
    st.error(
        "The model could not be loaded. Check that `models/` contains "
        "`plant_disease_model.keras` and `class_names.json` from the same training run, "
        "and that the packages in `requirements.txt` are installed.",
        icon=":material/error:",
    )
    st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
    st.stop()

undocumented = knowledge.missing(list(classifier.class_names))
if undocumented:
    st.sidebar.warning(f"No disease guide for: {', '.join(undocumented)}")

page = st.navigation(list(navigation.get_pages().values()), position="sidebar")
components.sidebar_status(classifier)
page.run()
components.footer()
