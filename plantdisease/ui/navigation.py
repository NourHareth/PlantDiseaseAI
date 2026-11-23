"""Page registry, so pages can link to each other without circular imports."""

from __future__ import annotations

import streamlit as st


def get_pages() -> dict[str, st.Page]:
    from plantdisease.ui.pages import diagnose, experiments, home, model_info

    return {
        "home": st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
        "diagnose": st.Page(diagnose.render, title="Diagnose", icon=":material/biotech:", url_path="diagnose"),
        "model": st.Page(model_info.render, title="Model info", icon=":material/neurology:", url_path="model"),
        "experiments": st.Page(experiments.render, title="Experiments", icon=":material/science:", url_path="experiments"),
    }
