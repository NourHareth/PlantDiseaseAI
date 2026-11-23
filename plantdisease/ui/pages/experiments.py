"""Results of the experiment suite (reads reports/results/, written by the scripts)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from plantdisease import config
from plantdisease.ui import charts, components


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> dict:  # mtime busts the cache when the file changes
    return json.loads(open(path, encoding="utf-8").read())


def _json(path) -> dict | None:
    return _load(str(path), path.stat().st_mtime) if path.exists() else None


def render() -> None:
    components.hero(
        "Experiments 🧪",
        "What was tested, how each model performed on lab and field photos, and how trustworthy "
        "its confidence is. The app itself keeps using the original model.",
    )
    summary = _json(config.RESULTS_DIR / "summary.json")
    if not summary or not summary.get("runs"):
        st.info(
            "No experiment results yet. Run `python -m scripts.run_experiments` and then "
            "`python -m scripts.make_report`.",
            icon=":material/science:",
        )
        return

    runs = summary["runs"]
    best_name = summary["best_run"]
    best = next(r for r in runs if r["name"] == best_name)
    best_result = _json(config.RESULTS_DIR / f"{best_name}.json") or {}
    baseline = next((r for r in runs if r["group"] == "baseline"), None)

    highlights = config.REPORTS_DIR / "HIGHLIGHTS.md"
    if highlights.exists():
        st.markdown(highlights.read_text(encoding="utf-8"))

    components.eyebrow(f"Highest field-photo accuracy: {best_name}")
    st.caption("Picked on the field test photos, so it describes the results rather than selecting a model; "
               "differences of a few points between runs are within noise (see the paired tests under *All runs*).")
    col1, col2, col3, col4 = st.columns(4)
    def points(a: float, b: float) -> str:
        return f"{(a - b) * 100:+.1f} pts vs original"

    col1.metric("Lab-photo accuracy", f"{best['pv_accuracy']:.1%}",
                delta=points(best["pv_accuracy"], baseline["pv_accuracy"]) if baseline else None)
    if best.get("pd_accuracy") is not None:
        col2.metric("Field-photo accuracy", f"{best['pd_accuracy']:.1%}",
                    delta=points(best["pd_accuracy"], baseline["pd_accuracy"])
                    if baseline and baseline.get("pd_accuracy") is not None else None)
    ece_change = best["pv_ece_calibrated"] - best["pv_ece"]
    col3.metric("Calibration error (ECE)", f"{best['pv_ece_calibrated']:.3f}",
                delta=f"{ece_change:+.3f} after scaling" if abs(ece_change) >= 0.001 else "already calibrated",
                delta_color="inverse" if abs(ece_change) >= 0.001 else "off",
                help="Expected calibration error on lab photos after temperature scaling (lower is better).")
    if best.get("ood_background_only_mahalanobis") is not None:
        col4.metric("Non-leaf AUROC", f"{best['ood_background_only_mahalanobis']:.2f}",
                    help="How well the Mahalanobis score separates real photos from photos with the leaf "
                         "removed (1.0 = perfect, 0.5 = guessing).")

    gap_tab, table_tab, calib_tab, ood_tab, figures_tab = st.tabs([
        ":material/compare_arrows: Lab vs field",
        ":material/table: All runs",
        ":material/tune: Calibration",
        ":material/block: Non-leaf detection",
        ":material/image: Figures",
    ])
    with gap_tab:
        st.markdown(
            "Accuracy on held-out **lab photos** (PlantVillage) next to accuracy on real **field photos** "
            "(PlantDoc). The distance between the two dots is the *domain gap*: what is lost when the "
            "model leaves the lab."
        )
        with_field = sum(1 for r in runs if r.get("pd_accuracy") is not None)
        st.altair_chart(charts.domain_gap_chart(runs), width="stretch", height=charts.domain_gap_height(with_field))
    with table_tab:
        table = pd.DataFrame([
            {
                "Run": r["name"], "Question": r["group"], "Backbone": r["backbone"], "Strategy": r["strategy"],
                "Size": r["image_size"], "Augmentation": r["augmentation"], "Background repl.": r["background_prob"],
                "Lab acc.": r["pv_accuracy"], "Field acc.": r.get("pd_accuracy"), "ECE": r["pv_ece"],
                "ECE (scaled)": r["pv_ece_calibrated"], "Grad-CAM lift": r.get("gradcam_lift"),
                "Params (M)": r["params"] / 1e6, "Latency (ms)": r["latency_ms"], "Train (min)": r.get("train_minutes"),
            }
            for r in runs
        ])
        percent = st.column_config.NumberColumn(format="percent")
        st.dataframe(table, hide_index=True, column_config={
            "Lab acc.": percent, "Field acc.": percent, "Background repl.": percent,
            "ECE": st.column_config.NumberColumn(format="%.3f"),
            "ECE (scaled)": st.column_config.NumberColumn(format="%.3f"),
            "Grad-CAM lift": st.column_config.NumberColumn(format="%.2f"),
            "Params (M)": st.column_config.NumberColumn(format="%.1f"),
            "Latency (ms)": st.column_config.NumberColumn(format="%.0f"),
        })
        if summary.get("seed_stats"):
            st.markdown("**Seed variation** (mean ± standard deviation)")
            st.dataframe(pd.DataFrame([
                {"Configuration": s["family"], "Runs": s["runs"],
                 "Lab acc.": f"{s['pv_accuracy']['mean']:.1%} ± {s['pv_accuracy']['std']:.1%}",
                 "Field acc.": f"{s['pd_accuracy']['mean']:.1%} ± {s['pd_accuracy']['std']:.1%}" if "pd_accuracy" in s else "–"}
                for s in summary["seed_stats"]
            ]), hide_index=True)
        if summary.get("comparisons"):
            st.markdown(
                "**Paired tests** (exact McNemar, Holm-corrected). Both models are scored on the same images; "
                "only images that exactly one of them gets right count. p (Holm) below 0.05: unlikely to be "
                "an accident of the test images."
            )
            st.dataframe(pd.DataFrame([
                {"Comparison": c["question"], "A": c["a"], "B": c["b"], "Images": c["images"],
                 "Acc. A": c["accuracy_a"], "Acc. B": c["accuracy_b"], "Change (points)": c["difference"] * 100,
                 "A only / B only": f"{c['only_a']} / {c['only_b']}", "p (Holm)": c["p_holm"]}
                for c in summary["comparisons"]
            ]), hide_index=True, column_config={
                "Acc. A": percent, "Acc. B": percent,
                "Change (points)": st.column_config.NumberColumn(format="%+.1f"),
                "p (Holm)": st.column_config.NumberColumn(format="%.3f"),
            })
    with calib_tab:
        if best_result.get("reliability"):
            st.markdown(
                "A calibrated model is right 80% of the time when it says 80%. Points below the dashed "
                "line mean over-confidence. **Temperature scaling** (one parameter fitted on the validation "
                "set) moves them onto the line without changing any prediction."
            )
            st.altair_chart(charts.reliability_chart(best_result["reliability"], best_result["temperature"]), width="stretch")
    with ood_tab:
        if best_result.get("ood"):
            st.markdown(
                "How well each score separates real test photos from images with **no leaf** "
                "(0.5 = guessing, 1.0 = perfect): the softmax confidence, the energy of the logits, "
                "and the Mahalanobis distance in the network's feature space."
            )
            st.altair_chart(charts.ood_chart(best_result["ood"]), width="content")
    with figures_tab:
        for name, caption in (
            ("background_replacement.png", "Background replacement used in training (masks can cut out dark lesions)"),
            ("confusion_plantvillage_best.png", "Confusion matrix on lab photos"),
            ("confusion_plantdoc_best.png", "Confusion matrix on field photos"),
            ("gradcam_field_tomato_leaf.png", "Grad-CAM of the original model on a field photo"),
        ):
            path = config.FIGURES_DIR / name
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")

    st.caption("Full write-up: `reports/REPORT.md` · tables: `reports/RESULTS.md` · raw numbers: `reports/results/`.")
