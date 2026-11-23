"""Charts. Single series, so one colour for every bar and no legend; values are
direct-labelled on bars of at least 1% and every bar has a tooltip."""

from __future__ import annotations

import altair as alt
import pandas as pd

from plantdisease.inference import Prediction
from plantdisease.knowledge import KnowledgeBase

BAR_COLOR = "#3aa865"  # validated for contrast and lightness on the dark surface
LABEL_COLOR = "#b9c9bf"
AXIS_COLOR = "#9aaba0"
GRID_COLOR = "rgba(154, 183, 163, 0.12)"
# Categorical series, dark-mode steps (validated all-pairs on the app surface #121b16)
SERIES = ["#3987e5", "#d95926", "#199e70"]


def format_probability(p: float) -> str:
    """One-decimal percentage that never rounds to a false certainty (99.96% is not "100.0%")."""
    if p >= 0.9995:
        return "> 99.9%"
    if 0 < p < 0.0005:
        return "< 0.1%"
    return f"{p:.1%}"


def _configure(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke=None)
        .configure_axis(gridColor=GRID_COLOR, domain=False, ticks=False, labelColor=AXIS_COLOR,
                        titleColor=AXIS_COLOR, labelFontSize=12)
        .configure_legend(labelColor=LABEL_COLOR, titleColor=AXIS_COLOR, orient="top", labelFontSize=12)
    )


def domain_gap_chart(runs: list[dict]) -> alt.Chart:
    """Dumbbell: lab-photo vs field-photo accuracy per run (two series, legend + direct labels)."""
    rows = [r for r in runs if r.get("pd_accuracy") is not None]
    order = [r["name"] for r in sorted(rows, key=lambda r: -r["pd_accuracy"])]
    # Short series names keep the legend inside the plot width on phones; the tooltip names the dataset.
    lab, field = "Lab photos", "Field photos"
    points = pd.DataFrame(
        [{"Run": r["name"], "Test set": lab, "Dataset": "PlantVillage test", "Accuracy": r["pv_accuracy"]} for r in rows]
        + [{"Run": r["name"], "Test set": field, "Dataset": "PlantDoc", "Accuracy": r["pd_accuracy"]} for r in rows]
    )
    points["Label"] = points["Accuracy"].map(lambda v: f"{v:.0%}")
    y = alt.Y("Run:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelPadding=8))
    colour = alt.Color("Test set:N", scale=alt.Scale(domain=[lab, field], range=SERIES[:2]), title=None)
    rules = alt.Chart(points).mark_line(color="#2a3a31", strokeWidth=3).encode(
        x="Accuracy:Q", y=y, detail="Run:N")
    dots = alt.Chart(points).mark_circle(size=110, opacity=1, stroke="#121b16", strokeWidth=2).encode(
        x=alt.X("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5), title=None),
        y=y, color=colour,
        tooltip=["Run:N", "Dataset:N", alt.Tooltip("Accuracy:Q", format=".1%")],
    )
    labels = alt.Chart(points[points["Test set"] == field]).mark_text(
        align="right", dx=-10, color=LABEL_COLOR, fontWeight=600).encode(x="Accuracy:Q", y=y, text="Label:N")
    # Extra left padding: Vega measures the run names before the web font loads, and the wider
    # rendered text would otherwise be clipped at the chart's left edge.
    return _configure((rules + dots + labels).properties(
        height=domain_gap_height(len(order)), padding={"left": 28, "right": 8, "top": 4, "bottom": 4}))


def domain_gap_height(n_runs: int) -> int:
    """Total pixel height incl. legend and axis (Streamlit sizes the whole chart to this)."""
    return 80 + 38 * n_runs


def reliability_chart(reliability: dict, temperature: float) -> alt.Chart:
    """Confidence vs accuracy per bin, before and after temperature scaling."""
    rows = []
    for key, name in (("raw", "Before calibration"), ("calibrated", f"After (T = {temperature:.2f})")):
        r = reliability[key]
        for conf, acc, count in zip(r["confidence"], r["accuracy"], r["count"]):
            if count >= 10:  # tiny bins are noise
                rows.append({"Series": name, "Confidence": conf, "Accuracy": acc, "Images": count})
    data = pd.DataFrame(rows)
    names = list(dict.fromkeys(data["Series"]))
    diagonal = alt.Chart(pd.DataFrame({"Confidence": [0, 1], "Accuracy": [0, 1]})).mark_line(
        color=AXIS_COLOR, strokeDash=[4, 4], strokeWidth=1).encode(x="Confidence:Q", y="Accuracy:Q")
    lines = alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2).encode(
        x=alt.X("Confidence:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Series:N", scale=alt.Scale(domain=names, range=[SERIES[1], SERIES[0]]), title=None),
        tooltip=["Series:N", alt.Tooltip("Confidence:Q", format=".1%"), alt.Tooltip("Accuracy:Q", format=".1%"), "Images:Q"],
    )
    return _configure((diagonal + lines).properties(height=340))


def ood_chart(ood: dict) -> alt.Chart:
    """AUROC of three OOD scores on two kinds of non-leaf images (grouped bars)."""
    methods = {"msp": "Max softmax", "energy": "Energy", "mahalanobis": "Mahalanobis"}
    sets = {"synthetic": "Synthetic (noise, flat colour)", "background_only": "Background only (leaf removed)"}
    rows = [
        {"Images": sets[s], "Method": methods[m], "AUROC": ood[s][m]["auroc"]}
        for s in sets if s in ood for m in methods
    ]
    data = pd.DataFrame(rows)
    data["Label"] = data["AUROC"].map(lambda v: f"{v:.2f}")
    base = alt.Chart(data).encode(
        x=alt.X("Method:N", title=None, sort=list(methods.values()), axis=alt.Axis(labels=False)),
        y=alt.Y("AUROC:Q", scale=alt.Scale(domain=[0, 1.05]), title="AUROC"),
        color=alt.Color("Method:N", scale=alt.Scale(domain=list(methods.values()), range=SERIES), title=None),
        tooltip=["Images:N", "Method:N", alt.Tooltip("AUROC:Q", format=".3f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4)
    # alt.value overrides the colour encoding inherited from `base`: labels wear text ink, not series colours
    text = base.mark_text(dy=-8, fontWeight=600).encode(text="Label:N", color=alt.value(LABEL_COLOR))
    return _configure(
        (bars + text).properties(width=180, height=260).facet(
            column=alt.Column("Images:N", title=None, header=alt.Header(labelColor=LABEL_COLOR, labelFontSize=12))
        )
    )


def probability_table(prediction: Prediction, knowledge: KnowledgeBase) -> pd.DataFrame:
    rows = [
        {"Class": knowledge.display_name(label), "Probability": probability}
        for _, label, probability in prediction.ranked()
    ]
    return pd.DataFrame(rows)


def probability_chart(table: pd.DataFrame) -> alt.Chart:
    data = table.sort_values("Probability", ascending=False)
    data = data.assign(Label=data["Probability"].map(format_probability))
    # An explicit order: sort="-x" is silently dropped when the bar and label layers are combined.
    order = data["Class"].tolist()
    base = alt.Chart(data).encode(
        y=alt.Y("Class:N", sort=order, title=None, axis=alt.Axis(labelLimit=240, labelPadding=8)),
        x=alt.X(
            "Probability:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", tickCount=5, title=None),
        ),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Probability:Q", format=".2%")],
    )
    bars = base.mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, height={"band": 0.72})
    labels = (
        base.transform_filter(alt.datum.Probability >= 0.01)
        .mark_text(align="left", baseline="middle", dx=6, color=LABEL_COLOR, fontWeight=600)
        .encode(text="Label:N")
    )
    return (
        (bars + labels)
        .properties(height=36 * len(data))
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=GRID_COLOR,
            domain=False,
            ticks=False,
            labelColor=AXIS_COLOR,
            labelFontSize=13,
        )
    )
