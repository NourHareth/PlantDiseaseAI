import numpy as np

from plantdisease.inference import Prediction
from plantdisease.ui.charts import format_probability, probability_chart, probability_table


def test_probabilities_never_round_to_false_certainty():
    assert format_probability(0.99961) == "> 99.9%"
    assert format_probability(0.9994) == "99.9%"
    assert format_probability(0.0002) == "< 0.1%"
    assert format_probability(0.0) == "0.0%"
    assert format_probability(0.4567) == "45.7%"


def test_experiment_charts_keep_labels_readable():
    from plantdisease.ui.charts import LABEL_COLOR, domain_gap_chart, ood_chart

    point = {"auroc": 0.9, "fpr_at_95_tpr": 0.1}
    ood = {s: {m: point for m in ("msp", "energy", "mahalanobis")} for s in ("synthetic", "background_only")}
    text_layer = ood_chart(ood).to_dict()["spec"]["layer"][1]
    assert text_layer["encoding"]["color"] == {"value": LABEL_COLOR}  # value labels in text ink, not series colours

    runs = [{"name": "a_rather_long_run_name", "pv_accuracy": 0.95, "pd_accuracy": 0.22}]
    assert domain_gap_chart(runs).to_dict()["padding"]["left"] >= 20  # room for run names (web-font width)


def test_chart_lists_classes_from_most_to_least_likely(knowledge):
    labels = ("Bacterial spot", "Early blight", "healthy", "Late blight")
    prediction = Prediction(np.array([0.1, 0.05, 0.25, 0.6]), labels)
    table = probability_table(prediction, knowledge)
    assert table["Class"].tolist() == ["Late blight", "Healthy", "Bacterial spot", "Early blight"]

    spec = probability_chart(table).to_dict()
    y_sort = spec["layer"][0]["encoding"]["y"]["sort"]
    assert y_sort == ["Late blight", "Healthy", "Bacterial spot", "Early blight"]
