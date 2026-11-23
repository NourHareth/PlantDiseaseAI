"""Aggregate experiment results into tables and figures.

    python -m scripts.make_report

Reads ``reports/results/*.json`` and writes:

* ``reports/results/summary.json``: one row per run (used by the app);
* ``reports/RESULTS.md``: the results tables in Markdown;
* ``reports/figures/*.png``: domain gap, calibration, OOD and attention charts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts import _bootstrap  # noqa: F401  (python -m scripts.make_report)
except ImportError:
    import _bootstrap  # noqa: F401  (python scripts/make_report.py)

import numpy as np

from plantdisease import config
from plantdisease.knowledge import KnowledgeBase

# Validated categorical palette (slots 1-3) and neutral ink for light-background figures
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]
MIN_BIN = 10  # reliability bins with fewer images are not drawn

SEED_FAMILIES = {
    "MobileNetV2 fine-tuned": ["mnv2_finetune", "mnv2_finetune_s1", "mnv2_finetune_s2"],
    "MobileNetV2 fine-tuned + background replacement": ["mnv2_finetune_bg", "mnv2_finetune_bg_s1", "mnv2_finetune_bg_s2"],
}


NOT_RUNS = {"summary.json", "comparisons.json", "detector_check.json"}  # aggregates next to the per-run results
BEST_RULE = ("highest PlantDoc (field-photo) accuracy among the experiment runs. This is chosen on test data, "
             "so it is a description, not a model selection, and margins of a few points are within noise.")


def load_results(results_dir: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name in NOT_RUNS or path.name.startswith("tflite_"):
            continue
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def best_ood(result: dict, ood_set: str) -> tuple[str, float]:
    scores = {m: v["auroc"] for m, v in result["ood"][ood_set].items() if isinstance(v, dict)}
    method = max(scores, key=scores.get)
    return method, scores[method]


def summary_row(name: str, r: dict) -> dict:
    run = r["run"]
    # Field accuracy: all PlantDoc images for models never trained on it (n=741), else none.
    pv, pd, pd_test = r["plantvillage_test"], r.get("plantdoc_all"), r.get("plantdoc_test")
    row = {
        "name": name,
        "group": run.get("group", ""),
        "backbone": run.get("backbone", ""),
        "strategy": run.get("strategy", ""),
        "image_size": run.get("image_size", 128),
        "augmentation": run.get("augmentation", "standard"),
        "background_prob": run.get("background_prob", 0.0),
        "extra_train": ",".join(run.get("extra_train_sources", [])),
        "seed": run.get("seed", 42),
        "pv_accuracy": pv["accuracy"],
        "pv_ci95": pv["accuracy_ci95"],
        "pv_macro_f1": pv["macro_f1"],
        "pv_ece": pv["ece"],
        "pv_ece_calibrated": pv["ece_calibrated"],
        "temperature": r.get("temperature"),
        "params": r["speed"]["params"],
        "latency_ms": r["speed"]["latency_ms_median"],
        "model_mb": r["speed"]["model_mb"],
        "train_minutes": round(run["train_seconds"] / 60, 1) if "train_seconds" in run else None,
        "gradcam_lift": r.get("gradcam", {}).get("lift"),
        "gradcam_on_leaf": r.get("gradcam", {}).get("attention_on_leaf"),
        "note": run.get("note", ""),
    }
    if pd:
        row.update({
            "pd_accuracy": pd["accuracy"],
            "pd_ci95": pd["accuracy_ci95"],
            "pd_macro_f1": pd["macro_f1"],
            "pd_ece": pd["ece"],
            "pd_ece_calibrated": pd["ece_calibrated"],
            "domain_gap": round(pv["accuracy"] - pd["accuracy"], 4),
        })
    if pd_test:
        row.update({"pd_test_accuracy": pd_test["accuracy"], "pd_test_ci95": pd_test["accuracy_ci95"]})
    for ood_set in ("synthetic", "background_only"):
        if ood_set in r.get("ood", {}):
            for method in ("msp", "energy", "mahalanobis"):
                row[f"ood_{ood_set}_{method}"] = r["ood"][ood_set][method]["auroc"]
    return row


def seed_stats(rows: dict[str, dict]) -> list[dict]:
    out = []
    for family, names in SEED_FAMILIES.items():
        present = [rows[n] for n in names if n in rows]
        if len(present) < 2:
            continue
        entry = {"family": family, "runs": len(present)}
        for key in ("pv_accuracy", "pd_accuracy", "pv_ece", "gradcam_lift"):
            values = [p[key] for p in present if p.get(key) is not None]
            if values:
                entry[key] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values, ddof=1)), 4)}
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_domain_gap(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in rows if "pd_accuracy" in r]
    rows.sort(key=lambda r: r["pd_accuracy"])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.4), dpi=150)
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.plot([r["pd_accuracy"], r["pv_accuracy"]], [i, i], color=GRID, linewidth=3, zorder=1)
    ax.scatter([r["pv_accuracy"] for r in rows], y, s=60, color=SERIES[0], zorder=2,
               edgecolor="white", linewidth=2, label="PlantVillage test (lab photos)")
    ax.scatter([r["pd_accuracy"] for r in rows], y, s=60, color=SERIES[1], zorder=2,
               edgecolor="white", linewidth=2, label="PlantDoc (field photos)")
    for i, r in enumerate(rows):
        outside = r["pd_accuracy"] <= r["pv_accuracy"]  # label on the side away from the other dot
        ax.annotate(f"{r['pd_accuracy']:.0%}", (r["pd_accuracy"], i), xytext=(-8 if outside else 8, 0),
                    textcoords="offset points", ha="right" if outside else "left", va="center", fontsize=8, color=INK)
    ax.axvline(0.1, color=MUTED, linestyle="--", linewidth=1, zorder=0)
    ax.annotate("chance (10 classes)", (0.1, len(rows) - 0.5), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=MUTED, va="center")
    ax.set_yticks(y, [r["name"] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Accuracy on lab photos vs. field photos", loc="left", fontsize=12, color=INK)
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    below = 0.4 / (fig.get_figheight() * ax.get_position().height)  # ~0.4 inch under the axis, any figure height
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -below), ncols=2, fontsize=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_reliability(result: dict, name: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5), dpi=150)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
    for key, colour, label in (("raw", SERIES[1], "Before"), ("calibrated", SERIES[0], f"After (T = {result['temperature']:.2f})")):
        rel = result["reliability"][key]
        mask = np.array(rel["count"]) >= MIN_BIN  # tiny bins are noise
        conf, acc = np.array(rel["confidence"])[mask], np.array(rel["accuracy"])[mask]
        ece = result["plantvillage_test"]["ece" if key == "raw" else "ece_calibrated"]
        ax.plot(conf, acc, color=colour, linewidth=2, marker="o", markersize=5, label=f"{label}, ECE {ece:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", color=INK)
    ax.set_ylabel("Accuracy", color=INK)
    ax.set_title(f"Reliability diagram · {name}", loc="left", fontsize=11, color=INK)
    ax.annotate(f"bins with ≥ {MIN_BIN} images", (0.98, 0.02), ha="right", fontsize=8, color=MUTED)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_confidence_gap(results: dict[str, dict], names: list[str], path: Path) -> None:
    """Mean confidence vs. accuracy on lab and field photos: over-confidence off the lab bench."""
    import matplotlib.pyplot as plt

    names = [n for n in names if n in results and "plantdoc_all" in results[n]]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), dpi=150, sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        r = results[name]
        sets = [("Lab", r["plantvillage_test"]), ("Field", r["plantdoc_all"])]
        x = np.arange(len(sets))
        for k, (key, label) in enumerate((("accuracy", "Accuracy"), ("mean_confidence", "Mean confidence"))):
            values = [s[key] for _, s in sets]
            bars = ax.bar(x + (k - 0.5) * 0.36, values, 0.33, color=SERIES[k], label=label)
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.0%}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, [s for s, _ in sets], color=INK)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title(name, loc="left", fontsize=10, color=INK)
        _style(ax)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Mean confidence vs. accuracy on lab and field photos", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_ood(result: dict, name: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    sets = [s for s in ("synthetic", "background_only") if s in result["ood"]]
    methods = [("msp", "Max softmax"), ("energy", "Energy"), ("mahalanobis", "Mahalanobis")]
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    width = 0.26
    x = np.arange(len(sets))
    for k, (method, label) in enumerate(methods):
        values = [result["ood"][s][method]["auroc"] for s in sets]
        bars = ax.bar(x + (k - 1) * width, values, width - 0.03, color=SERIES[k], label=label)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.annotate("chance", (x[-1] + 0.45, 0.5), xytext=(0, 3), textcoords="offset points",
                va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(x, ["Synthetic (noise, flat colour)", "Background only (leaf removed)"], color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUROC", color=INK)
    ax.set_title(f"Detecting non-leaf images · {name}", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, ncols=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_confusion(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    matrix = np.array(matrix)
    keep = matrix.sum(axis=1) > 0  # PlantDoc has no Target spot images
    shown_rows = [f"{l} ({n:,})" for l, n, k in zip(labels, matrix.sum(axis=1), keep) if k]
    totals = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)[keep]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(shown_rows) + 2.5), dpi=150)
    image = ax.imshow(norm, cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(shown_rows)), shown_rows, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontsize=11)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, format=lambda v, _: f"{v:.0%}")
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def pct(value, ci=None) -> str:
    if value is None:
        return "–"
    text = f"{value:.1%}"
    return f"{text} ({ci[0]:.1%}–{ci[1]:.1%})" if ci else text


def p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


SPLIT_NAMES = {"plantvillage_test": "PlantVillage test", "plantdoc_all": "PlantDoc all", "plantdoc_test": "PlantDoc test"}


def comparison_lines(comparisons: list[dict]) -> list[str]:
    lines = ["", "## Paired comparisons (exact McNemar test, Holm-corrected)", "",
             "Both models are scored on the same images. *A only / B only* counts the images that exactly one "
             "of them gets right; only these carry information. A Holm-corrected p below 0.05 means the "
             "difference is unlikely to come from the particular test images drawn. Variation between "
             "training seeds is a separate question (see the seed table).", "",
             "| Comparison | A | B | Test set | Images | Accuracy A → B | A only / B only | p | p (Holm) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in comparisons:
        lines.append(
            f"| {c['question']} | `{c['a']}` | `{c['b']}` | {SPLIT_NAMES.get(c['split'], c['split'])} | {c['images']} | "
            f"{c['accuracy_a']:.1%} → {c['accuracy_b']:.1%} ({c['difference'] * 100:+.1f} pts) | {c['only_a']} / {c['only_b']} | "
            f"{p_text(c['p_value'])} | {'**' + p_text(c['p_holm']) + '**' if c['p_holm'] < 0.05 else p_text(c['p_holm'])} |"
        )
    return lines


def markdown(rows: list[dict], seeds: list[dict], tflite: list[dict], comparisons: list[dict] | None = None,
             best: str | None = None) -> str:
    lines = [
        "# Experiment results",
        "",
        "Generated by `python -m scripts.make_report` from `reports/results/*.json`.",
        "Accuracy intervals are 95% bootstrap confidence intervals. PlantVillage = lab photos (leakage-aware "
        "test split); PlantDoc = real-world field photos (never used for training except in `*_mixed`).",
        "Latency is one image through the app's Keras model on this CPU.",
        "",
        "## Main results",
        "",
        "| Run | Backbone | Strategy | Size | Aug. | BG repl. | PlantVillage acc. | PlantDoc acc. | Gap | ECE (raw → T) | Params | Latency |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['backbone']} | {r['strategy']} | {r['image_size']} | {r['augmentation']} | "
            f"{r['background_prob']:.0%} | {pct(r['pv_accuracy'], r['pv_ci95'])} | "
            f"{pct(r.get('pd_accuracy'), r.get('pd_ci95'))} | "
            f"{'–' if r.get('domain_gap') is None else format(r['domain_gap'] * 100, '.1f') + ' pts'} | "
            f"{r['pv_ece']:.3f} → {r['pv_ece_calibrated']:.3f} | {r['params'] / 1e6:.1f} M | {r['latency_ms']:.0f} ms |"
        )
    if seeds:
        lines += ["", "## Seed variation (mean ± std over seeds)", "",
                  "| Configuration | Runs | PlantVillage acc. | PlantDoc acc. | Grad-CAM lift |", "|---|---|---|---|---|"]
        for s in seeds:
            def ms(key, fmt="{:.1%}"):
                if key not in s:
                    return "–"
                return f"{fmt.format(s[key]['mean'])} ± {fmt.format(s[key]['std'])}"
            lines.append(f"| {s['family']} | {s['runs']} | {ms('pv_accuracy')} | {ms('pd_accuracy')} | {ms('gradcam_lift', '{:.2f}')} |")
    if comparisons:
        lines += comparison_lines(comparisons)
    lines += ["", "## Out-of-distribution detection (AUROC, higher is better)", "",
              "| Run | Synthetic: MSP | Energy | Mahalanobis | Background-only: MSP | Energy | Mahalanobis |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{r.get(f'ood_{s}_{m}', float('nan')):.3f}" for s in ("synthetic", "background_only")
                 for m in ("msp", "energy", "mahalanobis")]
        lines.append(f"| `{r['name']}` | " + " | ".join(cells) + " |")
    lines += ["", "## Grad-CAM: does the model look at the leaf?", "",
              "Share of Grad-CAM mass inside the PlantVillage leaf mask, divided by the leaf's share of the "
              "image (lift > 1 means the model attends to the leaf more than chance).", "",
              "| Run | Attention on leaf | Lift |", "|---|---|---|"]
    for r in rows:
        if r.get("gradcam_lift") is not None:
            lines.append(f"| `{r['name']}` | {r['gradcam_on_leaf']:.1%} | {r['gradcam_lift']:.2f} |")
    if tflite:
        lines += ["", "## On-device export (TensorFlow Lite)", "",
                  "| Model | Variant | Accuracy | Agreement with Keras | Size | Latency |", "|---|---|---|---|---|---|"]
        for t in tflite:
            lines.append(f"| `{t['model']}` | Keras | {t['keras']['accuracy']:.1%} | – | {t['keras']['model_mb']:.1f} MB | – |")
            for variant in ("float32", "dynamic", "int8"):
                v = t[variant]
                lines.append(f"| `{t['model']}` | {variant} | {v['accuracy']:.1%} | {v['agreement_with_keras']:.1%} | "
                             f"{v['model_mb']:.1f} MB | {v['latency_ms_median']:.1f} ms |")
    notes = [f"- `{r['name']}`: {r['note']}" for r in rows if r.get("note")]
    if best:
        notes.append(f"- Detailed figures use `{best}`: the {BEST_RULE}")
    if notes:
        lines += ["", "## Notes", ""] + notes
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR, help="Where RESULTS.md and figures/ go.")
    parser.add_argument("--best", default=None, help="Run used for the detailed figures (default: best PlantDoc accuracy).")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")

    results = load_results(args.results_dir)
    if not results:
        print(f"No results in {args.results_dir}")
        return 1
    rows = [summary_row(name, r) for name, r in results.items()]
    by_name = {r["name"]: r for r in rows}
    seeds = seed_stats(by_name)
    tflite = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.results_dir.glob("tflite_*.json"))]
    comparisons_path = args.results_dir / "comparisons.json"
    comparisons = json.loads(comparisons_path.read_text(encoding="utf-8"))["comparisons"] if comparisons_path.exists() else []

    candidates = [r for r in rows if r["group"] != "baseline"] or rows
    best = args.best or max(candidates, key=lambda r: (r.get("pd_accuracy", 0), r["pv_accuracy"]))["name"]
    summary = {"runs": rows, "seed_stats": seeds, "best_run": best,
               "best_rule": "chosen with --best" if args.best else BEST_RULE,
               "comparisons": comparisons, "tflite": tflite}
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "RESULTS.md").write_text(
        markdown(rows, seeds, tflite, comparisons, best=None if args.best else best), encoding="utf-8")

    figures = args.reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    knowledge = KnowledgeBase.from_json()
    best_result = results[best]
    labels = [knowledge.display_name(c) for c in best_result.get("class_names", [])] or None
    figure_domain_gap(rows, figures / "domain_gap.png")
    figure_reliability(best_result, best, figures / "reliability_best.png")
    baseline_name = next((r["name"] for r in rows if r["group"] == "baseline"), None)
    figure_confidence_gap(results, [n for n in (baseline_name, best) if n], figures / "confidence_gap.png")
    figure_ood(best_result, best, figures / "ood_best.png")
    if labels is None:
        labels = [f"class {i}" for i in range(len(best_result["plantvillage_test"]["confusion_matrix"]))]
    figure_confusion(best_result["plantvillage_test"]["confusion_matrix"], labels,
                     f"PlantVillage test · {best}", figures / "confusion_plantvillage_best.png")
    field = best_result.get("plantdoc_all") or best_result.get("plantdoc_test")
    if field:
        figure_confusion(field["confusion_matrix"], labels,
                         f"PlantDoc field photos · {best}", figures / "confusion_plantdoc_best.png")
    print(f"{len(rows)} runs summarised; best run: {best}")
    print(f"Wrote {args.reports_dir / 'RESULTS.md'} and figures in {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
