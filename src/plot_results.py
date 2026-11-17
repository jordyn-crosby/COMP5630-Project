# Title: plot_results.py
# Description: Builds the two paper figures and the combined results table from the saved experiment outputs,
# so every figure and table in the paper comes from the same run of main.py and h3_feature_reduction.py.
#
#   results/model_performance.csv   one row per experiment x model (metrics + confusion counts)
#   results/model_performance.png   accuracy, macro F1 and ROC AUC per model, grouped by experiment
#   results/confusion_matrix.png    predicted vs actual for the best model (macro F1) in each experiment
#
# Experiments drawn, in order: A_pooled (protocol), B_temporal (protocol), H3_pooled/reduced (prespecified
# parsimony check), C_accuracy_tuned (sensitivity: unweighted classes, tuned on accuracy; not a headline),
# D_extended (sensitivity: A's split with five extra household/geography predictors; not a headline).
# Experiments whose folder is missing are skipped.
#
# Run:  python plot_results.py                (reads outputs/, writes results/)
#       python plot_results.py --outdir other_outputs --results other_results

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402

OUTPUT_DIR = "outputs"
RESULTS_DIR = "results"

# experiment folder -> label used in the paper
EXPERIMENTS = [
    ("A_pooled", "Pooled 70/30 (household-grouped)"),
    ("B_temporal", "Train 2019+2021, test 2023"),
    (os.path.join("H3_pooled", "reduced"), "Reduced (age, income, education, disability)"),
    ("C_accuracy_tuned", "Accuracy-tuned, unweighted (sensitivity)"),
    ("D_extended", "Extended, +5 predictors (sensitivity)"),
]
MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
BASELINE_IN = "Dummy (majority)"
BASELINE_OUT = "Baseline (always connected)"

# validated categorical palette (slots 1-4, light surface; slot 5 a distinct purple) + neutral for the baseline
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5cc4"]
NEUTRAL = "#9a9994"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"
# sequential blue ramp (light -> dark) for the confusion cells
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.titlecolor": TEXT, "text.color": TEXT,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def load_results(outdir):
    """Combine every available experiment's model_comparison.csv into one table in the paper's schema."""
    frames = []
    for folder, label in EXPERIMENTS:
        path = os.path.join(outdir, folder, "model_comparison.csv")
        if not os.path.exists(path):
            continue
        mc = pd.read_csv(path)
        mc["Model"] = mc["Model"].replace({BASELINE_IN: BASELINE_OUT})
        frames.append(pd.DataFrame({
            "Experiment": label,
            "Model": mc["Model"],
            "Test Size": mc[["TN", "FP", "FN", "TP"]].sum(axis=1),
            "Accuracy": mc["Accuracy"].round(3),
            "Balanced Accuracy": mc["Balanced Accuracy"].round(3),
            "Precision": mc["Precision (excluded)"].round(3),
            "Recall": mc["Recall (excluded)"].round(3),
            "F1 (excluded)": mc["F1 (excluded)"].round(3),
            "Macro F1": mc["Macro F1"].round(3),
            "ROC AUC": mc["ROC AUC"].round(3),
            "Correct Connected (TN)": mc["TN"], "False Alarms (FP)": mc["FP"],
            "Missed Excluded (FN)": mc["FN"], "Correct Excluded (TP)": mc["TP"],
        }))
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    order = {m: i for i, m in enumerate(MODELS + [BASELINE_OUT])}
    table["_m"] = table["Model"].map(order)
    table["_e"] = table["Experiment"].map({label: i for i, (_, label) in enumerate(EXPERIMENTS)})
    return table.sort_values(["_e", "_m"]).drop(columns=["_m", "_e"]).reset_index(drop=True)


def plot_performance(table, path):
    experiments = [e for _, e in EXPERIMENTS if e in set(table["Experiment"])]
    models = MODELS + [BASELINE_OUT]
    metrics = [("Accuracy", "Accuracy"), ("Macro F1", "Macro F1"), ("ROC AUC", "ROC AUC")]
    n_exp = len(experiments)
    width = 0.8 / n_exp
    gap = 0.015

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        best_val, best_xy = -1, None
        for ei, exp in enumerate(experiments):
            sub = table[table["Experiment"] == exp].set_index("Model")
            for mi, model in enumerate(models):
                if model not in sub.index:
                    continue
                v = float(sub.loc[model, col])
                x = mi - 0.4 + width * (ei + 0.5)
                color = NEUTRAL if model == BASELINE_OUT else SERIES[ei]
                ax.bar(x, v, width=width - gap, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
                # direct-label the best bar among the protocol and prespecified experiments only; the
                # sensitivity run is drawn but never labeled as the best
                if model != BASELINE_OUT and "sensitivity" not in exp and v > best_val:
                    best_val, best_xy = v, (x, v)
        if best_xy:
            ax.annotate(f"{best_val:.3f}", best_xy, xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold", color=TEXT)
        if col == "ROC AUC":
            ax.axhline(0.5, color=TEXT_2, linestyle=(0, (3, 3)), linewidth=0.8, zorder=2)
            ax.text(len(models) - 1, 0.515, "chance (0.5)", ha="center", va="bottom", fontsize=7.5, color=TEXT_2)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(["Logistic\nregression", "Decision\ntree", "Random\nforest", "XGBoost",
                            "Baseline\n(always connected)"], fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(axis="both", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("Score on held-out test set (0 to 1)", fontsize=8.5, color=TEXT_2)

    handles = [Patch(facecolor=SERIES[i], label=exp) for i, exp in enumerate(experiments)]
    handles.append(Patch(facecolor=NEUTRAL, label="Baseline (same experiments, in order)"))
    # wrap onto a second row once the experiments no longer fit across the figure
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.955), ncol=min(len(handles), 4),
               frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.4)
    fig.suptitle("How well each model predicts digital exclusion", x=0.045, y=0.995, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.045, 0.022, "Excluded = no Internet use at any location; cohort = adults 65+ with a CPS disability, "
                           "November 2019, 2021 and 2023 CPS supplements. Bars start at zero.\n"
                           "Best bar among the protocol experiments labeled; the sensitivity run trades excluded-class "
                           "recall for accuracy (see confusion matrices).",
             fontsize=7.5, color=TEXT_2, va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9 if len(handles) <= 4 else 0.86))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_confusion(table, path):
    experiments = [e for _, e in EXPERIMENTS if e in set(table["Experiment"])]
    n = len(experiments)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.3 * nrows), squeeze=False)

    for k, exp in enumerate(experiments):
        ax = axes[k // ncols][k % ncols]
        sub = table[(table["Experiment"] == exp) & (table["Model"] != BASELINE_OUT)]
        best = sub.sort_values("Macro F1", ascending=False).iloc[0]
        tn, fp, fn, tp = (int(best[c]) for c in ["Correct Connected (TN)", "False Alarms (FP)",
                                                  "Missed Excluded (FN)", "Correct Excluded (TP)"])
        total = tn + fp + fn + tp
        cells = [[tn, fp], [fn, tp]]
        row_tot = [tn + fp, fn + tp]
        labels = [["correct", "false alarm"], ["missed", "correct"]]
        for i in range(2):
            for j in range(2):
                share = cells[i][j] / row_tot[i] if row_tot[i] else 0
                color = BLUES[min(int(share * len(BLUES)), len(BLUES) - 1)]
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=3))
                ink = SURFACE if share >= 0.55 else TEXT
                ax.text(j + 0.5, 1 - i + 0.62, f"{cells[i][j]:,}", ha="center", va="center", fontsize=15,
                        fontweight="bold", color=ink)
                ax.text(j + 0.5, 1 - i + 0.36, labels[i][j], ha="center", va="center", fontsize=8.5, color=ink)
                ax.text(j + 0.5, 1 - i + 0.22, f"{share:.0%} of actually {'connected' if i == 0 else 'excluded'}",
                        ha="center", va="center", fontsize=7.5, color=ink)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["Predicted connected", "Predicted excluded"], fontsize=8.5)
        ax.set_yticks([1.5, 0.5])
        ax.set_yticklabels(["Actually\nconnected", "Actually\nexcluded"], fontsize=8.5)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{exp}: {best['Model']}\nn = {total:,}  |  accuracy {tn + tp:,} / {total:,} = {(tn + tp) / total:.3f}"
                     f"  |  recall (excluded) {tp / (tp + fn):.2f}",
                     fontsize=9, pad=26, loc="left")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Predicted vs actual digital exclusion, best model by macro F1 in each experiment",
                 x=0.03, y=0.995, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.0, w_pad=2.0)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(outdir=OUTPUT_DIR, results_dir=RESULTS_DIR):
    table = load_results(outdir)
    if table is None:
        print(f"No model_comparison.csv found under {outdir}/; run main.py first.")
        return None
    os.makedirs(results_dir, exist_ok=True)
    paths = {
        "table": os.path.join(results_dir, "model_performance.csv"),
        "performance": os.path.join(results_dir, "model_performance.png"),
        "confusion": os.path.join(results_dir, "confusion_matrix.png"),
    }
    table.to_csv(paths["table"], index=False)
    plot_performance(table, paths["performance"])
    plot_confusion(table, paths["confusion"])
    print(f"Experiments drawn: {sorted(set(table['Experiment']), key=[e for _, e in EXPERIMENTS].index)}")
    for k, p in paths.items():
        print(f"Saved {k}: {p}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="folder with the experiment subfolders")
    parser.add_argument("--results", default=RESULTS_DIR, help="where to write the table and figures")
    args = parser.parse_args()
    main(args.outdir, args.results)
