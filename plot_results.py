# Title: plot_results.py
# Description: Draws charts of model performance from results/model_performance.csv (written by main.py).
# Run on its own to redraw the charts without retraining: python plot_results.py
#
# Outputs:
#   results/model_performance.png  - Accuracy, Macro F1 and ROC AUC for every model in every experiment
#   results/confusion_matrix.png   - predicted vs actual for the best model in the first experiment

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write image files only, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_FILE = "results/model_performance.csv"
PERFORMANCE_CHART = "performance_chart"
CONFUSION_CHART = "confusion_chart"

BASELINE_PREFIX = "Baseline"
METRICS = ["Accuracy", "Macro F1", "ROC AUC"]

# colorblind-safe categorical order (validated for up to three series); one color per experiment
EXPERIMENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
BASELINE_COLOR = "#b4b2a9"
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
SEQUENTIAL_BLUE = ["#eef4fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#184f95"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def short_name(model):
    if model.startswith(BASELINE_PREFIX):
        return "Baseline\n(always\nconnected)"
    return model.replace(" ", "\n", 1)


# Accuracy, Macro F1 and ROC AUC side by side; bars grouped by experiment
def plot_performance(df, path):
    experiments = list(dict.fromkeys(df["Experiment"]))
    models = [m for m in dict.fromkeys(df["Model"]) if not m.startswith(BASELINE_PREFIX)]
    models.append(next(m for m in df["Model"] if m.startswith(BASELINE_PREFIX)))

    width = 0.8 / len(experiments)
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 5.2), sharey=True, facecolor=SURFACE)
    for ax, metric in zip(axes, METRICS):
        style_axes(ax)
        best_value, best_xy = -1, None
        for i, experiment in enumerate(experiments):
            rows = df[df["Experiment"] == experiment].set_index("Model")
            values = [rows.loc[m, metric] if m in rows.index else np.nan for m in models]
            colors = [BASELINE_COLOR if m.startswith(BASELINE_PREFIX) else EXPERIMENT_COLORS[i] for m in models]
            positions = x - 0.4 + width * (i + 0.5)
            ax.bar(positions, values, width, color=colors, edgecolor=SURFACE, linewidth=1.5)
            for pos, value, model in zip(positions, values, models):
                if not model.startswith(BASELINE_PREFIX) and value > best_value:
                    best_value, best_xy = value, (pos, value)

        # label only the highest bar in each panel
        ax.annotate(f"{best_value:.3f}", best_xy, xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=TEXT_PRIMARY, fontweight="bold")
        if metric == "ROC AUC":
            ax.axhline(0.5, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
            ax.text(len(models) - 0.5, 0.51, "chance (0.5)", ha="right", va="bottom",
                    fontsize=8, color=TEXT_SECONDARY)

        ax.set_title(metric, fontsize=12, color=TEXT_PRIMARY, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([short_name(m) for m in models])
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Score on held-out test set (0-1)", color=TEXT_SECONDARY)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in EXPERIMENT_COLORS[:len(experiments)]]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLOR))
    fig.legend(handles, experiments + ["Baseline (same three experiments, in order)"],
               loc="upper center", ncol=len(experiments) + 1, frameon=False, fontsize=9,
               labelcolor=TEXT_PRIMARY, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("How well each model predicts digital exclusion", fontsize=14, color=TEXT_PRIMARY,
                 x=0.06, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# 2x2 predicted-vs-actual grid for the best model (highest macro F1) in one experiment
def plot_confusion(df, path, experiment=None):
    if experiment is None:
        experiment = df["Experiment"].iloc[0]
    rows = df[(df["Experiment"] == experiment) & ~df["Model"].str.startswith(BASELINE_PREFIX)]
    best = rows.loc[rows["Macro F1"].idxmax()]

    tn, fp = best["Correct Connected (TN)"], best["False Alarms (FP)"]
    fn, tp = best["Missed Excluded (FN)"], best["Correct Excluded (TP)"]
    counts = np.array([[tn, fp], [fn, tp]])
    shares = counts / counts.sum(axis=1, keepdims=True)
    notes = [["correct", "false alarm"], ["missed", "correct"]]
    row_names = ["connected", "excluded"]

    fig, ax = plt.subplots(figsize=(7, 5.6), facecolor=SURFACE)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQUENTIAL_BLUE)
    ax.imshow(counts, cmap=cmap, vmin=0, vmax=counts.max())

    for r in range(2):
        for c in range(2):
            dark_cell = counts[r, c] > counts.max() * 0.55
            color = SURFACE if dark_cell else TEXT_PRIMARY
            ax.text(c, r - 0.08, f"{counts[r, c]:,}", ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)
            ax.text(c, r + 0.18, f"{notes[r][c]}\n{shares[r, c]:.0%} of actually {row_names[r]}",
                    ha="center", va="center", fontsize=9.5, color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted connected", "Predicted excluded"], color=TEXT_PRIMARY)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actually connected", "Actually excluded"], color=TEXT_PRIMARY)
    ax.tick_params(length=0)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=4)
    ax.tick_params(which="minor", length=0)

    total = counts.sum()
    fig.suptitle(f"{best['Model']}: predicted vs actual digital exclusion", fontsize=13,
                 color=TEXT_PRIMARY)
    ax.set_title(f"{experiment} test set, n={total:,}  |  accuracy {tn + tp:,} / {total:,} = "
                 f"{(tn + tp) / total:.3f}", fontsize=10, color=TEXT_SECONDARY, pad=34)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(results_file=RESULTS_FILE):
    if not Path(results_file).exists():
        print(f"No results table found at {results_file}; run main.py first.")
        return None

    df = pd.read_csv(results_file)
    out_dir = Path(results_file).parent
    paths = {
        PERFORMANCE_CHART: out_dir / "model_performance.png",
        CONFUSION_CHART: out_dir / "confusion_matrix.png",
    }
    plot_performance(df, paths[PERFORMANCE_CHART])
    plot_confusion(df, paths[CONFUSION_CHART])
    print(f"Saved charts to {paths[PERFORMANCE_CHART]} and {paths[CONFUSION_CHART]}")
    return paths


if __name__ == "__main__":
    main(*sys.argv[1:])
