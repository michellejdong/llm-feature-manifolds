"""Generate the summary figures used in the visual results guide."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parent
BLUE = "#3c678e"
TEAL = "#08786d"
ORANGE = "#a76529"
PURPLE = "#805d93"
GRAY = "#77889b"


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)


def discovery(results: Path, output: Path) -> None:
    corpora = ("Pile discovery", "Pile holdout", "WikiText", "News test")
    files = ("doc-cap-pile-fit", "doc-cap-pile-holdout", "doc-cap-wiki-train", "news-doc-cap-4")
    records = [{int(r["group"]): r for r in rows(results / "rankings" / f"{name}.csv")} for name in files]
    variants = ((138, "prototype_rank", "Weekdays · prototype persistence", BLUE),
                (251, "raw_rank", "Months · raw persistence", TEAL),
                (212, "raw_rank", "Years · raw persistence", ORANGE))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.2), layout="constrained")
    x = np.arange(len(corpora))
    for group, column, label, color in variants:
        left.plot(x, [int(table[group][column]) for table in records], "o-", label=label,
                  color=color, markersize=5, linewidth=1.5)
    left.axhspan(1, 25, color="#dceee7", alpha=0.7, zorder=0)
    left.set_yscale("log")
    left.set_ylim(1150, 0.8)
    left.set_yticks([1, 10, 25, 100, 1000], ["1", "10", "25", "100", "1,000"])
    left.set_xticks(x, corpora, rotation=15, ha="right")
    left.set_ylabel("Blind rank among 1,000 groups")
    left.set_title("Finding the groups blind", loc="left", weight="semibold")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncols=3)
    full = rows(results / "published/calendar-results.csv") + rows(results / "document-controls/calendar-results.csv")
    measurements = {(r["concept"], r["split"]): r for r in full}
    evaluation = ("pile-holdout", "wiki-train", "news-test")
    for concept, color in (("Weekdays", BLUE), ("Months", TEAL)):
        values = [100 * float(measurements[concept, split]["token_accuracy"]) for split in evaluation]
        right.plot(range(3), values, "o-", label=concept, color=color, linewidth=1.6, markersize=5)
        for i, value in enumerate(values):
            right.annotate(f"{value:.1f}%", (i, value), xytext=(0, 7), textcoords="offset points",
                           ha="center", color=color, fontsize=8)
    right.set_xticks(range(3), corpora[1:], rotation=15, ha="right")
    right.set_ylim(0, 109)
    right.set_ylabel("Named tokens nearest the right angle (%)")
    right.set_title("Reading the known groups after reveal", loc="left", weight="semibold")
    right.legend(frameon=False, loc="lower left", fontsize=8)
    for ax in (left, right):
        style(ax)
    fig.savefig(output / "discovery-and-transfer.png", dpi=210)
    plt.close(fig)


def steering(results: Path, output: Path) -> None:
    harmonic = json.loads((results / "steering/harmonics/harmonic-metadata.json").read_text())
    exact = {(r["condition"], r["dimensions"]): r for r in harmonic["exact_example"]}
    baseline = exact["harmonics", 2]["baseline_target_probability"]
    selection = [("harmonics", 2), ("harmonics", 8), ("harmonics", 11), ("pca", 11), ("random", 11), ("full_donor", 4096)]
    values = 100 * np.array([baseline, *[exact[key]["target_probability"] for key in selection]])
    names = ["Unedited", "2D calendar", "8D calendar", "11D calendar", "11D PCA", "11D random", "Full donor"]
    colors = [GRAY, TEAL, TEAL, TEAL, PURPLE, ORANGE, BLUE]
    natural = json.loads((results / "steering/published-sae/sae-plane-steering.json").read_text())
    conditions = [("natural_text_plane", "Natural-text plane", TEAL),
                  ("direct_sae_edit", "Direct SAE feature edit", BLUE),
                  ("supervised_plane_equal_norm", "Supervised plane, equal norm", PURPLE),
                  ("random_plane_equal_norm", "Random plane, equal norm", ORANGE)]
    natural_table = {(r["donor_shift"], r["condition"]): r for r in natural["aggregate"]}
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.15), layout="constrained")
    bars = left.barh(names, values, color=colors, height=0.65)
    left.bar_label(bars, labels=[f"{p:.1f}%" for p in values], padding=4, fontsize=9)
    left.set_xlim(0, 96)
    left.set_xlabel("February probability over the full vocabulary (%)")
    left.set_title("The November example", loc="left", weight="semibold")
    left.invert_yaxis()
    for j, (condition, label, color) in enumerate(conditions):
        selected = [natural_table[shift, condition] for shift in (1, 3)]
        mean = np.asarray([r["mean_target_gap_change"] for r in selected])
        lo, hi = np.array([r["ci95"] for r in selected]).T
        x = np.arange(2) + (j - 1.5) * 0.09
        right.errorbar(x, mean, yerr=[mean - lo, hi - mean], color=color,
                       marker="o", linewidth=1.15, markersize=4, capsize=3, label=label)
    right.axhline(0, color="#93a0ab", lw=0.7)
    right.set_xticks(np.arange(2), ["1 month later", "3 months later"])
    right.set_ylabel("Change in target vs. original answer logit gap")
    right.set_title("Editing a plane fit from natural text", loc="left", weight="semibold")
    right.legend(loc="upper left", frameon=False, fontsize=8)
    style(left, "x")
    style(right)
    fig.savefig(output / "calendar-edits.png", dpi=215)
    plt.close(fig)


def controls(results: Path, output: Path) -> None:
    observations = rows(results / "discovery/candidate-audit/candidate-audit.csv")
    selected = [(251, "Known months"), (138, "Known weekdays"), (77, "Group 77"),
                (334, "Group 334"), (519, "Group 519")]
    fig, ax = plt.subplots(figsize=(7.4, 3.7), layout="constrained")
    for i, (corpus, label, color) in enumerate((("news-test", "News", TEAL), ("wiki-test", "WikiText test", BLUE))):
        for j, (group, _) in enumerate(selected):
            current = [r for r in observations if r["split"] == corpus and int(r["group"]) == group and r["method"] == "raw"]
            deviations = np.array([float(r["observed"]) - float(r["null_median"]) for r in current])
            y = j + (i - 0.5) * 0.24
            marker = label if j == 0 else None
            ax.hlines(y, deviations.min(), deviations.max(), color=color, linewidth=1.0, alpha=0.55)
            ax.scatter(np.median(deviations), y, color=color, s=22, label=marker)
    ax.axvline(0, color="#94a1ac", linewidth=0.8)
    ax.set_yticks(range(len(selected)), [name for _, name in selected])
    ax.set_xlabel("Observed loop score minus median shuffled score")
    ax.set_title("Do the loops survive a sparse-coactivation control?", loc="left", weight="semibold")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.invert_yaxis()
    style(ax, "x")
    fig.savefig(output / "coactivation-controls.png", dpi=210)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT / "results")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/overview")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for plot in (discovery, steering, controls):
        plot(args.results, args.output)


if __name__ == "__main__":
    main()
