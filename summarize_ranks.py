"""Summarize known-feature recovery and agreement between corpus rankings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

from rank import write_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CORPORA = {"full-fit": "Pile discovery", "pile-holdout": "Pile holdout", "wiki-train": "WikiText"}
CONCEPTS = {138: "Weekdays", 251: "Months", 212: "20th-century years"}
SETTINGS = ("uniform", "balanced")


def load_table(path: Path) -> dict[int, dict]:
    with path.open(newline="") as f:
        rows = {int(row["group"]): row for row in csv.DictReader(f)}
    if set(rows) != set(range(1_000)):
        raise ValueError(f"Expected the published 1,000 groups in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables = {(corpus, sampler): load_table(args.rankings / f"{corpus}-{sampler}.csv")
              for corpus in CORPORA for sampler in SETTINGS}
    ranks = [
        {"concept": concept, "group": i, "corpus": CORPORA[corpus], "sampling": sampling,
         "active_tokens": table[i]["active_tokens"],
         **{k: int(v) for k, v in table[i].items() if k.endswith("_rank")}}
        for (corpus, sampling), table in tables.items() for i, concept in CONCEPTS.items()
    ]
    write_csv(args.output / "known-group-ranks.csv", ranks)

    stability = []
    for sampler in SETTINGS:
        fit = tables["full-fit", sampler]
        for corpus in tuple(CORPORA)[1:]:
            other = tables[corpus, sampler]
            for rank_key in (k for k in fit[0] if k.endswith("_rank")):
                a = np.array([int(fit[i][rank_key]) for i in range(1000)])
                b = np.array([int(other[i][rank_key]) for i in range(1000)])
                stability.append({"sampling": sampler, "method": rank_key.removesuffix("_rank"),
                                  "holdout": CORPORA[corpus], "spearman": float(spearmanr(a, b).statistic),
                                  "shared_top_25": int(((a <= 25) & (b <= 25)).sum()),
                                  "shared_top_100": int(((a <= 100) & (b <= 100)).sum())})
    write_csv(args.output / "ranking-stability.csv", stability)

    fig, axs = plt.subplots(1, len(CONCEPTS), figsize=(10.6, 4.25), sharey=True, layout="constrained")
    labels = ["Pile\ndiscovery", "Pile\nholdout", "WikiText"]
    choices = [
        ("raw_rank", "balanced", "Raw · balanced", "#0d766e"),
        ("raw_rank", "uniform", "Raw · uniform", "#6d85ad"),
        ("unit_rank", "balanced", "Unit norm · balanced", "#9a6840"),
        ("pca23_rank", "balanced", "PC 2–3 · balanced", "#99679a"),
    ]
    for ax, (group, concept) in zip(axs, CONCEPTS.items(), strict=True):
        for key, sampler, title, color in choices:
            values = [int(tables[corpus, sampler][group][key]) for corpus in CORPORA]
            ax.plot(labels, values, "o-", label=title, color=color, linewidth=1.65, markersize=4)
        ax.set_title(concept, loc="left", fontsize=11, weight="semibold")
        ax.set_yscale("log")
        ax.set_ylim(1100, 1)
        ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
        ax.tick_params(labelsize=9, length=0)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#b0bac7")
    axs[0].set_ylabel("Blind rank · lower is better", fontsize=10)
    handles, legend = axs[0].get_legend_handles_labels()
    fig.legend(handles, legend, loc="outside lower center", frameon=False, ncols=4, fontsize=8.5)
    fig.suptitle("Recovery of three published calendar feature groups", fontsize=15, weight="semibold")
    fig.savefig(args.output / "rank-transfer.png", dpi=210)
    plt.close(fig)


if __name__ == "__main__":
    main()
