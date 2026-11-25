"""Summarize matched SAE loss experiments across seeds and evaluation sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from rank import write_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OBJECTIVES = {
    "mse": ("MSE", "#71849b"),
    "relative-sqrt": ("Square-root weighting", "#a76529"),
    "relative-token": ("Per-token weighting", "#08786d"),
}


def draw(validation: list[dict], geometry: list[dict], output: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.1), layout="constrained")
    for seed in sorted({row["seed"] for row in validation}):
        paired = [row for row in validation if row["seed"] == seed]
        left.plot([r["full_validation_fvu"] * 100 for r in paired],
                  [r["mean_relative_token_error"] * 100 for r in paired], color="#cbd3dd", linewidth=1.15, zorder=1)
    for i, (loss, (name, color)) in enumerate(OBJECTIVES.items()):
        subset = [r for r in validation if r["loss"] == loss]
        left.scatter([r["full_validation_fvu"] * 100 for r in subset],
                     [r["mean_relative_token_error"] * 100 for r in subset], color=color, s=24, label=name, zorder=2)
        for j, concept in enumerate(("Weekdays", "Months")):
            rows = [r for r in geometry if r["loss"] == loss and r["split"] == "news-test" and r["concept"] == concept]
            scores = np.array([r["document_macro_accuracy"] for r in rows]) * 100
            x = j + (i - 1) * 0.21
            right.bar(x, scores.mean(), 0.17, color=color, alpha=0.82)
            right.scatter(x + np.linspace(-0.038, 0.038, len(scores)), scores, color="#132535",
                          edgecolor="white", linewidth=0.32, s=14, zorder=3)
    left.set_xlabel("Unexplained variance over validation tokens (%)")
    left.set_ylabel("Mean relative per-token error (%)")
    left.set_title("Reconstruction objectives", loc="left", fontsize=12, weight="semibold")
    left.legend(frameon=False, fontsize=8)
    right.set_xticks([0, 1], ["Weekdays", "Months"])
    right.set_ylabel("Document-macro angle accuracy (%)")
    right.set_title("Calendar geometry on news text", loc="left", fontsize=12, weight="semibold")
    right.set_ylim(0, 100)
    for ax in (left, right):
        ax.grid(axis="y", alpha=0.2, linewidth=0.75)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstructions", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validation = []
    model_names = {}
    for folder in args.models:
        config = json.loads((folder / "config.json").read_text())
        result = json.loads((folder / "validation.json").read_text())
        loss, seed = config.get("loss", "mse"), config["seed"]
        model_names[f"{seed}-{loss}"] = (seed, loss)
        validation.append({"seed": seed, "loss": loss, "steps": result["step"],
                           "full_validation_fvu": result["fvu_relative_to_fit_mean"],
                           "mean_relative_token_error": result["average_relative_token_error"]})
    validation.sort(key=lambda row: (row["seed"], list(OBJECTIVES).index(row["loss"])))
    actual = {(r["seed"], r["loss"]) for r in validation}
    expected = {(seed, loss) for seed, _ in actual for loss in OBJECTIVES}
    if actual != expected or len(actual) != len(validation):
        parser.error("Supply a matched run for every objective and seed")
    with args.reconstructions.open(newline="") as f:
        geometry = []
        for row in csv.DictReader(f):
            if row["representation"] in model_names:
                seed, loss = model_names[row["representation"]]
                geometry.append({"seed": seed, "loss": loss, "split": row["split"], "concept": row["concept"],
                                 "document_macro_accuracy": float(row["document_macro_accuracy"]),
                                 "token_accuracy": float(row["accuracy"]), "plane_fvu": float(row["plane_fvu"])})
    if {r["loss"] for r in geometry if r["split"] == "news-test"} != set(OBJECTIVES):
        parser.error("The reconstruction table needs news results for the matched objectives")
    write_csv(args.output / "matched-loss-validation.csv", validation)
    write_csv(args.output / "matched-loss-geometry.csv", geometry)
    draw(validation, geometry, args.output / "matched-loss-tradeoff.png")


if __name__ == "__main__":
    main()
