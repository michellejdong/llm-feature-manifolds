"""Measure transfer of the published twentieth-century year geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import PROJECT, load_decoder, load_groups, write_csv
from reveal import SPLIT_NAMES, labeled_points, load_dataset, read_ranking, sha256

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUP = 212


def phases(years: np.ndarray) -> np.ndarray:
    return np.exp(2j * np.pi * years / 100)


def calibrate(xy: np.ndarray, years: np.ndarray) -> tuple[complex, int, complex]:
    values = xy[:, 0] + 1j * xy[:, 1]
    observed = np.unique(years)
    if len(observed) < 3:
        raise ValueError("The fit corpus needs at least three distinct active years")
    centroids = np.array([values[years == year].mean() for year in observed])
    center = centroids.mean()
    unit = np.exp(1j * np.angle(centroids - center))
    phase, direction = max(((np.mean(phases(observed) * unit ** -s), s) for s in (-1, 1)),
                           key=lambda item: abs(item[0]))
    if abs(phase) < 1e-8:
        raise ValueError("There is no orientation to fit")
    return center, direction, phase / abs(phase)


def align(xy: np.ndarray, orientation: tuple) -> np.ndarray:
    center, sign, phase = orientation
    z = xy[:, 0] + 1j * xy[:, 1] - center
    return (z if sign == 1 else z.conj()) * phase


def measure(z: np.ndarray, years: np.ndarray, documents: np.ndarray) -> dict:
    observed = np.unique(years)
    errors = np.abs(np.angle(z / phases(years))) * 100 / (2 * np.pi)
    _, doc_ids, doc_counts = np.unique(documents, return_inverse=True, return_counts=True)
    per_doc = np.bincount(doc_ids, weights=errors) / doc_counts
    centered_years = np.array([z[years == yr].mean() for yr in observed])
    return {"named_tokens": len(years), "distinct_years": len(observed), "documents": len(doc_counts),
            "largest_document_share": float(doc_counts.max() / len(years)),
            "mean_circular_error_years": float(errors.mean()),
            "doc_macro_circular_error_years": float(per_doc.mean()),
            "fraction_within_five_years": float(np.mean(errors <= 5)),
            "year_centroid_mean_error": float(np.mean(np.abs(np.angle(centered_years / phases(observed)))) * 100 / (2 * np.pi))}


def draw(ax, z: np.ndarray, years: np.ndarray, split: str) -> None:
    rng = np.random.default_rng(20261125)
    sample = rng.choice(len(z), min(1800, len(z)), replace=False)
    ax.scatter(z[sample].real, z[sample].imag, c=years[sample] + 1900, cmap="viridis",
               vmin=1900, vmax=1999, s=7, alpha=0.36, linewidths=0)
    for i in range(0, 100, 10):
        in_decade = (years >= i) & (years < i + 10)
        if np.any(in_decade):
            centroid = z[in_decade].mean()
            ax.scatter(centroid.real, centroid.imag, color=plt.cm.viridis(i / 99), s=22,
                       edgecolor="#202936", linewidth=0.4)
    ax.set_title(SPLIT_NAMES.get(split, split), loc="left", fontsize=10, weight="semibold")
    ax.set_xlabel("Aligned PC 3 and PC 4", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, nargs="+", required=True, help="Calibration ranking first")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rankings) < 2:
        parser.error("Use at least one calibration and one evaluation ranking")
    decoder = load_decoder(args.work_dir)
    members = load_groups(PROJECT / "published-groups.json")[GROUP]
    tokenizer = AutoTokenizer.from_pretrained(args.work_dir / "assets/gpt2", local_files_only=True)
    ids = np.arange(len(tokenizer))
    words = tokenizer.batch_decode(ids.reshape(-1, 1).tolist())
    years = {i: int(word) for i, word in enumerate(map(str.strip, words))
             if len(word) == 4 and word.isascii() and word.isdecimal()}
    within = {code: yr - 1900 for code, yr in years.items() if 1900 <= yr <= 1999}
    fig, axs = plt.subplots(1, len(args.rankings), figsize=(3.2 * len(args.rankings), 3.7),
                            layout="constrained", sharex=True, sharey=True, squeeze=False)
    fig.suptitle("Twentieth-century years in GPT-2-small", fontsize=15, weight="semibold")
    legend = plt.cm.ScalarMappable(norm=plt.Normalize(1900, 1999), cmap="viridis")
    fig.colorbar(legend, ax=axs[0], orientation="horizontal", fraction=0.09, shrink=0.63, label="Token year")
    rows = []
    with threadpool_limits(limits=1):
        for i, path in enumerate(args.rankings):
            table, meta = read_ranking(path)
            data = load_dataset(args.work_dir, meta, len(decoder))
            background, points, labels, documents = labeled_points(data, members, decoder, within)
            if not len(labels):
                raise ValueError(f"No active twentieth-century years in {meta['split']}")
            if i == 0:
                center = background.mean(axis=0)
                _, _, right = np.linalg.svd(background - center, full_matrices=False)
                plane = right[2:4].T
                orientation = calibrate((points - center) @ plane, labels)
            coordinates = align((points - center) @ plane, orientation)
            active = np.unique(data["codes"][:, members].indices)
            coverage = {}
            for century in (1800, 1900, 2000):
                values = [code for code, year in years.items() if century <= year < century + 100]
                matches = np.isin(data["tokens"], values)
                coverage[f"activation_rate_{century}s"] = float(matches[active].sum() / max(1, matches.sum()))
            ranks = {key: int(v) for key, v in table[GROUP].items() if key.endswith("_rank")}
            rows.append({"split": meta["split"], "group": GROUP, **measure(coordinates, labels, documents), **coverage, **ranks})
            draw(axs[0, i], coordinates, labels, meta["split"])
            del data
    write_csv(args.output / "year-geometry.csv", rows)
    args.output.joinpath("year-geometry.json").write_text(json.dumps(
        {"ranking_sha256_before_reveal": {p.name: sha256(p) for p in args.rankings},
         "plane_fitted_on_components": [3, 4], "period_years": 100, "results": rows}, indent=2,
    ) + "\n")
    fig.savefig(args.output / "year-geometry.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
