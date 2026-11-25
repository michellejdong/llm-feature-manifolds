"""Stress-test blind shape scores by adding controlled rings to real activation clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from rank import (
    PROJECT, activation_shards, cluster_cloud, load_codes, load_decoder,
    load_documents, load_groups, score_mode, write_csv,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("raw", "unit", "pca23", "shell", "annulus", "prototype")


def shape_pair(background: np.ndarray, strength: float, rng: np.random.Generator):
    """Match the ring and disk's expected squared radius within each pair."""
    n, d = background.shape
    plane, _ = np.linalg.qr(rng.normal(size=(d, 2)), mode="reduced")
    centered = background - background.mean(axis=0)
    residual = centered - (centered @ plane) @ plane.T
    scale = np.sqrt(np.square(residual).sum(axis=1).mean())
    phase = rng.uniform(0, 2 * np.pi, size=n)
    direction = np.c_[np.cos(phase), np.sin(phase)]
    disk_radii = np.sqrt(2 * rng.uniform(size=(n, 1)))
    ring = direction @ plane.T
    disk = (direction * disk_radii) @ plane.T
    jitter = 0.04 * rng.normal(size=(n, 2)) @ plane.T
    base = residual + strength * scale * jitter
    return base + strength * scale * ring, base + strength * scale * disk


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for strength in sorted({row["relative_strength"] for row in rows}):
        for method in METHODS:
            pairs = [row for row in rows if row["relative_strength"] == strength and row["method"] == method]
            pos = np.array([row["ring_score"] for row in pairs])
            neg = np.array([row["disk_score"] for row in pairs])
            auc = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
            pair_wins = np.mean((pos > neg) + 0.5 * (pos == neg))
            summary.append({"relative_strength": strength, "method": method, "pairs": len(pairs),
                            "auc": float(auc), "paired_win_rate": float(pair_wins),
                            "average_score_gap": float(np.mean(pos - neg))})
    return summary


def plot_summary(summary: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.9, 4.4), layout="constrained")
    labels = {"raw": "Raw persistence", "unit": "Unit norm", "pca23": "PC 2–3 persistence",
              "shell": "Norm bands", "annulus": "Annulus fit", "prototype": "Prototype persistence"}
    for method in METHODS:
        rows = [entry for entry in summary if entry["method"] == method]
        ax.plot([entry["relative_strength"] for entry in rows], [entry["auc"] for entry in rows],
                marker="o", markersize=4, linewidth=1.65, label=labels[method])
    ax.set_xscale("log", base=2)
    values = sorted({row["relative_strength"] for row in summary})
    ax.set_xticks(values, [f"{n:g}×" for n in values])
    ax.axhline(0.5, color="#6b7280", lw=0.8, ls="--")
    ax.set_xlabel("Injected radius relative to background RMS")
    ax.set_ylabel("AUROC for ring vs. filled disk")
    ax.set_ylim(0, 1.02)
    ax.set_title("Loop recovery in real activation backgrounds", loc="left", fontsize=13, weight="semibold")
    ax.grid(alpha=0.17)
    ax.legend(loc="lower right", frameon=False, ncols=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=210)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".experiment"))
    parser.add_argument("--split", default="pile-fit")
    parser.add_argument("--backgrounds", type=int, default=24)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--points", type=int, default=192)
    parser.add_argument("--seed", type=int, default=56023)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.backgrounds, args.trials) < 1 or args.points < 12:
        parser.error("Use positive background and trial counts and at least 12 points")
    args.output.mkdir(parents=True, exist_ok=True)
    decoder = load_decoder(args.work_dir)
    groups = load_groups(PROJECT / "published-groups.json")
    shards = activation_shards(args.work_dir / "activations" / args.split, 0)
    codes = load_codes(shards, len(decoder))
    docs = load_documents(shards)
    rng = np.random.default_rng(args.seed)
    selected = [i for i in rng.permutation(list(groups)) if 6 <= len(groups[i]) <= 40]
    rows, included = [], []
    with threadpool_limits(limits=1):
        for group in selected:
            cloud, _, _ = cluster_cloud(codes, decoder, groups[group], args.points, rng, "uniform", documents=docs,
                                        per_document_limit=4)
            if len(cloud) != args.points or np.linalg.matrix_rank(cloud - cloud.mean(axis=0)) < 4:
                continue
            included.append(int(group))
            for strength in (0.25, 0.5, 1.0, 2.0):
                for trial in range(args.trials):
                    ring, disk = shape_pair(cloud, strength, rng)
                    for method in METHODS:
                        rows.append({"group": group, "relative_strength": strength, "trial": trial, "method": method,
                                     "ring_score": score_mode(ring, method)[0], "disk_score": score_mode(disk, method)[0]})
            print(f"Background {len(included)}/{args.backgrounds} · feature group {group}", flush=True)
            if len(included) == args.backgrounds:
                break
    if not rows:
        raise ValueError("No suitable activation clouds were found")
    write_csv(args.output / "geometry-benchmark-pairs.csv", rows)
    summary = summarize(rows)
    write_csv(args.output / "geometry-benchmark.csv", summary)
    (args.output / "geometry-benchmark.json").write_text(json.dumps(
        {"split": args.split, "points": args.points, "seed": args.seed,
         "trials_per_group_and_strength": args.trials, "background_groups": included,
         "uses_token_labels": False}, indent=2,
    ) + "\n")
    plot_summary(summary, args.output / "geometry-benchmark.png")


if __name__ == "__main__":
    main()
