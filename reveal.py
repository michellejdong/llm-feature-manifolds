"""Reveal calendar labels and measure geometry after blind candidate ranking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import PROJECT, activation_shards, load_codes, load_decoder, load_groups, write_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CALENDARS = {
    138: ("Weekdays", ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")),
    251: (
        "Months",
        ("January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"),
    ),
}
SPLIT_NAMES = {"pile-fit": "Pile discovery", "pile-holdout": "Pile holdout", "wiki-train": "WikiText"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ranking(path: Path) -> tuple[dict, dict]:
    with path.open(newline="") as f:
        rows = {int(row["group"]): row for row in csv.DictReader(f)}
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["ranking_uses_tokens"] is not False:
        raise ValueError(f"The ranking is not label-blind: {path}")
    return rows, meta


def load_dataset(work_dir: Path, meta: dict, width: int) -> dict:
    paths = activation_shards(work_dir / "activations" / meta["split"], meta["shards"])
    codes = load_codes(paths, width)
    tokens, documents = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            tokens.append(shard["tokens"])
            documents.append(shard["documents"])
    if codes.shape[0] != meta["tokens"]:
        raise ValueError("Ranking and current activation cache cover different data")
    return {"codes": codes, "tokens": np.concatenate(tokens), "documents": np.concatenate(documents)}


def labeled_points(data: dict, features: list[int], decoder: np.ndarray, ids: dict) -> tuple:
    local = data["codes"][:, features]
    active = np.unique(local.indices)
    labeled_mask = np.isin(data["tokens"][active], list(ids))
    labeled = active[labeled_mask]
    labels = np.array([ids[int(token)] for token in data["tokens"][labeled]], dtype=int)
    _, r = np.linalg.qr(decoder[features].T, mode="reduced")
    points = local[labeled].toarray() @ r.T
    rng = np.random.default_rng(20261125)
    background = np.sort(rng.choice(active, min(len(active), 16_000), replace=False))
    return local[background].toarray() @ r.T, points, labels, data["documents"][labeled]


def orient_circle(coordinates: np.ndarray, labels: np.ndarray, classes: int) -> tuple[complex, int, complex]:
    centroids = np.array([coordinates[labels == i].mean() for i in range(classes)])
    center = centroids.mean()
    directions = np.exp(1j * np.angle(centroids - center))
    ideal = np.exp(2j * np.pi * np.arange(classes) / classes)
    rotations = [(np.mean(ideal * (directions ** -sign)), sign) for sign in (1, -1)]
    phase, sign = max(rotations, key=lambda entry: abs(entry[0]))
    return center, sign, phase / max(abs(phase), 1e-8)


def angular_summary(points: np.ndarray, labels: np.ndarray, classes: int) -> dict:
    estimates = np.angle(points)
    ideal = 2 * np.pi * labels / classes
    errors = np.abs(np.angle(np.exp(1j * (estimates - ideal))))
    predicted = np.round(estimates * classes / (2 * np.pi)).astype(int) % classes
    centers = np.array([points[labels == i].mean() for i in range(classes) if np.any(labels == i)])
    ideal_centers = 2 * np.pi * np.unique(labels) / classes
    center_error = np.abs(np.angle(np.exp(1j * (np.angle(centers) - ideal_centers))))
    return {
        "labeled_tokens": int(len(labels)),
        "counts": np.bincount(labels, minlength=classes).tolist(),
        "token_accuracy": float(np.mean(predicted == labels)),
        "token_mae_degrees": float(np.degrees(np.mean(errors))),
        "centroid_mae_degrees": float(np.degrees(np.mean(center_error))),
    }


def geometry_panel(ax, points: np.ndarray, labels: np.ndarray, names: tuple, tags: str, rng) -> None:
    colors = plt.colormaps["twilight"](np.arange(len(names)) / len(names))
    for index, name in enumerate(names):
        group = points[labels == index]
        if not len(group):
            continue
        sample = rng.choice(group, min(160, len(group)), replace=False)
        center = group.mean()
        ax.scatter(sample.real, sample.imag, s=10, alpha=0.18, c=[colors[index]], linewidths=0)
        ax.scatter(center.real, center.imag, s=26, c=[colors[index]], linewidths=0.45, edgecolors="#111827", zorder=3)
        right = center.real >= 0
        ax.annotate(name[:3], (center.real, center.imag), xytext=(4 if right else -4, 3),
                    textcoords="offset points", ha="left" if right else "right",
                    color="#172033", fontsize=8, weight="medium", zorder=4)
    ax.set_title(tags, loc="left", fontsize=11, weight="medium", color="#172033", pad=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.axvline(0, color="#a9b2bf", linewidth=0.6, zorder=0)
    ax.axhline(0, color="#a9b2bf", linewidth=0.6, zorder=0)
    ax.set_xlabel("Aligned PC 2–3", color="#4b5563", fontsize=8)
    ax.tick_params(labelsize=7, colors="#4b5563", length=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#c8cfd8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".experiment"))
    parser.add_argument("--rankings", nargs="+", type=Path, required=True, help="Discovery ranking first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=Path, default=PROJECT / "published-groups.json")
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--calendar-groups", type=int, nargs=2, default=[138, 251], metavar=("WEEKDAYS", "MONTHS"))
    args = parser.parse_args()
    decoder = np.load(args.decoder, allow_pickle=False) if args.decoder else load_decoder(args.work_dir)
    groups = load_groups(args.groups)
    ranks, meta = zip(*(read_ranking(path) for path in args.rankings), strict=True)
    datasets = [load_dataset(args.work_dir, item, len(decoder)) for item in meta]
    tokenizer = AutoTokenizer.from_pretrained(args.work_dir / "assets/gpt2", local_files_only=True)
    used_ids = np.unique(np.concatenate([item["tokens"] for item in datasets]))
    strings = tokenizer.batch_decode(used_ids.astype(int).reshape(-1, 1).tolist())
    id_to_token = {int(i): word.strip() for i, word in zip(used_ids, strings, strict=True)}
    args.output.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, len(meta), figsize=(max(6, 4.5 * len(meta)), 7.0),
                            sharex="row", sharey="row", squeeze=False, layout="constrained")
    fig.suptitle("Calendar geometry in GPT-2-small", fontsize=17, weight="semibold", color="#172033")
    rows = []
    output = {"ranking_sha256_before_label_reveal": {path.name: sha256(path) for path in args.rankings}, "calendars": {}}
    doc_sets = [set(item["documents"]) for item in datasets]
    output["document_overlap_counts"] = {
        f"{meta[0]['split']} vs {item['split']}": len(doc_sets[0] & doc_sets[i])
        for i, item in enumerate(meta[1:], 1)
    }
    rng = np.random.default_rng(20261125)
    with threadpool_limits(limits=1):
        for row_index, (group, (title, names)) in enumerate(zip(args.calendar_groups, CALENDARS.values(), strict=True)):
            lookup = {word: i for i, word in enumerate(names)}
            ids = {i: lookup[token] for i, token in id_to_token.items() if token in lookup}
            fit_background, fit_points, fit_labels, _ = labeled_points(datasets[0], groups[group], decoder, ids)
            if len(np.unique(fit_labels)) < len(names):
                raise ValueError(f"Discovery sample is missing a calendar class for group {group}")
            center = fit_background.mean(axis=0)
            _, strength, right = np.linalg.svd(fit_background - center, full_matrices=False)
            axes = right[1:3].T
            coords = (fit_points - center) @ axes
            calibration = orient_circle(coords[:, 0] + 1j * coords[:, 1], fit_labels, len(names))
            phase_center, sign, phase = calibration
            calendar = {"group": group, "labels": names, "pc2_pc3_variance": float(np.sum(strength[1:3] ** 2) / np.sum(strength ** 2)), "splits": {}}
            for i, (data, table, info) in enumerate(zip(datasets, ranks, meta, strict=True)):
                _, points, labels, documents = labeled_points(data, groups[group], decoder, ids)
                coords = (points - center) @ axes
                complex_points = coords[:, 0] + 1j * coords[:, 1] - phase_center
                aligned = (complex_points if sign == 1 else np.conj(complex_points)) * phase
                metrics = angular_summary(aligned, labels, len(names))
                _, doc_counts = np.unique(documents, return_counts=True)
                metrics.update(labeled_documents=len(doc_counts), largest_document_fraction=float(doc_counts.max() / len(labels)))
                metrics.update({k: int(v) for k, v in table[group].items() if k.endswith("_rank")})
                row = {"concept": title, "group": group, "split": info["split"], "corpus_tokens": info["tokens"]}
                rows.append(row | {k: v for k, v in metrics.items() if k != "counts"})
                calendar["splits"][info["split"]] = metrics
                display_split = SPLIT_NAMES.get(info["split"], info["split"])
                geometry_panel(axs[row_index, i], aligned, labels, names, f"{title} · {display_split}", rng=rng)
            output["calendars"][title] = calendar
    write_csv(args.output / "calendar-results.csv", rows)
    (args.output / "calendar-results.json").write_text(json.dumps(output, indent=2) + "\n")
    fig.savefig(args.output / "calendar-geometry.png", dpi=200)
    plt.close(fig)

    top_rows = []
    for i, (data, table, info) in enumerate(zip(datasets, ranks, meta, strict=True)):
        for mode in info["modes"]:
            for record in sorted(table.values(), key=lambda row: int(row[f"{mode}_rank"]))[:10]:
                group = int(record["group"])
                active = np.unique(data["codes"][:, groups[group]].indices)
                strongest = Counter(id_to_token[int(token)] for token in data["tokens"][active]).most_common(8)
                top_rows.append({"split": info["split"], "mode": mode, "group": group,
                                 "rank": record[f"{mode}_rank"], "score": record[f"{mode}_score"],
                                 "most_frequent_active_tokens": json.dumps(strongest, ensure_ascii=False)})
    write_csv(args.output / "top-candidates.csv", top_rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
