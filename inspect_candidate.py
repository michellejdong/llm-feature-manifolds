"""Inspect a suffix-associated candidate after the geometry rankings are frozen."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import PROJECT, document_capped_rows, load_decoder, load_groups
from reveal import load_dataset, read_ranking

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TYPES = ("Exact suffix", "Attached fragment", "Word-initial token")
COLORS = ("#33658a", "#0d766e", "#bd6d36")


def suffix_type(token: str, suffix: str) -> int:
    clean = token.strip().casefold()
    if clean == suffix:
        return 0
    if clean.endswith(suffix):
        return 2 if token[:1].isspace() else 1
    return -1


def sample_group(data: dict, decoder: np.ndarray, group: list[int], limit: int) -> tuple:
    local = data["codes"][:, group]
    rows = document_capped_rows(local, data["documents"], limit)
    _, r = np.linalg.qr(decoder[group].T, mode="reduced")
    z = local[rows].toarray()
    strength = np.square(z).sum(axis=1)
    ordered = np.argsort(strength, kind="stable")[-min(len(strength), 256):]
    return z[ordered] @ r.T, rows[ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", type=int)
    parser.add_argument("--suffix", default="ing")
    parser.add_argument("--work-dir", type=Path, default=Path(".experiment"))
    parser.add_argument("--rankings", nargs="+", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    suffix = args.suffix.casefold()
    features = load_groups(PROJECT / "published-groups.json")[args.group]
    decoder = load_decoder(args.work_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.work_dir / "assets/gpt2", local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, len(args.rankings), figsize=(max(4.4 * len(args.rankings), 6), 4),
                            squeeze=False, sharex=True, sharey=True, layout="constrained")
    fig.suptitle(f"GPT-2 feature group {args.group} · −{suffix} token types", fontsize=15, weight="semibold")
    rows = []
    center, axes, centroids = None, None, None
    with threadpool_limits(limits=1):
        for ax, path in zip(axs[0], args.rankings, strict=True):
            rankings, meta = read_ranking(path)
            data = load_dataset(args.work_dir, meta, len(decoder))
            points, selected = sample_group(data, decoder, features, 4)
            if len(points) < 4 or points.shape[1] < 3:
                raise ValueError(f"Too few activations or dimensions for group {args.group}")
            visible = tokenizer.batch_decode(data["tokens"][selected].astype(int).reshape(-1, 1).tolist())
            labels = np.array([suffix_type(token, suffix) for token in visible])
            vocab = np.unique(data["tokens"])
            vocab_strings = tokenizer.batch_decode(vocab.astype(int).reshape(-1, 1).tolist())
            suffix_ids = [code for code, token in zip(vocab, vocab_strings, strict=True) if suffix_type(token, suffix) >= 0]
            corpus_rate = float(np.isin(data["tokens"], suffix_ids).mean())
            selected_rate = float((labels >= 0).mean())
            if center is None:
                center = points.mean(axis=0)
                _, _, v = np.linalg.svd(points - center, full_matrices=False)
                axes = v[1:3].T
                train_coordinates = (points - center) @ axes
                if all(np.any(labels == i) for i in range(len(TYPES))):
                    centroids = np.stack([train_coordinates[labels == i].mean(axis=0) for i in range(len(TYPES))])
            coords = (points - center) @ axes
            accuracy = None
            if centroids is not None and np.any(labels >= 0):
                best = np.square(coords[:, None, :] - centroids).sum(axis=-1).argmin(axis=1)
                accuracy = float(np.mean([np.mean(best[labels == i] == i) for i in np.unique(labels[labels >= 0])]))
            record = {"split": meta["split"], "raw_blind_rank": int(rankings[args.group]["raw_rank"]),
                      "selection_points": len(points), "selection_documents": int(len(np.unique(data["documents"][selected]))),
                      "corpus_suffix_rate": corpus_rate, "selected_suffix_rate": selected_rate,
                      "selected_type_counts": np.bincount(labels[labels >= 0], minlength=len(TYPES)).tolist(),
                      "mean_type_accuracy": accuracy,
                      "top_selected_tokens": Counter(token.strip() for token in visible).most_common(12)}
            rows.append(record)
            ax.scatter(*coords[labels < 0].T, c="#b9c2cc", s=12, alpha=0.35, linewidths=0, label="Other token")
            for i, (name, color) in enumerate(zip(TYPES, COLORS, strict=True)):
                ax.scatter(*coords[labels == i].T, c=color, s=16, alpha=0.62, linewidths=0, label=name)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(meta["split"], loc="left", fontsize=10, weight="medium")
            ax.set_xlabel("PC 2 and PC 3 projection", fontsize=8)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    handles, names = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", frameon=False, ncols=4, fontsize=8)
    fig.savefig(args.output / f"candidate-{args.group}.png", dpi=200)
    plt.close(fig)
    report = {"group": args.group, "suffix": suffix, "analysis": rows,
              "selection": "Strongest 256 active points after a cap of four per document"}
    (args.output / f"candidate-{args.group}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
