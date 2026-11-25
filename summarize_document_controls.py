"""Summarize cross-corpus rank stability after document-balanced sampling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rank import write_csv


def read(path: Path) -> dict[int, dict]:
    with path.open(newline="") as f:
        return {int(row["group"]): row for row in csv.DictReader(f)}


def rank_geometric_mean(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = np.exp(np.log(values).mean(axis=0))
    rank = np.empty(len(score), dtype=int)
    rank[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return score, rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.rankings) < 3:
        parser.error("Use rankings from at least three corpora")
    tables = [read(path) for path in args.rankings]
    groups = sorted(tables[0])
    if any(set(table) != set(groups) for table in tables):
        raise ValueError("Every corpus must use the same feature groups")
    modes = [key.removesuffix("_rank") for key in tables[0][groups[0]] if key.endswith("_rank")]
    if not modes:
        raise ValueError("The inputs need to be blind ranking tables")
    all_scores = []
    transfer = []
    for mode in modes:
        ranks = np.stack([[int(table[group][f"{mode}_rank"]) for group in groups] for table in tables])
        geomeans, order = rank_geometric_mean(ranks)
        for i, group in enumerate(groups):
            all_scores.append({"group": group, "method": mode, "consensus_rank": int(order[i]),
                               "geometric_mean_rank": float(geomeans[i]),
                               "worst_corpus_rank": int(ranks[:, i].max())})
        for i, path in enumerate(args.rankings):
            _, train_rank = rank_geometric_mean(np.delete(ranks, i, axis=0))
            for k in (25, 100):
                transfer.append({"omitted_corpus": path.stem, "method": mode, "top_k": k,
                                 "overlap": int(((train_rank <= k) & (ranks[i] <= k)).sum())})
    all_scores.sort(key=lambda row: (row["method"], row["consensus_rank"]))
    write_csv(args.output / "corpus-consensus-ranks.csv", all_scores)
    write_csv(args.output / "corpus-consensus-transfer.csv", transfer)


if __name__ == "__main__":
    main()
