"""Choose tighter SAE subgroups using agreement between unlabeled corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

from rank import (
    activation_shards, cluster_cloud, load_codes, load_decoder, load_documents,
    load_groups, score_mode, write_csv,
)


SEED = 52277
POINTS = 128
DOCUMENT_CAP = 4
MIN_GAIN = 0.02


def branches(features: list[int], decoder: np.ndarray, limit: int = 5) -> list[list[int]]:
    """Return branches that remain separate across the largest linkage intervals."""
    if len(features) < 4:
        return []
    distance = pdist(decoder[features].astype(np.float64), metric="cosine")
    tree = linkage(np.maximum(distance, 0), method="average")
    descendants = [[i] for i in features]
    levels = [0.0] * len(features)
    candidates = []
    for a, b, height, _ in tree:
        a, b = int(a), int(b)
        for child in (a, b):
            if len(descendants[child]) >= 3:
                persistence = max(0, height - levels[child])
                candidates.append((persistence, sorted(descendants[child])))
        descendants.append(descendants[a] + descendants[b])
        levels.append(height)
    candidates.sort(key=lambda entry: (-entry[0], len(entry[1]), entry[1]))
    return [members for _, members in candidates[:limit]]


def robust_scores(members, datasets, decoder, group) -> tuple[float, list[float]]:
    scores = []
    for index, (codes, documents) in enumerate(datasets):
        repetitions = []
        for rep in range(2):
            rng = np.random.default_rng([SEED, group, index, rep])
            cloud, _, _ = cluster_cloud(codes, decoder, members, POINTS, rng, "balanced",
                                        documents=documents, per_document_limit=DOCUMENT_CAP)
            repetitions.append(score_mode(cloud, "raw")[0])
        scores.append(float(np.mean(repetitions)))
    return min(scores), scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--fit-splits", nargs=2, default=["pile-fit", "wiki-train"])
    parser.add_argument("--fit-shards", type=int, nargs=2, default=[24, 20])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.fit_splits)) != 2 or min(args.fit_shards) < 1:
        parser.error("Use two different fit corpora and a positive shard count")
    groups = load_groups(args.groups)
    decoder = load_decoder(args.work_dir)
    datasets = []
    for split, max_shards in zip(args.fit_splits, args.fit_shards, strict=True):
        paths = activation_shards(args.work_dir / "activations" / split, max_shards)
        datasets.append((load_codes(paths, len(decoder)), load_documents(paths)))
    refined = {}
    table = []
    with threadpool_limits(limits=1):
        for i, (group, parent) in enumerate(sorted(groups.items()), 1):
            children = branches(parent, decoder)
            selected = parent
            base, base_scores = robust_scores(parent, datasets, decoder, group)
            best, best_scores = base, base_scores
            for child in children:
                score, by_corpus = robust_scores(child, datasets, decoder, group)
                if score >= base + MIN_GAIN and score > best:
                    best, best_scores, selected = score, by_corpus, child
            refined[group] = selected
            table.append({"group": group, "parent_size": len(parent), "chosen_size": len(selected),
                          "children_tested": len(children), "parent_minimum_fit_score": base,
                          "chosen_minimum_fit_score": best, "parent_fit_scores": json.dumps(base_scores),
                          "chosen_fit_scores": json.dumps(best_scores)})
            if i % 100 == 0:
                print(f"Refined {i:,}/{len(groups):,} groups", flush=True)
    write_csv(args.output / "group-refinements.csv", table)
    (args.output / "groups.json").write_text(json.dumps(refined, separators=(",", ":")) + "\n")
    manifest = {"fit_splits": args.fit_splits, "fit_shard_limits": args.fit_shards, "uses_tokens": False,
                "samples_per_fit_corpus": 2, "points_per_sample": POINTS, "document_cap": DOCUMENT_CAP,
                "minimum_fit_gain": MIN_GAIN, "seed": SEED,
                "refined_groups": sum(len(groups[g]) != len(members) for g, members in refined.items())}
    (args.output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
