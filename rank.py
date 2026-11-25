"""Rank reconstructed SAE groups by their geometry without reading token labels."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from ripser import ripser
from safetensors.numpy import load_file
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from geometry_metrics import adjacent_pc_planes, irreducibility

PROJECT = Path(__file__).resolve().parent
MODES = ("raw", "unit", "transverse")
ADDITIONAL_MODES = ("pca23", "shell", "annulus", "prototype", "pca-adjacent", "irreducibility")


def activation_shards(folder: Path, max_shards: int) -> list[Path]:
    paths = sorted(folder.glob("[0-9][0-9][0-9][0-9][0-9].npz"))
    if max_shards:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"No cached activation shards in {folder}")
    return paths


def load_codes(paths: list[Path], width: int) -> csc_matrix:
    values, indices, sizes = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            values.append(shard["values"])
            indices.append(shard["indices"])
            sizes.append(np.diff(shard["indptr"]))
    counts = np.concatenate(sizes)
    pointers = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    return csr_matrix(
        (np.concatenate(values), np.concatenate(indices), pointers),
        shape=(len(counts), width),
    ).tocsc()


def load_decoder(work_dir: Path) -> np.ndarray:
    weights = work_dir / "assets/sae/blocks.7.hook_resid_pre/sae_weights.safetensors"
    return load_file(weights)["W_dec"]


def load_documents(paths: list[Path]) -> np.ndarray:
    arrays = []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            arrays.append(shard["documents"])
    return np.concatenate(arrays)


def load_groups(path: Path) -> dict[int, list[int]]:
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def sample_rows(
    group_codes: csc_matrix, budget: int, rng: np.random.Generator, sampling: str,
    total_mass: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    active = np.unique(group_codes.indices)
    if len(active) <= budget or sampling == "uniform":
        if len(active) > budget:
            return np.sort(rng.choice(active, size=budget, replace=False)), len(active)
        return active, len(active)
    if sampling == "salient":
        members = np.searchsorted(active, group_codes.indices)
        mass = np.bincount(members, weights=np.square(group_codes.data), minlength=len(active))
        share = mass / np.maximum(total_mass[active], 1e-8)
        pool = min(2 * budget, len(active))
        best = np.argpartition(share, -pool)[-pool:]
        return np.sort(active[rng.choice(best, size=budget, replace=False)]), len(active)

    # Balance the feature groups' contributions using only activation strengths.
    # Draw from a high-activation pool for each live feature, then fill vacancies.
    live_features = np.flatnonzero(np.diff(group_codes.indptr))
    shares = np.ones(len(live_features), dtype=int) * (budget // len(live_features))
    shares[:budget % len(live_features)] += 1
    rows = []
    for feature, share in zip(rng.permutation(live_features), shares, strict=True):
        if not share:
            continue
        start, stop = group_codes.indptr[feature:feature + 2]
        values = group_codes.data[start:stop]
        feature_rows = group_codes.indices[start:stop]
        if sampling == "salient-balanced":
            values = np.square(values) / np.maximum(total_mass[feature_rows], 1e-8)
        pool_size = min(len(values), max(8 * share, 16))
        best = np.argpartition(values, -pool_size)[-pool_size:]
        selected = rng.choice(best, min(share, pool_size), replace=False)
        rows.extend(feature_rows[selected])
    chosen = np.unique(rows).astype(active.dtype)
    if len(chosen) < budget:
        remaining = np.setdiff1d(active, chosen, assume_unique=True)
        chosen = np.r_[chosen, rng.choice(remaining, size=budget - len(chosen), replace=False)]
    return np.sort(chosen), len(active)


def cluster_cloud(
    codes: csc_matrix, decoder: np.ndarray, features: list[int], budget: int,
    rng: np.random.Generator, sampling: str, total_mass: np.ndarray | None = None,
    documents: np.ndarray | None = None, per_document_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    group_codes = codes[:, features]
    eligible = None
    if per_document_limit:
        eligible = document_capped_rows(group_codes, documents, per_document_limit)
        group_codes = group_codes[eligible].tocsc()
        if total_mass is not None:
            total_mass = total_mass[eligible]
    rows, available = sample_rows(group_codes, budget, rng, sampling, total_mass)
    # QR expresses the reconstruction in an orthonormal basis of the decoder
    # span. Keeping all columns of that basis preserves Euclidean distances.
    _, r = np.linalg.qr(decoder[features].T, mode="reduced")
    return group_codes[rows].toarray() @ r.T, rows if eligible is None else eligible[rows], available


def document_capped_rows(codes: csc_matrix, documents: np.ndarray, limit: int) -> np.ndarray:
    active = np.unique(codes.indices)
    if not len(active):
        return active
    strengths = np.bincount(np.searchsorted(active, codes.indices), weights=codes.data.astype(float) ** 2,
                            minlength=len(active))
    doc_ids = documents[active]
    order = np.lexsort((active, -strengths, doc_ids))
    ordered_ids = doc_ids[order]
    offsets = np.arange(len(active))
    starts = np.maximum.accumulate(np.where(np.r_[True, ordered_ids[:-1] != ordered_ids[1:]], offsets, 0))
    return np.sort(active[order[offsets - starts < limit]])


def activation_mass(codes: csc_matrix) -> np.ndarray:
    mass = np.zeros(codes.shape[0], dtype=np.float32)
    for start in range(0, len(codes.data), 2_000_000):
        chunk = slice(start, start + 2_000_000)
        np.add.at(mass, codes.indices[chunk], np.square(codes.data[chunk]))
    return mass


def geometry(points: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return points
    if mode == "unit":
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / np.maximum(norms, 1e-8)
    centered = points - points.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    if mode == "pca23":
        return centered @ right[1:3].T
    return centered @ right[1:].T


def loop_scores(points: np.ndarray) -> tuple[float, float, float]:
    if len(points) < 4 or points.shape[1] < 2:
        return 0.0, 0.0, 0.0
    distances = pdist(points)
    scale = np.quantile(distances, 0.9)
    if not np.isfinite(scale) or scale <= 1e-8:
        return 0.0, 0.0, 0.0
    diagram = ripser(squareform(distances / scale), maxdim=1, distance_matrix=True)["dgms"][1]
    if not len(diagram):
        return 0.0, 0.0, 0.0
    lives = diagram[:, 1] - diagram[:, 0]
    best = int(lives.argmax())
    birth, death = diagram[best]
    return float(lives[best]), float(birth), float(death)


def gaussian_cloud(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = points.mean(axis=0)
    _, scales, axes = np.linalg.svd(points - center, full_matrices=False)
    noise = rng.normal(size=(len(points), len(scales)))
    return (noise * (scales / np.sqrt(len(points) - 1))) @ axes + center


def annulus_score(points: np.ndarray) -> float:
    """Reward an empty center, a narrow radius, and coverage of the circle."""
    if len(points) < 8 or points.shape[1] < 2:
        return 0.0
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    scores = []
    for first in range(min(2, len(axes) - 1)):
        plane = centered @ axes[first:first + 2].T
        scale = np.sqrt(np.mean(np.square(plane)))
        if scale < 1e-8:
            continue
        xy = plane / scale
        matrix = np.c_[2 * xy, np.ones(len(xy))]
        fitted = np.linalg.lstsq(matrix, np.square(xy).sum(axis=1), rcond=None)[0][:2]
        delta = xy - fitted
        radius = np.linalg.norm(delta, axis=1)
        lo, median, hi = np.quantile(radius, [0.2, 0.5, 0.8])
        if median < 1e-8:
            continue
        angles = np.sort(np.arctan2(delta[:, 1], delta[:, 0]))
        max_gap = np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max()
        coverage = max(0, 1 - max_gap / np.pi)
        scores.append(coverage * (lo / median) * max(0, 1 - (hi - lo) / median))
    return float(max(scores, default=0.0))


def score_mode(points: np.ndarray, mode: str) -> tuple[float, float, float]:
    if not len(points):
        return 0.0, 0.0, 0.0
    if mode == "annulus":
        return annulus_score(points), 0.0, 0.0
    if mode == "pca-adjacent":
        return max((loop_scores(xy) for xy in adjacent_pc_planes(points)), default=(0.0, 0.0, 0.0))
    if mode == "irreducibility":
        return irreducibility(points), 0.0, 0.0
    if mode == "prototype":
        plane = geometry(points, "pca23")
        if len(plane) < 12 or plane.shape[1] < 2:
            return 0.0, 0.0, 0.0
        clusters = min(24, int(np.sqrt(len(plane))), len(np.unique(plane, axis=0)))
        if clusters < 4:
            return 0.0, 0.0, 0.0
        model = KMeans(n_clusters=clusters, n_init=3, random_state=5001).fit(plane)
        sizes = np.bincount(model.labels_, minlength=clusters)
        return loop_scores(model.cluster_centers_[sizes >= max(2, len(plane) // 50)])
    if mode != "shell":
        return loop_scores(geometry(points, mode))
    # A cone can have circular cross sections even when its full cloud has no
    # hole. Score three separate bands of activation magnitude.
    order = np.argsort(np.linalg.norm(points, axis=1), kind="stable")
    return sorted(loop_scores(points[idx]) for idx in np.array_split(order, 3))[1]


def scores_for_group(points: np.ndarray, controls: int, modes: list[str], rng: np.random.Generator):
    reference = [gaussian_cloud(points, rng) for _ in range(controls)] if len(points) >= 4 else []
    outputs = {}
    for mode in modes:
        score, birth, death = score_mode(points, mode)
        result = {f"{mode}_score": score}
        if mode not in ("annulus", "irreducibility"):
            result |= {f"{mode}_birth": birth, f"{mode}_death": death}
        if controls:
            null = [score_mode(sample, mode)[0] for sample in reference]
            result[f"{mode}_null_median"] = float(np.median(null)) if null else 0.0
            result[f"{mode}_null_p"] = (1 + sum(s >= score for s in null)) / (1 + len(null))
            result[f"{mode}_null_excess"] = score - result[f"{mode}_null_median"]
        outputs.update(result)
    return outputs


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split")
    parser.add_argument("--work-dir", type=Path, default=Path(".experiment"))
    parser.add_argument("--groups", type=Path, default=PROJECT / "published-groups.json")
    parser.add_argument("--decoder", type=Path, help="Optional NumPy array of a separately trained decoder")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--points", type=int, default=192)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--per-document-limit", type=int, default=0,
                        help="Keep at most this many active points per document, chosen by SAE group strength")
    parser.add_argument("--sampling", choices=("uniform", "balanced", "salient", "salient-balanced"), default="uniform")
    parser.add_argument("--modes", choices=MODES + ADDITIONAL_MODES, nargs="+", default=list(MODES))
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--controls", type=int, default=0)
    parser.add_argument("--only-groups", type=int, nargs="+")
    args = parser.parse_args()
    if args.points < 4 or args.repeats < 1 or min(args.controls, args.per_document_limit) < 0:
        parser.error("Use at least four points, one repeat, and a nonnegative control count")
    decoder = np.load(args.decoder, allow_pickle=False) if args.decoder else load_decoder(args.work_dir)
    groups = load_groups(args.groups)
    selected = sorted(groups if args.only_groups is None else args.only_groups)
    paths = activation_shards(args.work_dir / "activations" / args.split, args.max_shards)
    codes = load_codes(paths, len(decoder))
    documents = load_documents(paths) if args.per_document_limit else None
    total_mass = activation_mass(codes) if args.sampling.startswith("salient") else None
    print(f"Scoring {len(selected):,} groups across {codes.shape[0]:,} tokens", flush=True)
    started = time.monotonic()
    rows = []
    with threadpool_limits(limits=1):
        for i, group_id in enumerate(selected, 1):
            members = groups[group_id]
            samples = []
            for repetition in range(args.repeats):
                entropy = [args.seed, group_id] if repetition == 0 else [args.seed, group_id, repetition]
                rng = np.random.default_rng(np.random.SeedSequence(entropy))
                points, _, available = cluster_cloud(codes, decoder, members, args.points, rng, args.sampling,
                                                      total_mass, documents, args.per_document_limit)
                samples.append(scores_for_group(points, args.controls, args.modes, rng))
            row = {
                "group": group_id, "features": len(members), "active_tokens": available,
                "sampled_tokens": len(points),
            }
            row.update({name: float(np.median([sample[name] for sample in samples])) for name in samples[0]})
            if args.repeats > 1:
                for mode in args.modes:
                    draws = [sample[f"{mode}_score"] for sample in samples]
                    row[f"{mode}_score_min"] = min(draws)
                    row[f"{mode}_score_max"] = max(draws)
            rows.append(row)
            if i % 100 == 0 or i == len(selected):
                print(f"{i:,}/{len(selected):,} in {time.monotonic() - started:.1f}s", flush=True)
    for mode in args.modes:
        ordered = sorted(rows, key=lambda row: (-row[f"{mode}_score"], row["group"]))
        for ordinal, row in enumerate(ordered, 1):
            row[f"{mode}_rank"] = ordinal
        if args.controls:
            ordered = sorted(rows, key=lambda row: (-row[f"{mode}_null_excess"], row["group"]))
            for ordinal, row in enumerate(ordered, 1):
                row[f"{mode}_null_excess_rank"] = ordinal
    first = args.modes[0]
    write_csv(args.output, sorted(rows, key=lambda row: row[f"{first}_rank"]))
    info = {
        "split": args.split, "tokens": codes.shape[0], "shards": len(paths),
        "sampling": args.sampling, "points_per_group": args.points, "seed": args.seed,
        "per_document_limit": args.per_document_limit,
        "modes": args.modes, "repeats": args.repeats, "gaussian_controls_per_group_per_repeat": args.controls,
        "ranking_uses_tokens": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
