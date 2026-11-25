"""Learn new groups of the published SAE latents without using token identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from extract import device_for
from rank import PROJECT, activation_shards, load_decoder, load_documents, load_groups


def sample_document_rows(documents: np.ndarray, limit: int, count: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(len(documents))
    order = perm[np.argsort(documents[perm], kind="stable")]
    docs = documents[order]
    seq = np.arange(len(order))
    starts = np.maximum.accumulate(np.where(np.r_[True, docs[1:] != docs[:-1]], seq, 0))
    eligible = order[seq - starts < limit]
    return np.sort(rng.choice(eligible, size=min(count, len(eligible)), replace=False))


def binary_sample(paths: list[Path], width: int, points: int, per_doc: int, top_k: int, seed: int) -> csr_matrix:
    docs = load_documents(paths)
    chosen = sample_document_rows(docs, per_doc, points, np.random.default_rng(seed))
    indices, sizes = [], []
    offset = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            ptr, feat, weight = shard["indptr"], shard["indices"], shard["values"]
        stop = offset + len(ptr) - 1
        lo, hi = np.searchsorted(chosen, [offset, stop])
        for row in chosen[lo:hi] - offset:
            first, end = ptr[row:row + 2]
            k = min(end - first, top_k)
            if k == 0:
                sizes.append(0)
                continue
            best = np.argpartition(weight[first:end], -k)[-k:]
            sizes.append(k)
            indices.append(feat[first + best])
        offset = stop
    indices = np.concatenate(indices).astype(np.int32)
    offsets = np.r_[0, np.cumsum(sizes, dtype=np.int32)]
    return csr_matrix((np.ones(len(indices), dtype=np.float32), indices, offsets), shape=(len(chosen), width))


def top_edges(matrix: csr_matrix, k: int) -> csr_matrix:
    row_ids, col_ids, values = [], [], []
    for i in range(matrix.shape[0]):
        lo, hi = matrix.indptr[i:i + 2]
        row = matrix.data[lo:hi]
        if not len(row):
            continue
        count = min(len(row), k)
        chosen = np.argpartition(row, -count)[-count:]
        row_ids.extend([i] * count)
        col_ids.extend(matrix.indices[lo + chosen])
        values.extend(row[chosen])
    return csr_matrix((np.asarray(values, dtype=np.float32), (row_ids, col_ids)), shape=matrix.shape)


def coactivation_graph(binary: csr_matrix, neighbors: int) -> csr_matrix:
    frequency = np.asarray(binary.sum(axis=0)).ravel()
    co = (binary.T @ binary).tocoo()
    keep = (co.row != co.col) & (co.data >= 3)
    i, j, counts = co.row[keep], co.col[keep], co.data[keep]
    scaled = counts / np.sqrt(frequency[i] * frequency[j]) * (counts / (counts + 8))
    weighted = csr_matrix((scaled, (i, j)), shape=co.shape)
    pruned = top_edges(weighted, neighbors)
    return pruned.maximum(pruned.T)


@torch.inference_mode()
def decoder_graph(decoder: np.ndarray, neighbors: int, device: str) -> csr_matrix:
    unit = torch.nn.functional.normalize(torch.as_tensor(decoder, dtype=torch.float32, device=device), dim=1)
    row_ids, col_ids, values = [], [], []
    for start in range(0, len(unit), 192):
        block = unit[start:start + 192]
        scores, indices = (block @ unit.T).topk(neighbors + 1, dim=1, sorted=False)
        scores, indices = scores.cpu().numpy(), indices.cpu().numpy()
        rows = start + np.arange(len(block))[:, None]
        keep = (indices != rows) & (scores > 0.08)
        row_ids.append(np.broadcast_to(rows, indices.shape)[keep])
        col_ids.append(indices[keep])
        values.append(scores[keep])
    graph = csr_matrix((np.concatenate(values), (np.concatenate(row_ids), np.concatenate(col_ids))),
                       shape=(len(unit), len(unit)))
    return graph.maximum(graph.T)


def communities(graph: nx.Graph, resolution: float, seed: int) -> dict[int, list[int]]:
    blocks = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
    result = sorted((sorted(block) for block in blocks), key=lambda block: (block[0], len(block)))
    return {i: block for i, block in enumerate(result) if len(block) >= 2}


def overlaps(new_groups: dict[int, list[int]], published: dict[int, list[int]]) -> list[dict]:
    candidate_sets = {i: set(features) for i, features in new_groups.items()}
    rows = []
    for published_id in (138, 251, 212):
        reference = set(published[published_id])
        alternatives = [(len(reference & group) / len(reference | group), len(reference & group), group_id)
                        for group_id, group in candidate_sets.items()]
        jaccard, total, match = max(alternatives, key=lambda entry: (entry[0], entry[1], -entry[2]))
        rows.append({"published_group": published_id, "new_group": match, "shared_features": total,
                     "published_features": len(reference), "matched_group_size": len(candidate_sets[match]),
                     "jaccard": jaccard})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".experiment"))
    parser.add_argument("--split", default="pile-fit")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=150_000)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--resolution", type=float, default=5.0)
    parser.add_argument("--target-groups", type=int, default=0,
                        help="Choose from up to six doubling steps by proximity to this size, without labels")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    if args.tokens < 100 or not 2 <= args.neighbors <= 100 or args.resolution <= 0:
        parser.error("Use at least 100 tokens, 2–100 neighbors, and a positive resolution")
    args.output.mkdir(parents=True, exist_ok=True)
    decoder = load_decoder(args.work_dir)
    paths = activation_shards(args.work_dir / "activations" / args.split, args.max_shards)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        print("Building document-balanced co-activation data", flush=True)
        binary = binary_sample(paths, len(decoder), args.tokens, 128, 12, args.seed)
        activity = coactivation_graph(binary, args.neighbors)
        print(f"Found {activity.nnz:,} co-activation edges from {binary.shape[0]:,} tokens", flush=True)
        cosine = decoder_graph(decoder, args.neighbors, device_for(args.device))
    variants = {"decoder": cosine, "joint": 0.65 * cosine + 0.35 * activity}
    reference = load_groups(PROJECT / "published-groups.json")
    results = {"fit_split": args.split, "sampled_tokens": binary.shape[0], "neighbors": args.neighbors,
               "resolution": args.resolution, "seed": args.seed, "uses_token_ids": False, "variants": {}}
    for name, graph in variants.items():
        network = nx.from_scipy_sparse_array(graph)
        passes = []
        selected_resolution = args.resolution
        for exponent in range(6 if args.target_groups else 1):
            resolution = args.resolution * 2 ** exponent
            trial = communities(network, resolution, args.seed)
            passes.append((resolution, len(trial)))
            if exponent == 0 or abs(len(trial) - args.target_groups) < abs(len(grouped) - args.target_groups):
                grouped, selected_resolution = trial, resolution
            print(f"{name} graph at resolution {resolution:g}: {len(trial):,} groups", flush=True)
            if args.target_groups and len(trial) >= args.target_groups:
                break
        (args.output / f"{name}-groups.json").write_text(json.dumps(grouped, separators=(",", ":")) + "\n")
        results["variants"][name] = {"n_groups": len(grouped), "largest_group": max(map(len, grouped.values())),
                                      "selected_resolution": selected_resolution, "resolution_sweep": passes,
                                      "published_overlaps": overlaps(grouped, reference)}
        print(f"{name} graph yielded {len(grouped):,} communities with at least two features", flush=True)
    (args.output / "clustering-results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
