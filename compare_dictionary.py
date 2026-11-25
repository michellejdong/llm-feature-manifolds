"""Compare an independently trained SAE with the published calendar subspaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import PROJECT, activation_shards, load_decoder, load_groups, write_csv
from reveal import CALENDARS, angular_summary, load_dataset, orient_circle, read_ranking


def reference_plane(data: dict, decoder: np.ndarray, features: list[int]) -> np.ndarray:
    local = data["codes"][:, features]
    active = np.unique(local.indices)
    rows = np.sort(np.random.default_rng(20261125).choice(active, min(len(active), 16_000), replace=False))
    q, r = np.linalg.qr(decoder[features].T, mode="reduced")
    cloud = local[rows].toarray() @ r.T
    _, _, axes = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
    return q @ axes[1:3].T


def projected_overlap(decoder: np.ndarray, features: list[int], plane: np.ndarray) -> float:
    u, sigma, _ = np.linalg.svd(decoder[features].T, full_matrices=False)
    basis = u[:, sigma > 1e-5 * sigma[0]]
    captured = np.square(basis.T @ plane).sum() / plane.shape[1]
    expected = basis.shape[1] / len(decoder[0])
    return float((captured - expected) / (1 - expected)) if expected < 1 else 0.0


def recode_named_tokens(work_dir: Path, ranks: list, model: Path, names: set[int]) -> list[dict]:
    import torch
    from safetensors.torch import load_file
    from train_sae import TopKSAE

    config = json.loads((model / "config.json").read_text())
    sae = TopKSAE(config["width"], config["k"], block_size=config.get("block_size", 1))
    sae.load_state_dict(load_file(model / "topk-sae.safetensors"))
    sae.eval()
    torch.set_num_threads(2)
    outputs = []
    with torch.inference_mode():
        for info in ranks:
            tokens, codes, values, counts = [], [], [], []
            paths = activation_shards(work_dir / "activations" / info["split"], info["shards"])
            for path in paths:
                with np.load(path, allow_pickle=False) as shard:
                    ids = shard["tokens"]
                    keep = np.isin(ids, list(names))
                    if not keep.any():
                        continue
                    hidden = torch.from_numpy(shard["residuals"][keep]).float()
                z, pos = sae.sparse_code(sae.normalized(hidden))
                values_here, codes_here = z.numpy(), pos.numpy()
                mask = values_here > 1e-6
                counts.append(mask.sum(axis=1))
                tokens.append(ids[keep])
                values.append(values_here[mask])
                codes.append(codes_here[mask])
            if tokens:
                flat_ids = np.concatenate(tokens)
                ptr = np.r_[0, np.cumsum(np.concatenate(counts), dtype=np.int32)]
                matrix = csr_matrix((np.concatenate(values), np.concatenate(codes), ptr),
                                    shape=(len(flat_ids), config["width"])).tocsc()
            else:
                flat_ids = np.empty(0, dtype=np.uint16)
                matrix = csr_matrix((0, config["width"]), dtype=np.float32).tocsc()
            outputs.append({"codes": matrix, "tokens": flat_ids})
    return outputs


def coordinates(data: dict, decoder: np.ndarray, features: list[int], plane: np.ndarray, ids: dict):
    candidate = np.flatnonzero(np.isin(data["tokens"], list(ids)))
    features_at_candidate = data["codes"][:, features][candidate].tocsr()
    active = features_at_candidate.getnnz(axis=1) > 0
    vals = features_at_candidate[active] @ (decoder[features] @ plane)
    labels = np.array([ids[int(token)] for token in data["tokens"][candidate[active]]], dtype=int)
    return vals[:, 0] + 1j * vals[:, 1], labels, len(candidate)


def score_representation(fit: tuple, test: tuple, classes: int) -> dict:
    points, labels, total = test
    if len(np.unique(fit[1])) != classes or not len(labels):
        return {"coverage": len(labels) / max(total, 1), "labeled_tokens": len(labels),
                "token_accuracy": None, "centroid_mae_degrees": None}
    center, sign, phase = orient_circle(fit[0], fit[1], classes)
    centered = points - center
    aligned = (centered if sign == 1 else np.conj(centered)) * phase
    metrics = angular_summary(aligned, labels, classes)
    return {"coverage": len(labels) / max(total, 1), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".experiment"))
    parser.add_argument("--published", type=Path, nargs="+", required=True, help="Fit ranking followed by test rankings")
    parser.add_argument("--trained", type=Path, nargs="+", help="Corresponding blind rankings from the trained SAE. Omit for a geometry-only evaluation of the matching groups.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trained and len(args.published) != len(args.trained):
        parser.error("Published and trained rankings must cover the same data splits")

    published_decoder = load_decoder(args.work_dir)
    trained_decoder = np.load(args.model / "decoder.npy", allow_pickle=False)
    pg = load_groups(PROJECT / "published-groups.json")
    tg = load_groups(args.model / "groups.json")
    _, published_meta = zip(*(read_ranking(path) for path in args.published), strict=True)
    published = [load_dataset(args.work_dir, info, len(published_decoder)) for info in published_meta]
    tokenizer = AutoTokenizer.from_pretrained(args.work_dir / "assets/gpt2", local_files_only=True)
    token_codes = np.unique(np.concatenate([data["tokens"] for data in published]))
    tokens = tokenizer.batch_decode(token_codes.astype(int).reshape(-1, 1).tolist())
    token_names = {int(code): token.strip() for code, token in zip(token_codes, tokens, strict=True)}
    trained_tables = None
    if args.trained:
        trained_tables, trained_meta = zip(*(read_ranking(path) for path in args.trained), strict=True)
        trained = [load_dataset(args.work_dir, info, len(trained_decoder)) for info in trained_meta]
        for reference, comparison in zip(published, trained, strict=True):
            if not (np.array_equal(reference["tokens"], comparison["tokens"])
                    and np.array_equal(reference["documents"], comparison["documents"])):
                raise ValueError("The two dictionaries were evaluated using different token contexts")
    else:
        all_names = {name for _, names in CALENDARS.values() for name in names}
        ids = {code for code, name in token_names.items() if name in all_names}
        trained = recode_named_tokens(args.work_dir, list(published_meta), args.model, ids)
    rows = []
    with threadpool_limits(limits=2):
        for group_id, (concept, names) in CALENDARS.items():
            labels = {code: names.index(token) for code, token in token_names.items() if token in names}
            plane = reference_plane(published[0], published_decoder, pg[group_id])
            # Match the reference plane against whole decoder groups. This never
            # uses month or weekday names, which only enter the evaluation below.
            candidates = sorted(
                ((projected_overlap(trained_decoder, features, plane), group) for group, features in tg.items()),
                reverse=True,
            )
            group = candidates[0][1]
            combined_features = sorted({feature for _, g in candidates[:3] for feature in tg[g]})
            variants = [
                ("published_group", group_id, pg[group_id], published, published_decoder),
                ("trained_best_group", group, tg[group], trained, trained_decoder),
                ("trained_three_groups", None, combined_features, trained, trained_decoder),
                ("trained_all_features", None, list(range(len(trained_decoder))), trained, trained_decoder),
            ]
            for source, group_number, features, datasets, decoder in variants:
                fit_points = coordinates(datasets[0], decoder, features, plane, labels)
                for i, data in enumerate(datasets):
                    test_points = fit_points if i == 0 else coordinates(data, decoder, features, plane, labels)
                    metrics = score_representation(fit_points, test_points, len(names))
                    rank = trained_tables[i][group]["raw_rank"] if trained_tables and source == "trained_best_group" else None
                    rows.append({"concept": concept, "dictionary": source, "group": group_number,
                                 "split": published_meta[i]["split"], "group_features": len(features),
                                 "adjusted_plane_overlap": candidates[0][0] if source == "trained_best_group" else None,
                                 "trained_raw_blind_rank": rank,
                                 **{k: metrics[k] for k in ("coverage", "labeled_tokens", "token_accuracy", "centroid_mae_degrees")}})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "dictionary-comparison.csv", rows)
    (args.output / "dictionary-comparison.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
