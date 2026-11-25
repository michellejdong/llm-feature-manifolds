"""Evaluate whole-residual reconstruction and calendar geometry on identical tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file as load_array_weights
from safetensors.torch import load_file
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from compare_dictionary import reference_plane
from rank import PROJECT, activation_shards, load_groups, write_csv
from reveal import CALENDARS, orient_circle, read_ranking, load_dataset
from train_sae import TopKSAE


def named_residuals(work_dir: Path, dataset: dict, meta: dict, ids: set[int]) -> tuple[np.ndarray, np.ndarray]:
    selected = np.flatnonzero(np.isin(dataset["tokens"], list(ids)))
    pieces = []
    paths = activation_shards(work_dir / "activations" / meta["split"], meta["shards"])
    offset = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            tokens = shard["tokens"]
            lo, hi = np.searchsorted(selected, [offset, offset + len(tokens)])
            if lo != hi:
                pieces.append(shard["residuals"][selected[lo:hi] - offset])
        offset += len(tokens)
    return selected, np.concatenate(pieces).astype(np.float32)


def open_model(label: str, model_dir: str) -> tuple[str, TopKSAE]:
    folder = Path(model_dir)
    settings = json.loads((folder / "config.json").read_text())
    model = TopKSAE(settings["width"], settings["k"], block_size=settings.get("block_size", 1))
    model.load_state_dict(load_file(folder / "topk-sae.safetensors"))
    return label, model.eval()


@torch.inference_mode()
def reconstruct(model: TopKSAE, residuals: np.ndarray) -> np.ndarray:
    result = []
    for start in range(0, len(residuals), 512):
        batch = torch.from_numpy(residuals[start:start + 512])
        decoded, _, _ = model(model.normalized(batch))
        result.append((decoded * model.scale + model.center).numpy())
    return np.concatenate(result)


def angle_metrics(test: np.ndarray, label: np.ndarray, documents: np.ndarray, fit: tuple, classes: int) -> dict:
    center, sign, rotation = fit
    points = test[:, 0] + 1j * test[:, 1] - center
    aligned = (points if sign == 1 else points.conj()) * rotation
    angles = np.angle(aligned)
    predicted = np.rint(angles * classes / (2 * np.pi)).astype(int) % classes
    exact = predicted == label
    _, inverse, counts = np.unique(documents, return_inverse=True, return_counts=True)
    document_hits = np.bincount(inverse, weights=exact) / counts
    angular_error = np.abs(np.angle(np.exp(1j * (angles - 2 * np.pi * label / classes))))
    return {"accuracy": float(exact.mean()), "document_macro_accuracy": float(document_hits.mean()),
            "mean_angular_error_degrees": float(np.degrees(angular_error.mean())),
            "documents": len(counts)}


def report(args, files: list[Path], models: dict[str, TopKSAE]) -> list[dict]:
    weights = load_array_weights(args.work_dir / "assets/sae/blocks.7.hook_resid_pre/sae_weights.safetensors")
    decoder = weights["W_dec"]
    groups = load_groups(PROJECT / "published-groups.json")
    tokens = AutoTokenizer.from_pretrained(args.work_dir / "assets/gpt2", local_files_only=True)
    names = {word for _, calendar in CALENDARS.values() for word in calendar}
    vocab = np.arange(len(tokens))
    strings = tokens.batch_decode(vocab.reshape(-1, 1).tolist())
    ids = {code: s.strip() for code, s in enumerate(strings) if s.strip() in names}
    rows = []
    planes = {}
    alignments = {}
    centers = {}
    for i, ranking in enumerate(files):
        _, meta = read_ranking(ranking)
        dataset = load_dataset(args.work_dir, meta, len(decoder))
        named, original = named_residuals(args.work_dir, dataset, meta, set(ids))
        code = dataset["codes"][named].tocsr()
        reconstructions = {"raw_residual": original,
                           "published_full": code @ decoder + weights["b_dec"]}
        reconstructions.update({name: reconstruct(model, original) for name, model in models.items()})
        for group, (concept, calendar) in CALENDARS.items():
            features = groups[group]
            if i == 0:
                planes[group] = reference_plane(dataset, decoder, features)
            plane = planes[group]
            mask = np.array([ids[int(token)] in calendar for token in dataset["tokens"][named]])
            label = np.array([calendar.index(ids[int(token)]) for token in dataset["tokens"][named[mask]]])
            docs = dataset["documents"][named[mask]]
            used = len(label)
            raw = original[mask]
            plane_true = raw @ plane
            if i == 0:
                centers[group] = raw.mean(axis=0)
            centered = raw - centers[group]
            plane_variance = float(np.square(centered @ plane).sum())
            total_variance = float(np.square(centered).sum())
            local = code[mask][:, features]
            keep = local.getnnz(axis=1) > 0
            active = local @ (decoder[features] @ plane)
            views = {name: (value[mask] @ plane, np.ones(used, dtype=bool),
                            float(np.square(raw - value[mask]).sum()) / total_variance)
                     for name, value in reconstructions.items()}
            views["published_group"] = (active, keep, None)
            for name, (points, subset, fvu) in views.items():
                if not np.any(subset):
                    continue
                key = (group, name)
                if i == 0:
                    xy = points[subset]
                    if len(np.unique(label[subset])) != len(calendar):
                        raise ValueError(f"Calibration lacks a calendar name for {key}")
                    alignments[key] = orient_circle(xy[:, 0] + 1j * xy[:, 1], label[subset], len(calendar))
                metrics = angle_metrics(points[subset], label[subset], docs[subset], alignments[key], len(calendar))
                plane_fvu = float(np.square(plane_true - points).sum() / plane_variance) if fvu is not None else None
                rows.append({"split": meta["split"], "concept": concept, "representation": name,
                             "named_tokens": used, "coverage": float(subset.mean()),
                             "full_residual_fvu": fvu, "plane_fvu": plane_fvu, **metrics})
        print(f"Scored {meta['split']} on {len(named):,} named tokens", flush=True)
        del dataset, code
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rankings", nargs="+", type=Path, required=True, help="Calibration split first")
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rankings) < 2 or any("=" not in spec for spec in args.model):
        parser.error("Use at least one calibration and one test ranking, with models specified as LABEL=PATH")
    models = dict(open_model(*spec.split("=", 1)) for spec in args.model)
    if set(models) & {"raw_residual", "published_full", "published_group"} or len(models) != len(args.model):
        parser.error("Reconstruction labels must be unique")
    torch.set_num_threads(3)
    with threadpool_limits(limits=3):
        rows = report(args, args.rankings, models)
    write_csv(args.output / "reconstruction-geometry.csv", rows)
    (args.output / "reconstruction-geometry.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
