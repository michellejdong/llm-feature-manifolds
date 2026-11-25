"""Test how much of the calendar intervention is carried outside its first harmonic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import mlx.core as mx
import numpy as np
from mlx_lm import load
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import write_csv
from steer_months import (
    MODEL, MONTHS, REVISION, donor_indices, fit_plane, match_norm, output_month_ids,
    paired_interval, project, prompts, run, target_gap,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIMENSIONS = (2, 4, 8, 11)
STYLES = ("paper", "paper-after")
FIT_OFFSETS = (3, 6, 10)
TEST_OFFSETS = (2, 5, 7)
DONOR_SHIFTS = (1, 3)
RANDOM_DRAWS = 3
SEED = 752350


def month_basis(hidden: np.ndarray, labels: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Extend the original plane by Fourier contrasts between month centroids."""
    centroids = np.stack([hidden[labels == m].mean(axis=0) for m in range(len(MONTHS))])
    theta = 2 * np.pi * np.arange(len(MONTHS)) / len(MONTHS)
    designs = [column for k in range(2, 6) for column in (np.cos(k * theta), np.sin(k * theta))]
    design = np.stack(designs + [np.cos(6 * theta)], axis=1)
    combined = np.column_stack([plane, centroids.T @ design])
    basis, upper = np.linalg.qr(combined, mode="reduced")
    if np.any(np.abs(np.diag(upper)) < 1e-5):
        raise ValueError("The fitted harmonic directions are rank deficient")
    return basis


def summarize(group: list[dict]) -> dict:
    shifts = np.array([row["target_gap_change"] for row in group])
    months = np.array([MONTHS.index(row["source_month"]) for row in group])
    low, high = paired_interval(shifts, months)
    return {"n": len(group), "mean_target_gap_change": float(shifts.mean()), "ci_low": low, "ci_high": high,
            "mean_edit_norm": float(np.mean([row["edit_norm"] for row in group])),
            "target_top_vocabulary_rate": float(np.mean([row["target_top_vocabulary"] for row in group]))}


def figure(rows: list[dict], output: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(9.1, 3.8), layout="constrained", sharey=True)
    colors = {"harmonics": "#087d79", "pca": "#8c639b", "random": "#936937"}
    for ax, donor in zip(axs, DONOR_SHIFTS, strict=True):
        focused = [row for row in rows if row["donor_shift"] == donor]
        full = next(row["mean_target_gap_change"] for row in focused if row["condition"] == "full_donor")
        ax.axhline(full, color="#8290a1", linewidth=1, linestyle="--", label="Full donor")
        ax.axhline(0, color="#9ba8b5", linewidth=0.65)
        for condition, color in colors.items():
            subset = [row for row in focused if row["condition"] == condition]
            x = np.array([row["dimensions"] for row in subset])
            y = np.array([row["mean_target_gap_change"] for row in subset])
            errors = np.stack([y - [row["ci_low"] for row in subset], [row["ci_high"] for row in subset] - y])
            ax.errorbar(x, y, yerr=errors, fmt="o-", color=color, capsize=3, linewidth=1.1,
                        markersize=3.5, label={"harmonics": "Calendar harmonics", "pca": "PCA, equal norm",
                                              "random": "Random, equal norm"}[condition])
        ax.set_title(f"Donor starts +{donor} month{'s' if donor > 1 else ''} later", loc="left", fontsize=11)
        ax.set_xticks(DIMENSIONS)
        ax.set_xlabel("Intervention dimensions")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Change in target vs. original logit gap")
    axs[0].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True, help="Prespecified steering layer selection JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Choose a positive batch size")
    settings = json.loads(args.selection.read_text())
    layer, method = settings["layer_pre_block"], settings["selected_plane_fit"]
    if settings["revision"] != REVISION:
        raise ValueError("The frozen layer was selected on a different checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(512 * 1024 ** 2)
    model, _ = load(str(args.model_dir.resolve()))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    fit = prompts(tokenizer, ("paper", "from"), FIT_OFFSETS)
    evaluation = prompts(tokenizer, STYLES, TEST_OFFSETS)
    month_ids = output_month_ids(tokenizer, fit[0])
    if not all(np.array_equal(month_ids, output_month_ids(tokenizer, p)) for p in fit + evaluation):
        raise ValueError("Month continuation IDs must agree across prompts")
    fit_data = run(model, tokenizer, fit, month_ids, args.batch_size, capture=(layer,), activations_only=True)
    h_fit = fit_data["hidden"][:, 0]
    labels = np.array([p.month for p in fit])
    with threadpool_limits(limits=2):
        plane, _, fit_r2 = fit_plane(h_fit, labels, method)
        basis = month_basis(h_fit, labels, plane)
        _, _, axes = np.linalg.svd(h_fit - h_fit.mean(axis=0), full_matrices=False)
    rng = np.random.default_rng(SEED)
    randoms = [np.linalg.qr(rng.normal(size=basis.shape).astype(np.float32), mode="reduced")[0]
               for _ in range(RANDOM_DRAWS)]
    original = run(model, tokenizer, evaluation, month_ids, args.batch_size, capture=(layer,), cache_layer=layer)
    prefixes = original["prefixes"]
    hidden = original["hidden"][:, 0]
    answers = np.array([p.answer for p in evaluation])
    results, overview, exact = [], [], []
    for shift in DONOR_SHIFTS:
        donors = donor_indices(evaluation, shift)
        target = answers[donors]
        baseline_gap = target_gap(original["months"], answers, target)
        baseline_target_rate = float(np.mean(original["top_token"] == month_ids[target]))
        delta = hidden[donors] - hidden
        first = project(delta, plane)
        perpendicular = delta - first
        first_norm = np.linalg.norm(first, axis=1, keepdims=True)
        conditions = {("full_donor", hidden.shape[1]): [delta],
                      ("complement", hidden.shape[1] - 2): [perpendicular],
                      ("complement_equal_norm", hidden.shape[1] - 2): [match_norm(perpendicular, first_norm)]}
        for d in DIMENSIONS:
            signal = project(delta, basis[:, :d])
            norms = np.linalg.norm(signal, axis=1, keepdims=True)
            conditions["harmonics", d] = [signal]
            conditions["pca", d] = [match_norm(project(delta, axes[:d].T), norms)]
            conditions["random", d] = [match_norm(project(delta, matrix[:, :d]), norms) for matrix in randoms]
        for (name, d), effects in conditions.items():
            scores = [run(model, tokenizer, evaluation, month_ids, args.batch_size, patch_layer=layer,
                          effects=edit, prefixes=prefixes) for edit in effects]
            gaps = np.stack([target_gap(s["months"], answers, target) for s in scores]) - baseline_gap
            correct = np.stack([s["top_token"] == month_ids[target] for s in scores])
            target_probabilities = np.stack([np.exp(s["log_probs"][np.arange(len(target)), target]) for s in scores])
            norms = np.linalg.norm(np.stack(effects), axis=-1).mean(axis=0)
            rows = []
            for i, p in enumerate(evaluation):
                row = {"condition": name, "dimensions": d, "donor_shift": shift, "style": p.style,
                       "offset": p.steps, "source_month": MONTHS[p.month], "target": MONTHS[target[i]],
                       "target_gap_change": float(gaps[:, i].mean()), "edit_norm": float(norms[i]),
                       "target_top_vocabulary": float(correct[:, i].mean()),
                       "baseline_target_probability": float(np.exp(original["log_probs"][i, target[i]])),
                       "target_probability": float(target_probabilities[:, i].mean())}
                rows.append(row)
                if shift == 1 and p.style == "paper" and p.month == 10 and p.steps == 2:
                    exact.append(row | {"prompt": p.text})
            results.extend(rows)
            summary = {"condition": name, "dimensions": d, "donor_shift": shift,
                       "baseline_target_top_vocabulary_rate": baseline_target_rate, **summarize(rows)}
            overview.append(summary)
            print(f"Donor {shift:+}, {name} d={d}: mean target-gap change {summary['mean_target_gap_change']:.3f}", flush=True)
        write_csv(args.output / "harmonic-interventions.csv", results)
        write_csv(args.output / "harmonic-summary.csv", overview)
    metadata = {"model": MODEL, "revision": REVISION, "pre_block_layer": layer,
                "fit_method": method, "first_harmonic_fit_r2": fit_r2, "fit_offsets": FIT_OFFSETS,
                "test_styles": STYLES, "test_offsets": TEST_OFFSETS, "dimensions": DIMENSIONS,
                "random_draws_per_dimension": RANDOM_DRAWS, "exact_example": exact}
    figure(overview, args.output / "harmonic-interventions.png")
    (args.output / "harmonic-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
