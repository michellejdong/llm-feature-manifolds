"""Read an answer-position calendar plane as a schedule changes over several sentences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import mlx.core as mx
import numpy as np
from mlx_lm import load
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import write_csv
from steer_months import MODEL, MONTHS, NUMBERS, REVISION, Prompt, output_month_ids, run

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAYERS = (12, 16, 20, 24, 28, 30)
FIT_OFFSETS = (1, 3, -2)
CALIBRATION_OFFSETS = (5, -4)
MOVES = ((2, -1, 4, -2), (-3, 5, -4, 1))
CONDITIONS = ("offsets_only", "stated_history")
HEADER = ("Let's do some calendar math.\n"
          "Start in January. Move forward six months. The current month is July.\n"
          "Start in July. Move back two months. The current month is May.\n"
          "Start in April. Move forward three months. Move back one month. The current month is June.\n")


def question(tokenizer, start: int, deltas: tuple[int, ...], condition: str, course: int) -> tuple[Prompt, dict]:
    text = HEADER + f"Start in {MONTHS[start]}."
    current = start
    for i, delta in enumerate(deltas):
        count = NUMBERS[abs(delta)].lower()
        unit = "month" if abs(delta) == 1 else "months"
        direction = "forward" if delta > 0 else "back"
        text += f" Move {direction} {count} {unit}."
        current = (current + delta) % len(MONTHS)
        if condition == "stated_history" and i + 1 < len(deltas):
            text += f" We are now in {MONTHS[current]}."
    text += " The current month is"
    ids = tokenizer.encode(text)
    prompt = Prompt(condition, start, sum(deltas), text, ids, len(ids) - 1)
    return prompt, {"condition": condition, "course": course, "depth": len(deltas),
                    "initial": start, "target": current}


def phase_probe(hidden: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = hidden.mean(axis=0)
    centered = hidden - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projection = axes[:12].T
    design = centered @ projection
    phase = np.exp(2j * np.pi * months / len(MONTHS))
    target = np.c_[phase.real, phase.imag]
    covariance = design.T @ design
    penalty = 0.05 * np.trace(covariance) / len(covariance)
    coefficients = np.linalg.solve(covariance + penalty * np.eye(len(covariance)), design.T @ target)
    return center, projection, coefficients


def decode(hidden: np.ndarray, probe: tuple) -> np.ndarray:
    center, projection, coefficients = probe
    xy = ((hidden - center) @ projection) @ coefficients
    return xy[:, 0] + 1j * xy[:, 1]


def angles_to_months(phase: np.ndarray) -> np.ndarray:
    return np.round(np.angle(phase) * len(MONTHS) / (2 * np.pi)).astype(int) % len(MONTHS)


def draw_trajectory(rows: list[dict], path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.7, 4.4), layout="constrained")
    theta = 2 * np.pi * np.arange(12) / 12
    ring = np.exp(1j * theta)
    coarse = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    color = "#0d766e"
    for ax, condition in zip(axs, CONDITIONS, strict=True):
        points = [row for row in rows if row["condition"] == condition and row["initial"] == "November" and row["course"] == 0]
        points.sort(key=lambda row: row["depth"])
        expected = np.array([ring[MONTHS.index(row["target"])] for row in points])
        predicted = np.exp(1j * np.array([row["probe_angle_radians"] for row in points]))
        ax.plot(coarse.real, coarse.imag, color="#bac4cf", lw=0.85)
        ax.scatter(ring.real, ring.imag, color="#9ba8b4", s=8)
        for i, label in enumerate(MONTHS):
            z = 1.17 * ring[i]
            ax.text(z.real, z.imag, label[:3], va="center", ha="center", fontsize=7, color="#475569")
        ax.plot(expected.real, expected.imag, "o--", color="#727b86", label="Correct trajectory", lw=1.0, markersize=3)
        ax.plot(predicted.real, predicted.imag, "o-", color=color, label="Activation readout", lw=1.55, markersize=4)
        for j, z in enumerate(predicted, 1):
            ax.annotate(str(j), (z.real, z.imag), xytext=(8 if j % 2 else -9, -8 if j % 2 else 5), textcoords="offset points",
                        color=color, ha="center", fontsize=7, weight="semibold")
        title = "Offsets only" if condition == "offsets_only" else "With earlier months stated"
        ax.set_title(title, fontsize=11, weight="semibold")
        ax.set_aspect("equal")
        ax.set_ylim(-1.29, 1.29)
        ax.set_xlim(-1.29, 1.29)
        ax.axis("off")
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8)
    fig.suptitle("A calendar trajectory beginning in November", fontsize=14, weight="semibold")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(512 * 1024 ** 2)
    print("Loading Mistral for calendar trajectories", flush=True)
    model, _ = load(str(args.model_dir.resolve()))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    fit = [question(tokenizer, m, (offset,), "offsets_only", 0) for offset in FIT_OFFSETS for m in range(12)]
    calibration = [question(tokenizer, m, (offset,), "offsets_only", 0) for offset in CALIBRATION_OFFSETS for m in range(12)]
    tests = [question(tokenizer, m, deltas[:depth], condition, k)
             for condition in CONDITIONS for k, deltas in enumerate(MOVES)
             for m in range(12) for depth in range(1, len(deltas) + 1)]
    ids = output_month_ids(tokenizer, fit[0][0])
    if not all(np.array_equal(ids, output_month_ids(tokenizer, prompt)) for prompt, _ in fit + calibration + tests):
        raise ValueError("Vocab IDs must agree across contexts")
    fit_features = run(model, tokenizer, [p for p, _ in fit], ids, args.batch_size, capture=LAYERS)["hidden"]
    cal_features = run(model, tokenizer, [p for p, _ in calibration], ids, args.batch_size, capture=LAYERS)["hidden"]
    fit_labels = np.array([label["target"] for _, label in fit])
    calibration_truth = np.array([label["target"] for _, label in calibration])
    probes = {}
    with threadpool_limits(limits=2):
        for i, layer in enumerate(LAYERS):
            probe = phase_probe(fit_features[:, i], fit_labels)
            phase = decode(cal_features[:, i], probe)
            residual = np.angle(phase) - 2 * np.pi * calibration_truth / 12
            probes[layer] = (float(np.cos(residual).mean()), probe)
    layer = max(probes, key=lambda key: probes[key][0])
    print(f"Selected answer-position layer {layer} using single-step calibration", flush=True)
    behavior = run(model, tokenizer, [p for p, _ in tests], ids, args.batch_size, capture=(layer,))
    z = decode(behavior["hidden"][:, 0], probes[layer][1])
    predicted = angles_to_months(z)
    model_months = behavior["months"].argmax(axis=1)
    logits_phase = softmax(behavior["months"], axis=1) @ np.exp(2j * np.pi * np.arange(12) / 12)
    rows = []
    for i, (_, item) in enumerate(tests):
        rows.append({"condition": item["condition"], "course": item["course"], "depth": item["depth"],
                     "initial": MONTHS[item["initial"]], "target": MONTHS[item["target"]],
                     "model_top_month": MONTHS[model_months[i]], "probe_month": MONTHS[predicted[i]],
                     "top_vocab_correct": bool(behavior["top_token"][i] == ids[item["target"]]),
                     "probe_angle_radians": float(np.angle(z[i])), "behavior_angle_radians": float(np.angle(logits_phase[i])),
                     "behavior_phase_strength": float(abs(logits_phase[i]))})
    summary = []
    for condition in CONDITIONS:
        for depth in range(1, 5):
            selected = [row for row in rows if row["condition"] == condition and row["depth"] == depth]
            summary.append({"condition": condition, "depth": depth, "n": len(selected),
                            "model_accuracy": float(np.mean([row["model_top_month"] == row["target"] for row in selected])),
                            "probe_accuracy": float(np.mean([row["probe_month"] == row["target"] for row in selected])),
                            "probe_model_agreement": float(np.mean([row["probe_month"] == row["model_top_month"] for row in selected]))})
    write_csv(args.output / "calendar-trajectories.csv", rows)
    write_csv(args.output / "calendar-trajectory-summary.csv", summary)
    (args.output / "calendar-trajectory-metadata.json").write_text(json.dumps(
        {"model": MODEL, "revision": REVISION, "pre_block_layer": layer,
         "fit_offsets": FIT_OFFSETS, "calibration_offsets": CALIBRATION_OFFSETS, "test_movement_sequences": MOVES,
         "calibration_circular_scores": {k: v[0] for k, v in probes.items()}}, indent=2,
    ) + "\n")
    draw_trajectory(rows, args.output / "calendar-trajectory.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
