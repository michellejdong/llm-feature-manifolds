"""Test the effect of calendar prompt framing and norm-matched activation edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import write_csv
from steer_months import (
    FIT_STYLES, MODEL, MONTHS, REVISION, donor_indices, fit_plane, match_norm,
    output_month_ids, paired_interval, project, prompts, run, target_gap,
)


FRAMES = ("after", "paper", "paper-after", "examples")


def run_edits(model, tokenizer, prompts_to_edit, month_ids, batch_size, layer, prefixes, edits):
    ids = list(edits)
    n = len(prompts_to_edit)
    outcomes = run(model, tokenizer, prompts_to_edit * len(ids), month_ids, batch_size,
                   patch_layer=layer, effects=np.concatenate(list(edits.values())),
                   prefixes=mx.concatenate([prefixes] * len(ids)))
    return {name: {key: value[i * n:(i + 1) * n] for key, value in outcomes.items()}
            for i, name in enumerate(ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--random-subspaces", type=int, default=6)
    parser.add_argument("--exact-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.layer < 32 or args.batch_size < 1 or args.random_subspaces < 2:
        parser.error("Provide a valid layer, a positive batch size, and at least two random subspaces")
    args.output.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(512 * 1024 ** 2)
    print("Loading Mistral for the framing check", flush=True)
    model, _ = load(str(args.model_dir.resolve()))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    grid = prompts(tokenizer, FRAMES, (2, 5))
    fit = prompts(tokenizer, FIT_STYLES, (3, 6, 10))
    month_ids = output_month_ids(tokenizer, grid[0])
    if not all(np.array_equal(month_ids, output_month_ids(tokenizer, p)) for p in grid):
        raise ValueError("Month tokens changed across frames")
    chosen = (args.layer,)
    h = run(model, tokenizer, fit, month_ids, args.batch_size, capture=chosen, activations_only=True)["hidden"][:, 0]
    with threadpool_limits(limits=2):
        plane, _, r2 = fit_plane(h, np.array([p.month for p in fit]))
    background = h.mean(axis=0)
    print(f"Fitted month plane at pre-block layer {args.layer}, R² {r2:.3f}", flush=True)
    rng = np.random.default_rng(1771)
    controls = []
    for _ in range(args.random_subspaces):
        random = rng.normal(size=plane.shape).astype(np.float32)
        controls.append(np.linalg.qr(random - project(random.T, plane).T, mode="reduced")[0])

    # Save the core comparison before running the larger evaluation grid.
    exact_prompts = [p for p in grid if p.steps == 2 and p.month in {10, 11}]
    exact_base = run(model, tokenizer, exact_prompts, month_ids, args.batch_size,
                     capture=chosen, cache_layer=args.layer)
    exact_rows = np.array([i for i, p in enumerate(exact_prompts) if p.month == 10])
    source, donor = exact_base["hidden"][exact_rows, 0], exact_base["hidden"][exact_rows + 1, 0]
    projected = project(donor - source, plane)
    norm = np.linalg.norm(projected, axis=1, keepdims=True)
    edits = {"plane": projected}
    edits.update({f"random_{i}": match_norm(project(donor - source, basis), norm)
                  for i, basis in enumerate(controls)})
    edits["mean_background_source"] = background + project(source - background, plane) - source
    edits["mean_background_donor"] = background + project(donor - background, plane) - source
    edits["whole_donor"] = donor - source
    edited = run_edits(model, tokenizer, [exact_prompts[i] for i in exact_rows], month_ids,
                       args.batch_size, args.layer, exact_base["prefixes"][mx.array(exact_rows)], edits)
    exact = []
    for f, j in enumerate(exact_rows):
        baseline = {key: exact_base[key][j] for key in ("months", "log_probs", "top_token")}
        for name, answer in [("baseline", baseline), *[(k, {m: v[f] for m, v in sample.items()})
                                                        for k, sample in edited.items()]]:
            exact.append({"frame": exact_prompts[j].style, "prompt": exact_prompts[j].text, "edit": name,
                          "argmax_token": tokenizer.decode(int(answer["top_token"])),
                          "top_month": MONTHS[answer["months"].argmax()],
                          "january_probability": float(np.exp(answer["log_probs"][0])),
                          "february_probability": float(np.exp(answer["log_probs"][1])),
                          "february_minus_january_logit": float(answer["months"][1] - answer["months"][0])})
    write_csv(args.output / "framing-exact.csv", exact)
    print("Saved the exact prompt in four frames", flush=True)
    if args.exact_only:
        return

    base = run(model, tokenizer, grid, month_ids, args.batch_size, capture=chosen, cache_layer=args.layer)
    answer = np.array([p.answer for p in grid])
    donor = donor_indices(grid, 1)
    target = answer[donor]
    baseline = {}
    for frame in FRAMES:
        rows = np.array([p.style == frame for p in grid])
        baseline[frame] = {"n": int(rows.sum()),
                           "top_month_accuracy": float(np.mean(base["months"][rows].argmax(axis=1) == answer[rows])),
                           "top_token_accuracy": float(np.mean(base["top_token"][rows] == month_ids[answer[rows]]))}
    info = {"model": MODEL, "revision": REVISION, "layer_pre_block": args.layer, "fit_probe_r2": r2,
            "fit_offsets": [3, 6, 10], "test_offsets": [2, 5], "baseline": baseline,
            "random_subspaces": args.random_subspaces}
    (args.output / "framing-summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Baseline accuracies: {baseline}", flush=True)

    # The two frames below are fixed before seeing the held-out accuracies.
    subset = np.array([i for i, p in enumerate(grid) if p.style in {"paper", "paper-after"}])
    intervention_prompts = [grid[i] for i in subset]
    delta = base["hidden"][donor[subset], 0] - base["hidden"][subset, 0]
    projected = project(delta, plane)
    norm = np.linalg.norm(projected, axis=1, keepdims=True)
    prefixes = base.pop("prefixes")[mx.array(subset)]
    restricted = base["months"][subset]
    base_gap = target_gap(restricted, answer[subset], target[subset])
    plane_run = run_edits(model, tokenizer, intervention_prompts, month_ids, args.batch_size, args.layer,
                         prefixes, {"plane": projected, "whole_donor": delta})
    print("Finished the plane and full-donor edits", flush=True)
    random_run = run_edits(model, tokenizer, intervention_prompts, month_ids, args.batch_size, args.layer,
                           prefixes, {str(i): match_norm(project(delta, b), norm) for i, b in enumerate(controls)})
    shift = target_gap(plane_run["plane"]["months"], answer[subset], target[subset]) - base_gap
    full = target_gap(plane_run["whole_donor"]["months"], answer[subset], target[subset]) - base_gap
    random = np.stack([target_gap(out["months"], answer[subset], target[subset]) - base_gap
                       for out in random_run.values()]).mean(axis=0)
    month = np.array([p.month for p in intervention_prompts])
    info["steering"] = {}
    rows = []
    for frame in ("paper", "paper-after"):
        use = np.array([p.style == frame for p in intervention_prompts])
        change = shift[use] - random[use]
        info["steering"][frame] = {"n": int(use.sum()), "plane_mean_logit_shift": float(shift[use].mean()),
                                   "random_mean_logit_shift": float(random[use].mean()),
                                   "full_donor_mean_logit_shift": float(full[use].mean()),
                                   "plane_minus_random_ci95": paired_interval(change, month[use]),
                                   "target_top_month_rate": float(np.mean(
                                       plane_run["plane"]["months"][use].argmax(axis=1) == target[subset][use]))}
    for i, p in enumerate(intervention_prompts):
        rows.append({"frame": p.style, "start_month": MONTHS[p.month], "offset": p.steps,
                     "baseline_correct_month": bool(restricted[i].argmax() == answer[subset[i]]),
                     "plane_logit_shift": float(shift[i]), "random_logit_shift": float(random[i]),
                     "whole_donor_logit_shift": float(full[i])})
    write_csv(args.output / "framing-pairs.csv", rows)
    (args.output / "framing-summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
