"""Compare single-site month edits with repeated two-dimensional coordinate replacements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import write_csv
from steer_months import (FIT_STYLES, MODEL, MONTHS, REVISION, donor_indices, fit_plane,
                          output_month_ids, paired_interval, project, prompts, run, target_gap)


def run_path(model, entries, prefixes, start_layer, bases, donors, month_ids, batch_size, target_norms=None):
    core = model.model
    layer_ids = sorted(bases)
    columns = mx.array(month_ids)
    data = {name: [] for name in ("months", "log_probs", "top_token", "norms")}
    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        rows = mx.arange(len(batch))
        ends = mx.array([len(p.ids) - 1 for p in batch])
        positions = mx.array([p.position for p in batch])
        hidden = prefixes[start:start + len(batch)]
        mask = create_attention_mask(hidden, None)
        selector = mx.arange(hidden.shape[1])[None, :] == positions[:, None]
        actual_norms = []
        for layer, block in enumerate(core.layers[start_layer:], start=start_layer):
            if layer in bases:
                i = layer_ids.index(layer)
                basis = mx.array(bases[layer])
                donor = mx.array(donors[layer][start:start + len(batch)])
                before = hidden[rows, positions].astype(mx.float32)
                patch = ((donor - before) @ basis) @ basis.T
                norm = mx.linalg.norm(patch, axis=1, keepdims=True)
                if target_norms is not None:
                    target = mx.array(target_norms[start:start + len(batch), i:i + 1])
                    patch = mx.where(norm > 1e-7, patch / mx.maximum(norm, 1e-7), basis[:, 0][None, :]) * target
                actual_norms.append(mx.linalg.norm(patch, axis=1))
                hidden = hidden + selector[:, :, None] * patch[:, None, :].astype(hidden.dtype)
            hidden = block(hidden, mask, cache=None)
        logits = model.lm_head(core.norm(hidden[rows, ends])).astype(mx.float32)
        restricted = logits[:, columns]
        outputs = {"months": restricted, "log_probs": restricted - mx.logsumexp(logits, axis=1, keepdims=True),
                   "top_token": mx.argmax(logits, axis=1), "norms": mx.stack(actual_norms, axis=1)}
        mx.eval(*outputs.values())
        for name, value in outputs.items():
            data[name].append(np.asarray(value))
    return {name: np.concatenate(values) for name, values in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path,
                        help="mistral-example.json from the earlier layer sweep")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--random-subspaces", type=int, default=6)
    args = parser.parse_args()
    if args.batch_size < 1 or args.random_subspaces < 2:
        parser.error("Use a positive batch size and at least two random controls")
    previous = json.loads(args.calibration.read_text())
    if previous["revision"] != REVISION or not previous["compact_fit"]:
        raise ValueError("Use the compact calibration from the same Mistral checkpoint")
    selected = sorted(previous["layer_calibration"], key=lambda row: -row["mean_logit_shift"])
    methods = {}
    for row in selected:
        if len(methods) == 3:
            break
        methods.setdefault(row["layer"], row["fit_method"])
    if len(methods) < 3:
        raise ValueError("At least three calibrated layers are required")
    layer_ids = tuple(sorted(methods))
    first_layer = layer_ids[0]
    single = previous["layer_pre_block"]
    args.output.mkdir(parents=True, exist_ok=True)

    mx.set_cache_limit(512 * 1024 ** 2)
    print(f"Loading Mistral to test layers {layer_ids}, with single-layer baseline {single}", flush=True)
    model, _ = load(str(args.model_dir.resolve()))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    fit = prompts(tokenizer, FIT_STYLES, (3, 6, 10))
    tests = prompts(tokenizer, ("paper",), (1, 2, 8, 11))
    ids = output_month_ids(tokenizer, tests[0])
    if not all(np.array_equal(ids, output_month_ids(tokenizer, prompt)) for prompt in tests):
        raise ValueError("Expected identical month token IDs across prompts")
    fit_data = run(model, tokenizer, fit, ids, args.batch_size, capture=layer_ids, activations_only=True)["hidden"]
    labels = np.array([p.month for p in fit])
    with threadpool_limits(limits=2):
        bases = {layer: fit_plane(fit_data[:, i], labels, methods[layer])[0]
                 for i, layer in enumerate(layer_ids)}
    base = run(model, tokenizer, tests, ids, args.batch_size, capture=layer_ids, cache_layer=first_layer)
    prefixes = base.pop("prefixes")
    target_rows = donor_indices(tests, 1)
    donors = {layer: base["hidden"][target_rows, i] for i, layer in enumerate(layer_ids)}
    original = np.array([p.answer for p in tests])
    target = original[target_rows]
    month = np.array([p.month for p in tests])
    baseline_gap = target_gap(base["months"], original, target)
    print(f"Baseline correct within months: {np.mean(base['months'].argmax(axis=1) == original):.3f}", flush=True)

    no_op = run_path(model, tests[:1], prefixes[:1], first_layer, {single: bases[single]},
                     donors, ids, 1, target_norms=np.zeros((1, 1), dtype=np.float32))
    if np.max(np.abs(no_op["months"] - base["months"][:1])) > 0.075:
        raise ValueError("The cached zero-edit forward changed baseline logits")
    multiple = run_path(model, tests, prefixes, first_layer, bases, donors, ids, args.batch_size)
    single_base = {single: bases[single]}
    one = run_path(model, tests, prefixes, first_layer, single_base, donors, ids, args.batch_size)
    strength = np.linalg.norm(multiple["norms"], axis=1, keepdims=True)
    matched_one = run_path(model, tests, prefixes, first_layer, single_base, donors, ids, args.batch_size,
                           target_norms=strength)
    print("Finished calendar edits at one and three layers", flush=True)
    rng = np.random.default_rng(39192)
    random_results = []
    for _ in range(args.random_subspaces):
        random_bases = {}
        for layer, plane in bases.items():
            draw = rng.normal(size=plane.shape).astype(np.float32)
            random_bases[layer] = np.linalg.qr(draw - project(draw.T, plane).T, mode="reduced")[0]
        outcome = run_path(model, tests, prefixes, first_layer, random_bases, donors, ids, args.batch_size,
                           target_norms=multiple["norms"])
        if not np.allclose(outcome["norms"], multiple["norms"], atol=0.005, rtol=0.001):
            raise ValueError("Random intervention did not match the three layer-wise norms")
        random_results.append(outcome)

    choices = {"single": one, "norm_matched_single": matched_one, "three_layers": multiple}
    changes = {name: target_gap(out["months"], original, target) - baseline_gap for name, out in choices.items()}
    changes["norm_matched_random"] = np.stack([target_gap(out["months"], original, target) - baseline_gap
                                               for out in random_results]).mean(axis=0)
    i = next(i for i, p in enumerate(tests) if p.month == 10 and p.steps == 2)
    pair = [original[i], target[i]]
    exact = {"prompt": tests[i].text, "original_answer": MONTHS[pair[0]], "target_answer": MONTHS[pair[1]],
             "probabilities": {"baseline": np.exp(base["log_probs"][i, pair]).tolist(), **{
                 name: np.exp(out["log_probs"][i, pair]).tolist() for name, out in choices.items()},
                 "norm_matched_random": np.stack([np.exp(out["log_probs"][i, pair]) for out in random_results]).mean(axis=0).tolist()},
             "three_layer_norms": multiple["norms"][i].tolist()}
    rows = []
    for j, p in enumerate(tests):
        rows.append({"month": MONTHS[p.month], "offset": p.steps,
                     "original_answer": MONTHS[original[j]], "target_answer": MONTHS[target[j]],
                     **{name: float(shifts[j]) for name, shifts in changes.items()}})
    summary = {"model": MODEL, "revision": REVISION, "layers": layer_ids,
               "anchor_layer": single, "fit_method_by_layer": methods, "fit_offsets": [3, 6, 10],
               "test_offsets": [1, 2, 8, 11], "examples": len(tests), "random_subspaces": len(random_results),
               "baseline_top_month_accuracy": float(np.mean(base["months"].argmax(axis=1) == original)),
               "target_top_month_rate": {name: float(np.mean(out["months"].argmax(axis=1) == target))
                                         for name, out in choices.items()},
               "logit_gap_changes": {name: {"mean": float(values.mean()), "ci95": paired_interval(values, month)}
                                      for name, values in changes.items()},
               "three_layer_vs_norm_matched_single_ci95": paired_interval(changes["three_layers"] - changes["norm_matched_single"], month),
               "three_layer_vs_random_ci95": paired_interval(changes["three_layers"] - changes["norm_matched_random"], month),
               "exact": exact}
    write_csv(args.output / "multi-layer-pairs.csv", rows)
    (args.output / "multi-layer-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
