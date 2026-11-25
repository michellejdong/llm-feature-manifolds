"""Patch a 2D month subspace in quantized Mistral, with norm-matched controls."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import write_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL = "mlx-community/Mistral-7B-v0.1-q"
REVISION = "1b5b8ebdb79dcba7d6a8fc002619aaa3043c9f34"
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
NUMBERS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
           "Nine", "Ten", "Eleven", "Twelve")
LAYERS = (2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30)
FIT_STYLES = ("paper", "from")
TEST_STYLES = ("after", "later")
FIT_STEPS = (1, 3, 4, 6, 8, 10, 12)
CALIBRATION_STEPS = (2, 5, 7, 9, 11)


@dataclass(frozen=True)
class Prompt:
    style: str
    month: int
    steps: int
    text: str
    ids: list[int]
    position: int

    @property
    def answer(self) -> int:
        return (self.month + self.steps) % len(MONTHS)


def prompts(tokenizer, styles: tuple[str, ...], steps: tuple[int, ...]) -> list[Prompt]:
    entries = []
    for style in styles:
        for n in steps:
            for m, month in enumerate(MONTHS):
                number = NUMBERS[n]
                unit = "month" if n == 1 else "months"
                prefix = ". " if n not in (9, 11, 12) else ""
                texts = {
                    "paper": f"{prefix}Let's do some calendar math. {number} months from {month} is",
                    "from": f"{number} {unit} from {month} is",
                    "calendar-exercise": f"Calendar exercise: {number.lower()} {unit} after {month} is",
                    "after": f"{number} {unit} after {month} is",
                    "later": f"Starting in {month}, {number.lower()} {unit} later it will be",
                    "paper-after": f"Let's do some calendar math. {number} {unit} after {month} is",
                    "examples": ("Two months after April is June. Three months after July is October. "
                                 f"{number} {unit} after {month} is"),
                }
                text = texts[style]
                tokenization = tokenizer(text, return_offsets_mapping=True)
                month_start = text.rindex(month)
                positions = [i for i, (a, b) in enumerate(tokenization["offset_mapping"])
                             if b > month_start and a < month_start + len(month)]
                if len(positions) != 1:
                    raise ValueError(f"Expected a single input month token: {text!r} at {positions}")
                entries.append(Prompt(style, m, n, text, tokenization["input_ids"], positions[0]))
    return entries


def output_month_ids(tokenizer, prompt: Prompt) -> np.ndarray:
    ids = []
    for month in MONTHS:
        continuation = tokenizer.encode(f"{prompt.text} {month}")
        if len(continuation) != len(prompt.ids) + 1 or continuation[:-1] != prompt.ids:
            raise ValueError(f"Expected a single continuation token for {month} in {prompt.text!r}")
        ids.append(continuation[-1])
    return np.asarray(ids, dtype=np.int32)


def forward(model, ids, ends, positions, capture, patch_layer, effects, cache_layer=None, prefix=None,
            activations_only=False):
    core = model.model
    hidden = core.embed_tokens(ids) if prefix is None else prefix
    first_layer = 0 if prefix is None else patch_layer
    mask = create_attention_mask(hidden, None)
    batch = mx.arange(len(ids))
    activations = []
    cached = None
    for i, block in enumerate(core.layers[first_layer:], start=first_layer):
        if i == cache_layer:
            cached = hidden
        if i in capture:
            activations.append(hidden[batch, positions].astype(mx.float32))
            if activations_only and i == capture[-1]:
                return None, activations, cached
        if i == patch_layer:
            locations = mx.arange(ids.shape[1])[None, :] == positions[:, None]
            hidden = hidden + locations[:, :, None] * effects[:, None, :].astype(hidden.dtype)
        hidden = block(hidden, mask, cache=None)
    return model.lm_head(core.norm(hidden[batch, ends])).astype(mx.float32), activations, cached


def run(model, tokenizer, entries, month_ids, batch_size, capture=(), patch_layer=None, effects=None,
        cache_layer=None, prefixes=None, activations_only=False):
    month_columns = mx.array(month_ids)
    results = {key: [] for key in ("months", "log_probs", "top_token", "hidden")}
    cache = []
    lengths = [p.position + 1 if activations_only else len(p.ids) for p in entries]
    length = max(lengths) if prefixes is None else prefixes.shape[1]
    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        batch_lengths = lengths[start:start + batch_size]
        ids = mx.array([p.ids[:n] + [tokenizer.eos_token_id] * (length - n)
                        for p, n in zip(batch, batch_lengths, strict=True)])
        ends = mx.array([n - 1 for n in batch_lengths])
        positions = mx.array([p.position for p in batch])
        edit = mx.array(effects[start:start + batch_size]) if effects is not None else None
        prefix = prefixes[start:start + batch_size] if prefixes is not None else None
        logits, activations, saved = forward(model, ids, ends, positions, capture, patch_layer, edit,
                                             cache_layer, prefix, activations_only)
        outputs = {}
        if logits is not None:
            restricted = logits[:, month_columns]
            outputs = {"months": restricted, "log_probs": restricted - mx.logsumexp(logits, axis=1, keepdims=True),
                       "top_token": mx.argmax(logits, axis=1)}
        if activations:
            outputs["hidden"] = mx.stack(activations, axis=1)
        mx.eval(*outputs.values())
        if saved is not None:
            cache.append(saved)
        for name, output in outputs.items():
            results[name].append(np.asarray(output))
    outcome = {name: np.concatenate(values) for name, values in results.items() if values}
    if cache:
        outcome["prefixes"] = mx.concatenate(cache)
    return outcome


def fit_plane(activations: np.ndarray, labels: np.ndarray, method: str = "all-rows") -> tuple[np.ndarray, np.ndarray, float]:
    if method == "centroids":
        activations = np.stack([activations[labels == i].mean(axis=0) for i in range(len(MONTHS))])
        labels = np.arange(len(MONTHS))
    centered = activations - activations.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    top = axes[:5].T
    angles = 2 * np.pi * labels / len(MONTHS)
    targets = np.c_[np.cos(angles), np.sin(angles)]
    reduced = centered @ top
    coefficients = np.linalg.lstsq(reduced, targets, rcond=None)[0]
    basis, _ = np.linalg.qr(top @ coefficients, mode="reduced")
    r2 = 1 - np.square(reduced @ coefficients - targets).sum() / np.square(targets).sum()
    return basis, top[:, :1], float(r2)


def donor_indices(entries: list[Prompt], shift: int) -> np.ndarray:
    lookup = {(p.style, p.month, p.steps): i for i, p in enumerate(entries)}
    return np.array([lookup[p.style, (p.month + shift) % len(MONTHS), p.steps] for p in entries])


def project(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (values @ basis) @ basis.T


def match_norm(values: np.ndarray, goal: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any((denominator < 1e-7) & (goal > 1e-7)):
        raise ValueError("An intervention control has no direction to normalize")
    return values * (goal / np.maximum(denominator, 1e-7))


def target_gap(logits: np.ndarray, original: np.ndarray, target: np.ndarray) -> np.ndarray:
    indices = np.arange(len(original))
    return logits[indices, target] - logits[indices, original]


def baseline_metrics(entries, scores, month_ids) -> dict:
    answers = np.array([p.answer for p in entries])
    return {
        "n": len(entries),
        "top_month_accuracy": float(np.mean(scores["months"].argmax(axis=1) == answers)),
        "top_vocabulary_accuracy": float(np.mean(scores["top_token"] == month_ids[answers])),
    }


def paired_interval(values: np.ndarray, month_ids: np.ndarray) -> list[float]:
    estimates = np.array([values[month_ids == i].mean() for i in range(len(MONTHS)) if i in month_ids])
    rng = np.random.default_rng(762)
    draws = rng.choice(estimates, size=(4_000, len(estimates)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def compare_paths(model, tokenizer, entries, month_ids, batch_size, layer, plane, fit, fit_labels, test):
    """Compare a fitted ellipse with its chord at intermediate month shifts."""
    fit_centroids = np.stack([fit[fit_labels == m].mean(axis=0) for m in range(12)]) @ plane
    origin = fit_centroids.mean(axis=0)
    _, sigma, eigenvectors = np.linalg.svd(fit_centroids - origin, full_matrices=False)
    stretch = np.maximum(sigma, 1e-6)
    to_circle = eigenvectors.T / stretch
    from_circle = stretch[:, None] * eigenvectors
    take = np.array([i for i, p in enumerate(entries) if p.steps in {2, 5, 7}])
    subset = [entries[i] for i in take]
    source_answer = np.array([p.answer for p in subset])
    case_month = np.array([p.month for p in subset])
    start = (test["hidden"][take, 0] @ plane - origin) @ to_circle
    dest = (test["hidden"][donor_indices(entries, 3)[take], 0] @ plane - origin) @ to_circle
    z0, z1 = start[:, 0] + 1j * start[:, 1], dest[:, 0] + 1j * dest[:, 1]
    theta = np.angle(z1 * np.conj(z0))

    rows = []
    for moved in (1, 2):
        alpha = moved / 3
        target = (source_answer + moved) % 12
        z = ((1 - alpha) * np.abs(z0) + alpha * np.abs(z1)) * np.exp(1j * (np.angle(z0) + alpha * theta))
        circular = (np.c_[z.real, z.imag] - start) @ from_circle @ plane.T
        direct = (alpha * (dest - start)) @ from_circle @ plane.T
        paths = {
            "ellipse": circular,
            "chord": direct,
            "norm_matched_chord": match_norm(direct, np.linalg.norm(circular, axis=1, keepdims=True)),
        }
        baseline = target_gap(test["months"][take], source_answer, target)
        for name, edits in paths.items():
            intervened = run(model, tokenizer, subset, month_ids, batch_size, patch_layer=layer, effects=edits)
            improvement = target_gap(intervened["months"], source_answer, target) - baseline
            rows.append({"intermediate_months": moved, "path": name, "n": len(subset),
                         "mean_target_gap_change": float(improvement.mean()),
                         "ci95": paired_interval(improvement, case_month),
                         "top_month_target_rate": float(np.mean(intervened["months"].argmax(axis=1) == target))})
    return rows


def exact_example(model, tokenizer, entries, month_ids, layer, plane, random_bases,
                  background=None, style="after") -> dict:
    case = next(p for p in entries if p.style == style and p.month == 10 and p.steps == 2)
    donor = next(p for p in entries if p.style == case.style and p.month == 11 and p.steps == 2)
    base = run(model, tokenizer, [case, donor], month_ids, 2, capture=(layer,))
    delta = base["hidden"][1:, 0] - base["hidden"][:1, 0]
    edit = project(delta, plane)
    norms = np.linalg.norm(edit, axis=1, keepdims=True)
    all_edits = np.concatenate([edit] + [match_norm(project(delta, b), norms) for b in random_bases])
    if background is not None:
        source, target = base["hidden"][:, 0]
        target_states = np.stack([background + project(x - background, plane) for x in (source, target)])
        all_edits = np.r_[all_edits, target_states - source, delta]
    patched = run(model, tokenizer, [case] * len(all_edits), month_ids, len(all_edits), patch_layer=layer, effects=all_edits)
    columns = [case.answer, donor.answer]
    base_gap = base["months"][0, columns[1]] - base["months"][0, columns[0]]
    shifts = patched["months"][:, columns[1]] - patched["months"][:, columns[0]] - base_gap
    random_slice = slice(1, len(random_bases) + 1)
    result = {"prompt": case.text, "original_answer": MONTHS[case.answer], "target_answer": MONTHS[donor.answer],
            "probabilities": {"baseline": np.exp(base["log_probs"][0, columns]).tolist(),
                              "circle": np.exp(patched["log_probs"][0, columns]).tolist(),
                              "random_mean": np.exp(patched["log_probs"][random_slice][:, columns]).mean(axis=0).tolist()},
            "original_top_token": tokenizer.decode(int(base["top_token"][0])),
            "patched_top_token": tokenizer.decode(int(patched["top_token"][0])),
            "circle_logit_shift": float(shifts[0]), "random_logit_shift_mean": float(shifts[random_slice].mean())}
    if background is not None:
        result["background_ablation_diagnostic"] = {
            "same_month_probabilities": np.exp(patched["log_probs"][-3, columns]).tolist(),
            "target_month_probabilities": np.exp(patched["log_probs"][-2, columns]).tolist(),
            "target_vs_same_month_logit_shift": float(shifts[-2] - shifts[-3]),
            "full_donor_probabilities": np.exp(patched["log_probs"][-1, columns]).tolist(),
            "full_donor_logit_shift": float(shifts[-1]),
        }
    return result


def plot_summary(output: Path, exact: dict, aggregate: dict) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    colors = ("#7891a8", "#0d766e", "#a37539")
    labels = ("Baseline", "Calendar plane", "Random subspaces")
    for i, (name, values) in enumerate(exact["probabilities"].items()):
        left.bar(np.arange(2) + 0.23 * (i - 1), values, width=0.2, color=colors[i], label=labels[i])
    left.set_xticks([0, 1], [exact["original_answer"], exact["target_answer"]])
    left.set_ylabel("Probability over full vocabulary")
    left.set_title("November → December, +2 months", loc="left", fontsize=11)
    left.legend(loc="best", frameon=False, fontsize=8)
    xy = [(item["mean"], item["ci95"]) for item in aggregate.values()]
    right.axvline(0, color="#828b99", lw=0.9)
    ys = np.arange(len(xy))
    right.set_yticks(ys, ["2D calendar plane", "Random 2D subspaces", "Top 1D PC"])
    for i, (mean, (lo, hi)) in enumerate(xy):
        right.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", capsize=4,
                       color=("#0d766e", "#a37539", "#64748b")[i])
    right.invert_yaxis()
    right.set_xlabel("Change in target vs. original answer logit gap")
    right.set_title("Held-out prompts · 95% bootstrap intervals", loc="left", fontsize=11)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18, linewidth=0.6)
    fig.savefig(output / "mistral-steering.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--random-subspaces", type=int, default=8)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--compare-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exact-only", action="store_true", help="Stop after the held-out November-to-December example")
    parser.add_argument("--compact-fit", action="store_true", help="Use a reduced, prespecified fit and calibration grid")
    parser.add_argument("--fixed-layer", type=int, help="Skip calibration and evaluate an independently selected layer")
    parser.add_argument("--candidate-layers", type=int, nargs="+")
    parser.add_argument("--plane-fit", choices=("all-rows", "centroids"), nargs="+", default=["all-rows"])
    parser.add_argument("--calibration-style", choices=("calendar-exercise", "paper", "paper-after"), default="calendar-exercise")
    parser.add_argument("--calibration-offsets", type=int, nargs="+")
    parser.add_argument("--test-styles", choices=("after", "later", "paper", "paper-after", "examples"),
                        nargs="+", default=list(TEST_STYLES))
    parser.add_argument("--example-style", default="after")
    parser.add_argument("--test-offsets", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--donor-shifts", type=int, nargs="+", default=[1, -1, 3, -3])
    args = parser.parse_args()
    if args.random_subspaces < 2 or args.batch_size < 1:
        parser.error("Use at least two random subspaces and a positive batch size")
    if args.fixed_layer is not None and not (0 <= args.fixed_layer < 32):
        parser.error("The fixed layer must be between 0 and 31")
    if args.candidate_layers and any(not 0 <= layer < 32 for layer in args.candidate_layers):
        parser.error("Candidate layers must be between 0 and 31")
    if 2 not in args.test_offsets or any(n < 1 or n > 12 for n in args.test_offsets):
        parser.error("Test offsets must be between 1 and 12 and include 2")
    if args.calibration_offsets and any(n < 1 or n > 12 for n in args.calibration_offsets):
        parser.error("Calibration offsets must be between 1 and 12")
    if args.example_style not in args.test_styles or 1 not in args.donor_shifts or any(n % 12 == 0 for n in args.donor_shifts):
        parser.error("Include the example style and the +1 donor shift in the test settings")
    if args.download:
        snapshot_download(MODEL, revision=REVISION, local_dir=args.model_dir,
                          allow_patterns=["*.json", "*.safetensors"], token=False)
    mx.set_cache_limit(512 * 1024 ** 2)
    print("Loading the 4-bit Mistral checkpoint", flush=True)
    model, _ = load(str(args.model_dir.resolve()))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    model.eval()
    layers = (4, 8, 12, 20, 28) if args.compact_fit else LAYERS
    if args.candidate_layers:
        layers = tuple(sorted(set(args.candidate_layers)))
    if args.fixed_layer is not None:
        layers = (args.fixed_layer,)
    fit = prompts(tokenizer, FIT_STYLES, (3, 6, 10) if args.compact_fit else FIT_STEPS)
    default_calibration_offsets = (5, 9) if args.compact_fit else CALIBRATION_STEPS
    calibration_offsets = tuple(sorted(set(args.calibration_offsets))) if args.calibration_offsets else default_calibration_offsets
    calibration = [] if args.fixed_layer is not None else prompts(tokenizer, (args.calibration_style,), calibration_offsets)
    test_styles = tuple(dict.fromkeys(args.test_styles))
    test = prompts(tokenizer, test_styles, tuple(sorted(set(args.test_offsets))))
    month_ids = output_month_ids(tokenizer, fit[0])
    if not all(np.array_equal(month_ids, output_month_ids(tokenizer, p)) for p in fit + calibration + test):
        raise ValueError("Output tokenizer differs across prompt templates")
    args.output.mkdir(parents=True, exist_ok=True)

    first = fit[0]
    input_sample = mx.array([first.ids])
    true_logits = model(input_sample)[:, -1].astype(mx.float32)
    our_logits, _, _ = forward(model, input_sample, mx.array([len(first.ids) - 1]), mx.array([first.position]), (), None, None)
    max_difference = float(mx.max(mx.abs(true_logits - our_logits)).item())
    if max_difference > 0.05:
        raise ValueError(f"Manual forward diverged from the library model: {max_difference}")
    print(f"Unpatched forward max logit difference: {max_difference:.5f}", flush=True)

    fit_data = run(model, tokenizer, fit, month_ids, args.batch_size, capture=layers, activations_only=True)
    print("Captured fit activations", flush=True)
    fit_labels = np.array([p.month for p in fit])
    selected = []
    plane_methods = tuple(dict.fromkeys(args.plane_fit))
    with threadpool_limits(limits=4):
        planes = {(layer, kind): fit_plane(fit_data["hidden"][:, i], fit_labels, kind)
                  for i, layer in enumerate(layers) for kind in plane_methods}
    if calibration:
        calibration_data = run(model, tokenizer, calibration, month_ids, args.batch_size, capture=layers)
        print("Captured calibration activations", flush=True)
        calibration_answers = np.array([p.answer for p in calibration])
        indices = donor_indices(calibration, 1)
        paired = calibration_answers[indices]
        base = target_gap(calibration_data["months"], calibration_answers, paired)
        for i, layer in enumerate(layers):
            hidden = calibration_data["hidden"][:, i]
            change = np.concatenate([project(hidden[indices] - hidden, planes[layer, kind][0]) for kind in plane_methods])
            patched = run(model, tokenizer, calibration * len(plane_methods), month_ids,
                          args.batch_size, patch_layer=layer, effects=change)
            for kind, changed in zip(plane_methods, np.split(patched["months"], len(plane_methods)), strict=True):
                score = float((target_gap(changed, calibration_answers, paired) - base).mean())
                r2 = planes[layer, kind][2]
                selected.append((score, layer, kind, r2))
                print(f"Calibration layer {layer:02} {kind}: mean target-gap shift {score:.4f}, fitted R² {r2:.3f}", flush=True)
    layer, method = max(selected, key=lambda row: row[0])[1:3] if selected else (layers[0], plane_methods[0])
    plane, first_pc, r2 = planes[layer, method]
    print(f"Frozen intervention layer {layer} · {method}", flush=True)
    rng = np.random.default_rng(1771)
    random_bases = []
    for _ in range(args.random_subspaces):
        candidates = rng.normal(size=plane.shape).astype(np.float32)
        basis, _ = np.linalg.qr(candidates - project(candidates.T, plane).T, mode="reduced")
        random_bases.append(basis)
    early = exact_example(model, tokenizer, test, month_ids, layer, plane, random_bases,
                          background=fit_data["hidden"][:, layers.index(layer)].mean(axis=0), style=args.example_style)
    run_metadata = {"model": MODEL, "revision": REVISION, "layer_pre_block": layer, "compact_fit": args.compact_fit,
                    "fit_examples": len(fit), "selected_plane_fit": method, "calibration_examples": len(calibration),
                    "calibration_style": args.calibration_style, "calibration_offsets": calibration_offsets,
                    "test_styles": test_styles, "test_offsets": sorted(set(args.test_offsets)),
                    "donor_shifts": args.donor_shifts,
                    "layer_calibration": [{"layer": l, "fit_method": m, "mean_logit_shift": s, "r2": r}
                                          for s, l, m, r in selected]}
    (args.output / "mistral-example.json").write_text(json.dumps(
        run_metadata | {"exact": early}, indent=2,
    ) + "\n")
    print(json.dumps(early), flush=True)
    if args.exact_only:
        return

    test_data = run(model, tokenizer, test, month_ids, args.batch_size, capture=(layer,), cache_layer=layer)
    prefixes = test_data.pop("prefixes")
    baseline = {style: baseline_metrics([p for p in test if p.style == style],
                                        {k: v[np.array([p.style == style for p in test])] for k, v in test_data.items()},
                                        month_ids) for style in test_styles}
    print(f"Held-out baseline: {baseline}", flush=True)
    hiddens = test_data["hidden"][:, 0]
    src_answers = np.array([p.answer for p in test])
    src_months = np.array([p.month for p in test])
    no_op = run(model, tokenizer, test[:1], month_ids, 1, patch_layer=layer,
                effects=np.zeros_like(hiddens[:1]), prefixes=prefixes[:1])
    if np.max(np.abs(no_op["months"] - test_data["months"][:1])) > 0.05:
        raise ValueError("Cached prefix changed unpatched month scores")
    exact_i = next(i for i, p in enumerate(test) if p.style == args.example_style and p.month == 10 and p.steps == 2)
    results, combined = [], {name: [] for name in ("circle", "random", "pc1", "source")}
    exact = {}
    for shift in dict.fromkeys(args.donor_shifts):
        donor = donor_indices(test, shift)
        dst_answers = src_answers[donor]
        delta = hiddens[donor] - hiddens
        projected = project(delta, plane)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        base_gap = target_gap(test_data["months"], src_answers, dst_answers)
        circle = run(model, tokenizer, test, month_ids, args.batch_size, patch_layer=layer,
                     effects=projected, prefixes=prefixes)
        random = [run(model, tokenizer, test, month_ids, args.batch_size, patch_layer=layer,
                      effects=match_norm(project(delta, basis), norms), prefixes=prefixes) for basis in random_bases]
        single = run(model, tokenizer, test, month_ids, args.batch_size, patch_layer=layer,
                     effects=match_norm(project(delta, first_pc), norms), prefixes=prefixes)
        donor_full = run(model, tokenizer, test, month_ids, args.batch_size, patch_layer=layer,
                         effects=delta, prefixes=prefixes)
        circle_gap = target_gap(circle["months"], src_answers, dst_answers) - base_gap
        random_gap = np.stack([target_gap(sample["months"], src_answers, dst_answers) - base_gap for sample in random])
        pc1_gap = target_gap(single["months"], src_answers, dst_answers) - base_gap
        full_gap = target_gap(donor_full["months"], src_answers, dst_answers) - base_gap
        combined["circle"].append(circle_gap)
        combined["random"].append(random_gap.mean(axis=0))
        combined["pc1"].append(pc1_gap)
        combined["source"].append(src_months)
        for i, p in enumerate(test):
            results.append({"prompt_style": p.style, "prompt": p.text, "start_month": MONTHS[p.month],
                            "offset": p.steps, "donor_shift": shift, "donor_month": MONTHS[test[donor[i]].month],
                            "original_answer": MONTHS[src_answers[i]], "target_answer": MONTHS[dst_answers[i]],
                            "baseline_top_month": MONTHS[test_data["months"][i].argmax()],
                            "patched_top_month": MONTHS[circle["months"][i].argmax()],
                            "circle_logit_shift": float(circle_gap[i]), "random_mean_shift": float(random_gap[:, i].mean()),
                            "top_pc_shift": float(pc1_gap[i]), "full_patch_shift": float(full_gap[i]),
                            "perturbation_norm": float(norms[i, 0])})
        if shift == 1:
            choices = [src_answers[exact_i], dst_answers[exact_i]]
            exact = {
                "prompt": test[exact_i].text, "original_answer": MONTHS[choices[0]], "target_answer": MONTHS[choices[1]],
                "probabilities": {
                    "baseline": np.exp(test_data["log_probs"][exact_i, choices]).tolist(),
                    "circle": np.exp(circle["log_probs"][exact_i, choices]).tolist(),
                    "random_mean": np.mean([np.exp(s["log_probs"][exact_i, choices]) for s in random], axis=0).tolist(),
                },
                "original_top_token": tokenizer.decode(int(test_data["top_token"][exact_i])),
                "patched_top_token": tokenizer.decode(int(circle["top_token"][exact_i])),
                "circle_logit_shift": float(circle_gap[exact_i]),
                "random_logit_shift_mean": float(random_gap[:, exact_i].mean()),
            }
        write_csv(args.output / "mistral-paired-interventions.csv", results)
        print(f"Held-out donor shift {shift:+}: plane {circle_gap.mean():.3f}, random {random_gap.mean():.3f}", flush=True)
    source = np.concatenate(combined["source"])
    aggregate = {name: {"mean": float(np.concatenate(values).mean()),
                        "ci95": paired_interval(np.concatenate(values), source)}
                 for name, values in combined.items() if name != "source"}
    contrast = np.concatenate(combined["circle"]) - np.concatenate(combined["random"])
    result = run_metadata | {
              "native_forward_max_abs_difference": max_difference, "fit_plane_r2": r2,
              "baseline": baseline, "random_subspaces": args.random_subspaces, "paired_examples": len(results),
              "exact": exact, "aggregate_logit_shifts": aggregate,
              "circle_minus_random": {"mean": float(contrast.mean()), "ci95": paired_interval(contrast, source)}}
    (args.output / "mistral-steering.json").write_text(json.dumps(result, indent=2) + "\n")
    plot_summary(args.output, exact, aggregate)
    print(json.dumps({"exact": exact, "means": aggregate}, indent=2), flush=True)
    if args.compare_paths:
        paths = compare_paths(model, tokenizer, test, month_ids, args.batch_size, layer, plane,
                              fit_data["hidden"][:, layers.index(layer)], fit_labels, test_data)
        (args.output / "mistral-paths.json").write_text(json.dumps(paths, indent=2) + "\n")
        print(json.dumps(paths, indent=2), flush=True)


if __name__ == "__main__":
    main()
