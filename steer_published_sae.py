"""Fit the published Mistral SAE month plane on natural text and intervene in it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from safetensors import safe_open
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from extract import get_documents
from rank import loop_scores, write_csv
from reveal import angular_summary, orient_circle
from steer_months import (
    MODEL, MONTHS, REVISION, donor_indices, fit_plane, match_norm, output_month_ids,
    paired_interval, project, prompts, run, target_gap,
)


SAE_MODEL = "JoshEngels/Mistral-7B-Residual-Stream-SAEs"
SAE_REVISION = "d2d2a64135d2a2c8e7cdfe18c07297af251c744b"
SAE_SHA256 = "342d7c8d26776db7787f4454ade76c2cb973a32ef3434926d9e270d76a1906d0"
LAYER = 8
# The authors' month cluster is index 832 after singleton groups are removed.
GROUP = [7411, 33259, 49189, 46031, 9117, 57916, 26027, 16820, 41121, 23434, 39714,
         59285, 47182, 22809, 17555, 52568, 8934, 16406, 63163, 15477, 54144]
FIT_OFFSETS = (3, 6, 10)
TEST_OFFSETS = (2, 5, 7)


def load_group(weight_path: Path) -> tuple:
    with weight_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SAE_SHA256:
            raise ValueError("Published Mistral SAE checksum mismatch")
    with safe_open(weight_path, framework="np") as state:
        encoder = np.ascontiguousarray(state.get_tensor("W_enc")[:, GROUP])
        bias = state.get_tensor("b_enc")[GROUP]
        decoder = state.get_tensor("W_dec")[GROUP]
    return encoder, bias, decoder


def encode_sae(hidden: np.ndarray, encoder: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # The published conversion expects a residual with norm sqrt(d_model).
    scale = np.sqrt(hidden.shape[-1]) / np.maximum(np.linalg.norm(hidden, axis=-1, keepdims=True), 1e-8)
    codes = np.maximum(0, (hidden * scale.astype(hidden.dtype)) @ encoder + bias)
    return codes, scale.astype(hidden.dtype)


def collect_cloud(model, tokenizer, documents, encoder, bias, doc_count, max_length, batch_size) -> tuple:
    rng = np.random.default_rng(273411)
    selection = rng.choice(len(documents), min(doc_count, len(documents)), replace=False)
    emitted, code_chunks, word_chunks, used_docs = 0, [], [], 0
    for start in range(0, len(selection), batch_size):
        examples = [documents[i] for i in selection[start:start + batch_size]]
        examples = tokenizer(examples, truncation=True, max_length=max_length)["input_ids"]
        length = max(map(len, examples))
        padded = mx.array([seq + [tokenizer.eos_token_id] * (length - len(seq)) for seq in examples])
        hidden = model.model.embed_tokens(padded)
        mask = create_attention_mask(hidden, None)
        for layer in model.model.layers[:LAYER]:
            hidden = layer(hidden, mask, cache=None)
        as_array = np.asarray(hidden.astype(mx.float32))
        for h, token_ids in zip(as_array, examples, strict=True):
            h = h[1:len(token_ids)]
            emitted += len(h)
            codes, _ = encode_sae(h, encoder, bias)
            strength = np.square(codes).sum(axis=1)
            strongest = np.argsort(-strength, kind="stable")[:12]
            strongest = strongest[strength[strongest] > 0]
            if len(strongest):
                used_docs += 1
                code_chunks.append(codes[strongest])
                word_chunks.append(np.asarray(token_ids[1:])[strongest])
        if (start // batch_size + 1) % 12 == 0:
            print(f"Read {min(start + batch_size, len(selection)):,}/{len(selection):,} natural documents", flush=True)
    if not code_chunks:
        raise ValueError("The published SAE group did not activate in the natural source")
    return np.concatenate(code_chunks), np.concatenate(word_chunks), emitted, used_docs


def audit_plane(run_data, fit_data, plane, fit_prompts, test_prompts) -> dict:
    fit_projection = fit_data @ plane
    labels = np.array([p.month for p in fit_prompts])
    center, sign, phase = orient_circle(fit_projection[:, 0] + 1j * fit_projection[:, 1], labels, 12)
    xy = run_data @ plane
    z = xy[:, 0] + 1j * xy[:, 1] - center
    return angular_summary((z if sign == 1 else z.conj()) * phase, np.array([p.month for p in test_prompts]), 12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sae-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--documents", type=int, default=768)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--natural-batch-size", type=int, default=12)
    args = parser.parse_args()
    if min(args.documents, args.context, args.natural_batch_size) < 1:
        parser.error("Use positive natural-text collection settings")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.download:
        path = Path(hf_hub_download(SAE_MODEL, "mistral_7b_layer_8/sae_weights.safetensors",
                                    revision=SAE_REVISION, cache_dir=args.sae_dir, token=False))
    else:
        path = args.sae_dir / "sae_weights.safetensors"
    encoder, bias, decoder = load_group(path)
    print("Verified and loaded the 21 published Mistral month features", flush=True)
    mx.set_cache_limit(512 * 1024 ** 2)
    model, _ = load(str(args.model_dir.resolve()))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    texts = get_documents(args.work_dir / "assets", "news-test")
    with threadpool_limits(limits=2):
        codes, token_ids, total, docs = collect_cloud(model, tokenizer, texts, encoder, bias,
                                                     args.documents, args.context, args.natural_batch_size)
        q, r = np.linalg.qr(decoder.T, mode="reduced")
        coordinates = codes @ r.T
        _, s, axes = np.linalg.svd(coordinates - coordinates.mean(axis=0), full_matrices=False)
        plane = q @ axes[1:3].T
    plane_file = args.output / "natural-text-plane.npy"
    np.save(plane_file, plane)
    # Finish fitting and persist the plane before decoding any tokens.
    with plane_file.open("rb") as stream:
        plane_checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    revealed = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    tokens = [token.replace("▁", "").casefold() for token in revealed]
    counts = {name: tokens.count(name.casefold()) for name in MONTHS}
    plane_points = coordinates @ axes[1:3].T
    sample = np.random.default_rng(32).choice(len(plane_points), min(448, len(plane_points)), replace=False)
    fit_details = {"checkpoint": SAE_MODEL, "checkpoint_revision": SAE_REVISION, "weight_sha256": SAE_SHA256,
                   "features": GROUP, "fit_uses_token_labels": False, "pre_block_layer": LAYER,
                   "documents_requested": args.documents, "documents_with_group_activity": docs,
                   "context_tokens": args.context, "total_natural_text_tokens": total, "fit_points": len(codes),
                   "fit_loop_score": loop_scores(plane_points[sample])[0], "active_full_month_names": counts,
                   "plane_sha256_before_label_reveal": plane_checksum,
                   "pc2_pc3_variance_fraction": float(np.square(s[1:3]).sum() / np.square(s).sum())}
    (args.output / "natural-text-fit.json").write_text(json.dumps(fit_details, indent=2) + "\n")
    print(json.dumps(fit_details), flush=True)

    fit = prompts(tokenizer, ("paper", "from"), FIT_OFFSETS)
    test = prompts(tokenizer, ("paper", "paper-after"), TEST_OFFSETS)
    month_ids = output_month_ids(tokenizer, fit[0])
    if not all(np.array_equal(month_ids, output_month_ids(tokenizer, p)) for p in fit + test):
        raise ValueError("Month output IDs must agree across prompts")
    h_fit = run(model, tokenizer, fit, month_ids, 48, capture=(LAYER,), activations_only=True)["hidden"][:, 0]
    with threadpool_limits(limits=2):
        supervised, _, _ = fit_plane(h_fit, np.array([p.month for p in fit]))
    base = run(model, tokenizer, test, month_ids, 48, capture=(LAYER,), cache_layer=LAYER)
    hidden = base["hidden"][:, 0]
    readout = audit_plane(hidden, h_fit, plane, fit, test)
    codes, scales = encode_sae(hidden, encoder, bias)
    group_only = (codes @ decoder) / scales
    answers = np.array([p.answer for p in test])
    months = np.array([p.month for p in test])
    rng = np.random.default_rng(104924)
    random_basis = [np.linalg.qr(rng.normal(size=plane.shape).astype(np.float32), mode="reduced")[0]
                    for _ in range(6)]
    rows, overview = [], []
    for shift in (1, 3):
        source = donor_indices(test, shift)
        target = answers[source]
        delta = hidden[source] - hidden
        unsupervised = project(delta, plane)
        norms = np.linalg.norm(unsupervised, axis=1, keepdims=True)
        alternatives = {"natural_text_plane": [unsupervised],
                        "direct_sae_edit": [project(group_only[source] - group_only, plane)],
                        "supervised_plane_equal_norm": [match_norm(project(delta, supervised), norms)],
                        "random_plane_equal_norm": [match_norm(project(delta, b), norms) for b in random_basis],
                        "full_donor": [delta]}
        gap = target_gap(base["months"], answers, target)
        for condition, edits in alternatives.items():
            responses = [run(model, tokenizer, test, month_ids, 48, patch_layer=LAYER, effects=edit,
                             prefixes=base["prefixes"]) for edit in edits]
            changes = np.mean([target_gap(s["months"], answers, target) for s in responses], axis=0) - gap
            probabilities = np.mean([np.exp(s["log_probs"][np.arange(len(test)), target]) for s in responses], axis=0)
            wins = np.mean([s["top_token"] == month_ids[target] for s in responses], axis=0)
            mean_norms = np.mean([np.linalg.norm(x, axis=1) for x in edits], axis=0)
            for i, p in enumerate(test):
                rows.append({"donor_shift": shift, "condition": condition, "frame": p.style,
                             "source_month": MONTHS[p.month], "offset": p.steps,
                             "target_answer": MONTHS[target[i]], "edit_norm": float(mean_norms[i]),
                             "target_logit_gap_change": float(changes[i]), "target_probability": float(probabilities[i]),
                             "base_target_probability": float(np.exp(base["log_probs"][i, target[i]])),
                             "vocabulary_target_top_rate": float(wins[i])})
            overview.append({"donor_shift": shift, "condition": condition, "mean_target_gap_change": float(changes.mean()),
                             "ci95": paired_interval(changes, months), "mean_norm": float(mean_norms.mean()),
                             "vocabulary_target_top_rate": float(wins.mean())})
            print(f"{shift:+} {condition}: mean change {changes.mean():.4f}", flush=True)
    write_csv(args.output / "sae-plane-pairs.csv", rows)
    report = {"model": MODEL, "model_revision": REVISION, "test_offsets": TEST_OFFSETS,
              "fit": fit_details, "heldout_input_month_readout": readout,
              "plane_overlap_with_supervised": float(np.square(plane.T @ supervised).sum() / 2),
              "aggregate": overview,
              "exact": [r for r in rows if r["source_month"] == "November" and r["offset"] == 2
                        and r["donor_shift"] == 1 and r["frame"] == "paper"]}
    (args.output / "sae-plane-steering.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
