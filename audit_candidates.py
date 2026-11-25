"""Compare blind loop candidates with coactivation-shuffled controls on evaluation text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from rank import (
    PROJECT, activation_shards, document_capped_rows, load_codes, load_decoder,
    load_documents, load_groups, sample_rows, score_mode, write_csv,
)
from reveal import read_ranking, sha256


REFERENCE_GROUPS = (138, 251)
MODES = ("raw", "prototype")
SEED = 479204


def shuffle_codes(code: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Preserve each row's occupancy and each feature's nonzero value histogram."""
    if len(code) < 2:
        return code.copy(), 0.0
    support = code > 0
    occurrences = [set(np.flatnonzero(row)) for row in support]
    for a, b in rng.integers(len(code), size=(10 * len(code), 2)):
        left, right = occurrences[a] - occurrences[b], occurrences[b] - occurrences[a]
        if not left or not right:
            continue
        pool = sorted(left | right)
        new_left = set(rng.choice(pool, len(left), replace=False))
        occurrences[a] = (occurrences[a] - left) | new_left
        occurrences[b] = (occurrences[b] - right) | (set(pool) - new_left)
    shuffled_support = np.zeros_like(support)
    for i, active in enumerate(occurrences):
        shuffled_support[i, list(active)] = True
    shuffled = np.zeros_like(code)
    for feature, col in enumerate(code.T):
        shuffled[shuffled_support[:, feature], feature] = rng.permutation(col[support[:, feature]])
    changed = np.count_nonzero(support & ~shuffled_support) / max(1, support.sum())
    return shuffled, float(changed)


def choose_candidates(tables: list[dict], limits: tuple[int, int]) -> list[dict]:
    group_ids = sorted(tables[0])
    if any(set(table) != set(group_ids) for table in tables):
        raise ValueError("Discovery rankings must compare identical sets of groups")
    records = []
    for mode, limit in zip(MODES, limits, strict=True):
        ranks = np.array([[int(table[group][f"{mode}_rank"]) for group in group_ids] for table in tables])
        consensus = np.exp(np.log(ranks).mean(axis=0))
        order = np.argsort(consensus, kind="stable")
        selected = [i for i in order if group_ids[i] not in REFERENCE_GROUPS][:limit]
        for i in [*selected, *(group_ids.index(group) for group in REFERENCE_GROUPS)]:
            group = group_ids[i]
            records.append({"group": group, "method": mode, "reference": group in REFERENCE_GROUPS,
                            "discovery_ranks": ranks[:, i].tolist(),
                            "discovery_geometric_mean_rank": float(consensus[i])})
    return records


def audit(local, basis: np.ndarray, mode: str, rng: np.random.Generator,
          points: int, draws: int) -> dict:
    sampled, _ = sample_rows(local, points, rng, "balanced")
    code = local[sampled].toarray()
    observed = score_mode(code @ basis.T, mode)[0]
    null = []
    for _ in range(draws):
        shuffled, changed = shuffle_codes(code, rng)
        null.append((score_mode(shuffled @ basis.T, mode)[0], changed))
    scores, changes = np.array(null).T
    return {
        "observed": observed, "null_median": float(np.median(scores)),
        "null_q95": float(np.quantile(scores, 0.95)),
        "null_exceedances": int(np.count_nonzero(scores >= observed)),
        "null_draws": draws, "mean_fraction_of_support_moved": float(changes.mean()),
        "sample_points": len(sampled),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--discovery-rankings", nargs="+", type=Path, required=True)
    parser.add_argument("--evaluation", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=192)
    parser.add_argument("--draws", type=int, default=31)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.points < 16 or args.draws < 1 or args.repeats < 1 or len(args.discovery_rankings) < 2:
        parser.error("Use at least sixteen points, a null draw, a repeat, and two discovery rankings")

    tables, metadata = zip(*(read_ranking(path) for path in args.discovery_rankings), strict=True)
    if set(args.evaluation) & {item["split"] for item in metadata}:
        parser.error("Evaluation corpora must be separate from candidate discovery")
    shortlist = choose_candidates(tables, (8, 4))
    groups = load_groups(PROJECT / "published-groups.json")
    decoder = load_decoder(args.work_dir)
    root = args.work_dir / "activations"
    selections = {}
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"discovery_rankings": {path.name: sha256(path) for path in args.discovery_rankings},
                "evaluation": args.evaluation, "draws_per_repeat": args.draws, "repeats": args.repeats,
                "document_cap": 4, "seed": SEED, "points": args.points, "shortlist": shortlist}
    (args.output / "shortlist.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with threadpool_limits(limits=1):
        for split_index, split in enumerate(args.evaluation):
            paths = activation_shards(root / split, 0)
            codes = load_codes(paths, len(decoder))
            documents = load_documents(paths)
            selected = {}
            for item in shortlist:
                group, mode = item["group"], item["method"]
                members = groups[group]
                local = codes[:, members]
                eligible = document_capped_rows(local, documents, protocol["document_cap"])
                local = local[eligible].tocsc()
                _, basis = np.linalg.qr(decoder[members].T, mode="reduced")
                for rep in range(args.repeats):
                    rng = np.random.default_rng([SEED, split_index, group, MODES.index(mode), rep])
                    result = audit(local, basis, mode, rng, args.points, args.draws)
                    rows.append({"split": split, "group": group, "method": mode, "repeat": rep,
                                 "reference": item["reference"], "eligible_points": len(eligible),
                                 "evaluation_tokens": codes.shape[0], **result})
                if group not in selected and len(eligible):
                    strength = np.asarray(local.power(2).sum(axis=1)).ravel()
                    strongest = np.argsort(-strength, kind="stable")
                    _, first = np.unique(documents[eligible[strongest]], return_index=True)
                    selected[group] = eligible[strongest[np.sort(first)[:128]]]
                print(f"{split} group {group} {mode} finished", flush=True)
            selections[split] = (paths, selected)
            del codes, documents
    output = args.output / "candidate-audit.csv"
    write_csv(output, rows)
    protocol["completed_audit_before_token_reveal_sha256"] = sha256(output)
    (args.output / "shortlist.json").write_text(json.dumps(protocol, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.work_dir / "assets/gpt2", local_files_only=True)
    lexical = []
    for split, (paths, selected) in selections.items():
        tokens = []
        for path in paths:
            with np.load(path, allow_pickle=False) as shard:
                tokens.append(shard["tokens"])
        tokens = np.concatenate(tokens)
        for group, indexes in selected.items():
            visible = tokenizer.batch_decode(tokens[indexes].astype(int).reshape(-1, 1).tolist())
            counts = Counter(token.strip().casefold() for token in visible)
            lexical.append({"split": split, "group": group, "documents": len(indexes),
                            "distinct_tokens": len(counts),
                            "most_frequent_tokens": json.dumps(counts.most_common(10), ensure_ascii=False)})
    write_csv(args.output / "candidate-labels.csv", lexical)


if __name__ == "__main__":
    main()
