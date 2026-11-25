"""Export compact real-data examples for the CPU tutorials."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rank import cluster_cloud, load_decoder, load_groups, write_csv
from reveal import CALENDARS, labeled_points, load_dataset, read_ranking


def export_gpt2(args) -> None:
    decoder = load_decoder(args.work_dir)
    _, fit_info = read_ranking(args.fit)
    fit = load_dataset(args.work_dir, fit_info, len(decoder))
    groups = load_groups(PROJECT / "published-groups.json")
    special = {138: "Published weekdays", 251: "Published months", 212: "Published years",
               271: "Candidate associated with -ing", 334: "Candidate associated with legs and feet"}
    entries = [138, 251, 212, 271, 334, 77, 162, 325, 544, 61, 492, 872]
    rng = np.random.default_rng(2241)
    names = CALENDARS[138][1]
    tokenizer = AutoTokenizer.from_pretrained(args.work_dir / "assets/gpt2", local_files_only=True)
    words = tokenizer.batch_decode(np.arange(len(tokenizer)).reshape(-1, 1).tolist())
    day_ids = {i: names.index(word.strip()) for i, word in enumerate(words) if word.strip() in names}
    matrices, labels = {}, []
    with threadpool_limits(limits=1):
        for i, group in enumerate(rng.permutation(entries), 1):
            key = f"G{i:02d}"
            members = groups[int(group)]
            _, rows, _ = cluster_cloud(fit["codes"], decoder, members, 128, rng, "balanced",
                                       documents=fit["documents"], per_document_limit=4)
            matrices[f"{key}_codes"] = fit["codes"][:, members][rows].toarray()
            matrices[f"{key}_decoder"] = decoder[members]
            labels.append({"anonymous_id": key, "published_group": int(group),
                           "description": special.get(group, "Comparison group")})
        fit_bg, _, _, _ = labeled_points(fit, groups[138], decoder, day_ids)
        del fit
        _, test_info = read_ranking(args.test)
        test = load_dataset(args.work_dir, test_info, len(decoder))
        _, holdout, weekday_id, _ = labeled_points(test, groups[138], decoder, day_ids)
    np.savez_compressed(args.output / "blind-groups.npz", **matrices)
    write_csv(args.output / "group-reveal.csv", labels)
    np.savez_compressed(args.output / "weekday-reveal.npz", fit_cloud=fit_bg, news_points=holdout,
                        weekday_id=weekday_id, weekday_names=np.array(names))


def export_steering(args) -> None:
    source = PROJECT / "results/steering/harmonics/harmonic-interventions.csv"
    conditions = {("harmonics", 2), ("harmonics", 8), ("harmonics", 11), ("pca", 11),
                  ("random", 11), ("full_donor", 4096)}
    with source.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if (row["condition"], int(row["dimensions"])) in conditions]
    write_csv(args.output / "steering-examples.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "notebooks/data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    export_gpt2(args)
    export_steering(args)


if __name__ == "__main__":
    main()
