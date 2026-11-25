"""Test transfer from naturally observed calendar names to controlled new wordings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from extract import centered_hidden, device_for, model_and_sae
from rank import PROJECT, load_decoder, load_groups, write_csv
from reveal import CALENDARS, labeled_points, load_dataset, orient_circle, read_ranking

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEMPLATES = (
    "The team will meet {date} to review the schedule.",
    "You can pick up the ticket {date} at the front desk.",
    "We expect a response {date} and will call you when it arrives.",
    "The library announced that it would reopen {date} after the maintenance work.",
    "Our office plans to send a reminder {date} about the upcoming deadline.",
    "The next round of interviews begins {date} with a short introduction.",
    "The family decided to start their trip {date} once everyone was ready.",
    "A new shipment is due {date} according to the latest delivery notice.",
)
FORMS = ("full", "lowercase", "uppercase", "abbreviated", "early", "late")


@dataclass(frozen=True)
class Example:
    context: int
    item: int
    form: str
    ids: list[int]
    end_token: int


def examples(tokenizer, concept: str, names: tuple[str, ...]) -> list[Example]:
    preposition = "on" if concept == "Weekdays" else "in"
    out = []
    for context, pattern in enumerate(TEMPLATES):
        for item, name in enumerate(names):
            for form in FORMS:
                word = {"lowercase": name.lower(), "uppercase": name.upper(),
                        "abbreviated": name[:3] + ("." if len(name) > 3 else "")}.get(form, name)
                qualifier = f"very {form} " if form in {"early", "late"} else ""
                surface = f"{qualifier}{preposition} {word}"
                text = pattern.format(date=surface)
                start = text.index(surface) + len(surface) - len(word)
                end = start + len(word.rstrip("."))
                data = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                tokens = [i for i, (a, b) in enumerate(data["offset_mapping"]) if a < end and b > start]
                if not tokens:
                    raise ValueError(f"Unable to locate the calendar date in {text!r}")
                out.append(Example(context, item, form, [tokenizer.eos_token_id, *data["input_ids"]], tokens[-1] + 1))
    return out


@torch.inference_mode()
def encode(model, weights, corpus, features, tokenizer, device, batch_size):
    parts = []
    enc = weights["W_enc"][:, features].contiguous()
    bias = weights["b_enc"][features]
    for start in range(0, len(corpus), batch_size):
        batch = corpus[start:start + batch_size]
        length = max(p.end_token for p in batch) + 1
        input_ids = [p.ids[:p.end_token + 1] + [tokenizer.eos_token_id] * (length - p.end_token - 1) for p in batch]
        hidden = centered_hidden(model, torch.tensor(input_ids, device=device))
        chosen = hidden[torch.arange(len(batch), device=device), [p.end_token for p in batch]]
        values = torch.relu((chosen - weights["b_dec"]) @ enc + bias)
        parts.append(values.cpu().numpy())
    return np.concatenate(parts)


def plot_formats(rows: list[dict], output: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True, layout="constrained")
    for ax, concept in zip(axs, ("Weekdays", "Months"), strict=True):
        formats = [row for row in rows if row["concept"] == concept]
        x = np.arange(len(formats))
        ax.bar(x, [row["activation_coverage"] for row in formats], color="#bccbd6", width=0.65, label="Group fires")
        ax.bar(x, [row["accuracy_including_inactive"] for row in formats], color="#11756c", width=0.65, label="Correct angle")
        ax.set_xticks(x, [row["form"] for row in formats], rotation=28, ha="right", fontsize=8)
        ax.set_title(concept, loc="left", weight="semibold", fontsize=11)
        ax.set_ylim(0, 1.04)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.24)
    axs[0].set_ylabel("Fraction of controlled prompts")
    axs[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Does GPT-2's calendar plane transfer to different forms?", fontsize=14, weight="semibold")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".experiment"))
    parser.add_argument("--fit-ranking", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    device = device_for(args.device)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.work_dir / "assets/gpt2", local_files_only=True)
    decoder = load_decoder(args.work_dir)
    groups = load_groups(PROJECT / "published-groups.json")
    _, meta = read_ranking(args.fit_ranking)
    data = load_dataset(args.work_dir, meta, len(decoder))
    present = np.unique(data["tokens"])
    strings = tokenizer.batch_decode(present.astype(int).reshape(-1, 1).tolist())
    token_names = dict(zip(present, (word.strip() for word in strings), strict=True))
    model, weights = model_and_sae(args.work_dir / "assets", device)
    tables, timings = [], []
    with threadpool_limits(limits=2):
        for group, (concept, names) in CALENDARS.items():
            features = groups[group]
            lookup = {i: names.index(name) for i, name in token_names.items() if name in names}
            background, labeled, labels, _ = labeled_points(data, features, decoder, lookup)
            center = background.mean(axis=0)
            _, _, axes = np.linalg.svd(background - center, full_matrices=False)
            basis = axes[1:3].T
            fit = (labeled - center) @ basis
            origin, sign, rotation = orient_circle(fit[:, 0] + 1j * fit[:, 1], labels, len(names))
            corpus = examples(tokenizer, concept, names)
            z = encode(model, weights, corpus, features, tokenizer, device, args.batch_size)
            _, r = np.linalg.qr(decoder[features].T, mode="reduced")
            coords = (z @ r.T - center) @ basis
            phase = coords[:, 0] + 1j * coords[:, 1] - origin
            phase = (phase if sign == 1 else phase.conj()) * rotation
            active = np.max(z, axis=1) > 1e-6
            predicted = np.round(np.angle(phase) * len(names) / (2 * np.pi)).astype(int) % len(names)
            matches = predicted == np.array([p.item for p in corpus])
            for form in FORMS:
                part = np.array([p.form == form for p in corpus])
                hits = part & active & matches
                tables.append({"concept": concept, "form": form, "examples": int(part.sum()),
                               "activation_coverage": float(active[part].mean()),
                               "accuracy_including_inactive": float(hits.sum() / part.sum()),
                               "accuracy_given_activation": float(matches[part & active].mean()) if (part & active).any() else None})
            neutral = {(p.context, p.item): phase[i] for i, p in enumerate(corpus) if p.form == "full" and active[i]}
            for form, expected in (("early", -1), ("late", 1)):
                shifts = np.array([np.angle(phase[i] * np.conj(neutral[p.context, p.item]))
                                   for i, p in enumerate(corpus) if active[i] and p.form == form
                                   and (p.context, p.item) in neutral])
                timings.append({"concept": concept, "modifier": form, "paired_examples": len(shifts),
                                "mean_shift_degrees": float(np.degrees(shifts.mean())) if len(shifts) else None,
                                "expected_sign_fraction": float(np.mean(shifts * expected > 0)) if len(shifts) else None})
            print(f"Encoded {len(corpus)} held-out formats for {concept.lower()}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "calendar-format-transfer.csv", tables)
    write_csv(args.output / "calendar-time-modifiers.csv", timings)
    plot_formats(tables, args.output / "calendar-format-transfer.png")


if __name__ == "__main__":
    main()
