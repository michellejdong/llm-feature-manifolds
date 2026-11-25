"""Cache published SAE activations from a frozen GPT-2-small on public text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

if TYPE_CHECKING:
    from transformers import GPT2Model


GPT2 = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SAE = "jbloom/GPT2-Small-SAEs-Reformatted"
SAE_REVISION = "57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9"
PILE = "NeelNanda/pile-10k"
PILE_REVISION = "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
PILE_FILE = "data/train-00000-of-00001-4746b8785c874cc7.parquet"
WIKI = "Salesforce/wikitext"
WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKI_TRAIN = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
WIKI_TEST = "wikitext-103-raw-v1/test-00000-of-00001.parquet"
NEWS = "abisee/cnn_dailymail"
NEWS_REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
NEWS_TEST = "3.0.0/test-00000-of-00001.parquet"
HOOK = "blocks.7.hook_resid_pre"
LAYER = 7
WIDTH = 24_576
CONTEXT = 128
SHARD_SEQUENCES = 256
FIRING_THRESHOLD = 1e-6


def download_assets(assets: Path) -> None:
    jobs = [
        ("model", GPT2, GPT2_REVISION, name, "gpt2")
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    ]
    jobs += [
        ("model", SAE, SAE_REVISION, f"{HOOK}/{name}", "sae")
        for name in ("cfg.json", "sae_weights.safetensors")
    ]
    jobs += [("dataset", PILE, PILE_REVISION, PILE_FILE, "pile")]
    jobs += [("dataset", WIKI, WIKI_REVISION, name, "wiki") for name in (WIKI_TRAIN, WIKI_TEST)]

    def download(job: tuple[str, str, str, str, str]) -> None:
        repo_type, repo, revision, filename, directory = job
        hf_hub_download(
            repo, filename, revision=revision, repo_type=repo_type,
            local_dir=assets / directory, token=False,
        )
        print(f"Ready {directory}/{filename}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, jobs))


def device_for(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_and_sae(assets: Path, device: str) -> tuple[GPT2Model, dict[str, torch.Tensor]]:
    from transformers import GPT2Model

    model = GPT2Model.from_pretrained(
        assets / "gpt2", local_files_only=True, use_safetensors=True, attn_implementation="sdpa",
    )
    # The stream entering block 7 is the output of blocks 0 through 6.
    model.h = model.h[:LAYER]
    model.ln_f = torch.nn.Identity()
    model.requires_grad_(False).eval().to(device)
    checkpoint = assets / "sae" / HOOK / "sae_weights.safetensors"
    weights = {name: array.to(device) for name, array in load_file(checkpoint).items()}
    expected = {"W_enc": (768, WIDTH), "W_dec": (WIDTH, 768), "b_enc": (WIDTH,), "b_dec": (768,)}
    for name, shape in expected.items():
        if weights[name].shape != shape:
            raise ValueError(f"Unexpected shape for {name}: {weights[name].shape}")
    return model, weights


def centered_hidden(model: GPT2Model, input_ids: torch.Tensor) -> torch.Tensor:
    hidden = model(input_ids, use_cache=False).last_hidden_state
    # TransformerLens centers the weights writing to this stream. In GPT-2,
    # this removes a per-token constant along the all-ones direction.
    return hidden - hidden.mean(dim=-1, keepdim=True)


def parquet_text(path: Path, column: str = "text") -> list[str]:
    return pq.read_table(path, columns=[column])[column].to_pylist()


def wiki_articles(lines: list[str]) -> list[str]:
    article = []
    articles = []
    for line in lines:
        if re.fullmatch(r"\s*=\s+[^=]+?\s+=\s*", line) and article:
            articles.append("\n".join(article))
            article = []
        if line.strip():
            article.append(line)
    if article:
        articles.append("\n".join(article))
    return articles


def get_documents(assets: Path, split: str) -> list[str]:
    if split == "news-test":
        return parquet_text(assets / "news" / NEWS_TEST, column="article")
    if split.startswith("pile-"):
        docs = parquet_text(assets / "pile" / PILE_FILE)
        holdout = split == "pile-holdout"
        return [
            text for text in docs
            if (int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little") % 5 == 0)
            == holdout
        ]
    filename = WIKI_TEST if split == "wiki-test" else WIKI_TRAIN
    return wiki_articles(parquet_text(assets / "wiki" / filename))


def sequences(assets: Path, split: str, count: int):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(assets / "gpt2", local_files_only=True)
    documents = get_documents(assets, split)
    doc_order = np.random.default_rng(20261125).permutation(len(documents))
    eos = tokenizer.eos_token_id
    seen: set[bytes] = set()
    emitted = 0
    for start in range(0, len(doc_order), 64):
        batch = [documents[i] for i in doc_order[start:start + 64]]
        tokens = tokenizer(batch, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        for document, ids in zip(batch, tokens, strict=True):
            identity = hashlib.blake2b(document.encode(), digest_size=16).digest()
            if identity in seen:
                continue
            seen.add(identity)
            document_id = int.from_bytes(identity[:8], "little")
            for start_token in range(0, len(ids), CONTEXT - 1):
                context = ids[start_token:start_token + CONTEXT - 1]
                if len(context) < CONTEXT // 4:
                    continue
                length = len(context) + 1
                yield [eos, *context, *([eos] * (CONTEXT - length))], length, document_id
                emitted += 1
                if emitted == count:
                    return


@torch.inference_mode()
def encode_batch(
    model: GPT2Model, weights: dict[str, torch.Tensor], token_batch: np.ndarray,
    lengths: np.ndarray, device: str, residuals: bool,
) -> dict[str, np.ndarray]:
    tokens = torch.as_tensor(token_batch, dtype=torch.long, device=device)
    hidden = centered_hidden(model, tokens).reshape(-1, 768)
    keep = (np.arange(CONTEXT) < lengths[:, None]).ravel()
    result: dict[str, np.ndarray] = {"tokens": token_batch.ravel()[keep].astype(np.uint16)}
    if residuals:
        result["residuals"] = hidden.cpu().numpy()[keep].astype(np.float16)

    rows, columns, values = [], [], []
    for start in range(0, len(hidden), 512):
        block = hidden[start:start + 512]
        z = torch.relu((block - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"])
        # Transfer the compact top activations. If a token fills the buffer,
        # recover that token's whole row so this remains a lossless threshold.
        largest, feature_ids = z.topk(128, dim=1, sorted=False)
        largest, feature_ids = largest.cpu().numpy(), feature_ids.cpu().numpy()
        active = largest > FIRING_THRESHOLD
        overflow = np.flatnonzero(active.all(axis=1))
        active[overflow] = False
        token_rows, slots = np.nonzero(active)
        token_cols, token_values = feature_ids[token_rows, slots], largest[token_rows, slots]
        if len(overflow):
            dense = z[torch.as_tensor(overflow, device=device)].cpu().numpy()
            full_rows, full_cols = np.nonzero(dense > FIRING_THRESHOLD)
            token_rows = np.r_[token_rows, overflow[full_rows]]
            token_cols = np.r_[token_cols, full_cols]
            token_values = np.r_[token_values, dense[full_rows, full_cols]]
            order = np.argsort(token_rows, kind="stable")
            token_rows, token_cols, token_values = token_rows[order], token_cols[order], token_values[order]
        rows.append(token_rows + start)
        columns.append(token_cols)
        values.append(token_values)

    rows = np.concatenate(rows)
    include = keep[rows]
    row_sizes = np.bincount(rows[include], minlength=len(hidden))[keep]
    result["indptr"] = np.r_[0, np.cumsum(row_sizes, dtype=np.int32)]
    result["indices"] = np.concatenate(columns)[include].astype(np.uint16)
    result["values"] = np.concatenate(values)[include].astype(np.float32)
    return result


def write_shard(path: Path, batches: list[dict[str, np.ndarray]], documents: list[np.ndarray]) -> None:
    sizes = np.concatenate([np.diff(batch["indptr"]) for batch in batches])
    output = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0] if key != "indptr"}
    output["indptr"] = np.r_[0, np.cumsum(sizes, dtype=np.int32)]
    output["documents"] = np.concatenate(documents)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **output)
    tmp.replace(path)


def extract(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    output = args.work_dir / "activations" / args.split
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "split": args.split, "layer": LAYER, "context": CONTEXT,
        "shard_sequences": SHARD_SEQUENCES, "contains_residuals": args.residuals,
        "encoder": "lossless-topk-threshold-v1",
        "model_revision": GPT2_REVISION, "sae_revision": SAE_REVISION,
        "corpus_revision": (PILE_REVISION if args.split.startswith("pile-")
                            else {"news-test": NEWS_REVISION}.get(args.split, WIKI_REVISION)),
    }
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError(f"Cache settings differ from {manifest}")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Extracting {args.split} on {device}", flush=True)
    started = time.monotonic()
    model, weights = model_and_sae(args.work_dir / "assets", device)
    source = iter(sequences(args.work_dir / "assets", args.split, args.sequences))
    shard_index, processed = 0, 0
    from itertools import islice

    while shard := list(islice(source, SHARD_SEQUENCES)):
        path = output / f"{shard_index:05}.npz"
        if path.exists():
            with np.load(path) as saved:
                if len(saved["tokens"]) != sum(row[1] for row in shard):
                    raise ValueError(f"Cached shard does not match the regenerated sequence stream: {path}")
        else:
            batches, document_arrays = [], []
            for start in range(0, len(shard), args.batch_size):
                tokens, lengths, ids = zip(*shard[start:start + args.batch_size], strict=True)
                lengths_array = np.asarray(lengths)
                batches.append(encode_batch(model, weights, np.array(tokens), lengths_array, device, args.residuals))
                document_arrays.append(np.repeat(np.array(ids, dtype=np.uint64), lengths_array))
            write_shard(path, batches, document_arrays)
        processed += len(shard)
        shard_index += 1
        print(f"{args.split} shard {shard_index:03}, {processed:,} sequences, {time.monotonic() - started:.1f}s", flush=True)
    print(f"Completed {args.split}, {processed:,} sequences", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".experiment"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download")
    commands.add_parser("download-news")
    run = commands.add_parser("extract")
    run.add_argument("split", choices=("pile-fit", "pile-holdout", "wiki-train", "wiki-test", "news-test"))
    run.add_argument("--sequences", type=int, default=4_096)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--device", default="auto")
    run.add_argument("--residuals", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "download":
        download_assets(args.work_dir / "assets")
    elif args.command == "download-news":
        hf_hub_download(NEWS, NEWS_TEST, revision=NEWS_REVISION, repo_type="dataset",
                        local_dir=args.work_dir / "assets" / "news", token=False)
    else:
        if args.sequences < SHARD_SEQUENCES or args.sequences % SHARD_SEQUENCES:
            parser.error(f"--sequences must be a positive multiple of {SHARD_SEQUENCES}")
        if args.batch_size < 1 or args.threads < 1:
            parser.error("--batch-size and --threads must be positive")
        extract(args)


if __name__ == "__main__":
    main()
