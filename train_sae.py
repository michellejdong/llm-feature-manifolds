"""Train and evaluate a separate TopK SAE on cached GPT-2 residuals."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from threadpoolctl import threadpool_limits

from extract import device_for
from rank import activation_shards


LOSS_POWERS = {"mse": 0.0, "relative-sqrt": 0.5, "relative-token": 1.0}


class TopKSAE(torch.nn.Module):
    def __init__(self, width: int, k: int, dimension: int = 768, block_size: int = 1):
        super().__init__()
        self.k = k
        self.block_size = block_size
        self.decoder = torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(width, dimension), dim=1))
        self.encoder = torch.nn.Parameter(self.decoder.detach().T.contiguous().clone())
        self.encoder_bias = torch.nn.Parameter(torch.zeros(width))
        self.decoder_bias = torch.nn.Parameter(torch.zeros(dimension))
        self.register_buffer("center", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(()))

    def normalized(self, batch: torch.Tensor) -> torch.Tensor:
        return (batch - self.center) / self.scale

    def sparse_code(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preactivation = (normalized - self.decoder_bias) @ self.encoder + self.encoder_bias
        if self.block_size == 1:
            values, indices = preactivation.topk(self.k, dim=-1, sorted=False)
            return values.relu(), indices
        blocks = preactivation.relu().reshape(len(normalized), -1, self.block_size)
        _, selected = blocks.square().sum(dim=-1).topk(self.k // self.block_size, sorted=False)
        values = torch.gather(blocks, 1, selected[..., None].expand(-1, -1, self.block_size))
        offsets = torch.arange(self.block_size, device=normalized.device)
        indices = selected[..., None] * self.block_size + offsets
        return values.flatten(1), indices.flatten(1)

    def forward(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values, indices = self.sparse_code(normalized)
        code = normalized.new_zeros((len(normalized), len(self.decoder))).scatter(1, indices, values)
        prediction = code @ self.decoder + self.decoder_bias
        return prediction, values, indices


def cache_data(paths: list[Path]) -> np.ndarray:
    counts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            if "residuals" not in shard:
                raise ValueError(f"Residuals were not saved in {path}")
            counts.append(len(shard["indptr"]) - 1)
    data = np.empty((sum(counts), 768), dtype=np.float16)
    start = 0
    for path, count in zip(paths, counts, strict=True):
        with np.load(path, allow_pickle=False) as shard:
            data[start:start + count] = shard["residuals"]
        start += count
    return data


@torch.inference_mode()
def evaluate(sae: TopKSAE, rows: np.ndarray, device: str) -> dict:
    numerator, denominator, relative, active = 0.0, 0.0, 0.0, 0
    used = np.zeros(len(sae.decoder), dtype=bool)
    for start in range(0, len(rows), 1024):
        batch = torch.from_numpy(rows[start:start + 1024]).to(device, dtype=torch.float32)
        inputs = sae.normalized(batch)
        output, values, indices = sae(inputs)
        errors = (inputs - output).square().mean(dim=1)
        strength = inputs.square().mean(dim=1)
        numerator += errors.sum().item()
        denominator += strength.sum().item()
        relative += (errors / strength.clamp_min(0.05)).sum().item()
        live = values.cpu().numpy() > 1e-6
        active += int(live.sum())
        used[indices.cpu().numpy()[live]] = True
    return {"fvu_relative_to_fit_mean": numerator / denominator,
            "average_relative_token_error": relative / len(rows),
            "mean_nonzero_latents": active / len(rows), "latents_seen": int(used.sum()),
            "evaluation_tokens": len(rows)}


def train(args: argparse.Namespace) -> None:
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train_rows = cache_data(activation_shards(args.work_dir / "activations" / args.fit, args.fit_shards))
    validation_paths = activation_shards(args.work_dir / "activations" / args.validation, args.validation_shards)
    validation = cache_data(validation_paths)
    eval_rng = np.random.default_rng(291053)
    validation = validation[eval_rng.choice(len(validation), min(24_576, len(validation)), replace=False)]
    sample = train_rows[eval_rng.choice(len(train_rows), min(32_768, len(train_rows)), replace=False)].astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"fit": args.fit, "validation": args.validation, "fit_tokens": len(train_rows),
              "validation_shards": len(validation_paths), "width": args.width, "k": args.k,
              "seed": args.seed, "batch_size": args.batch_size, "learning_rate": args.lr}
    if args.block_size > 1:
        config["block_size"] = args.block_size
    if args.loss != "mse":
        config["loss"] = args.loss
    metadata = args.output / "config.json"
    if metadata.exists() and json.loads(metadata.read_text()) != config:
        raise ValueError(f"Checkpoint configuration has changed: {metadata}")
    metadata.write_text(json.dumps(config, indent=2) + "\n")

    sae = TopKSAE(args.width, args.k, block_size=args.block_size).to(device)
    center = sample.mean(axis=0)
    scale = np.sqrt(np.mean((sample - center) ** 2))
    sae.center.copy_(torch.from_numpy(center).to(device))
    sae.scale.fill_(float(scale))
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    checkpoint = args.output / "optimizer.pt"
    starting_step = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sae.load_state_dict(saved["sae"])
        optimizer.load_state_dict(saved["optimizer"])
        starting_step = saved["step"]
    if args.resume_lr is not None:
        if not starting_step:
            raise ValueError("--resume-lr requires an existing checkpoint")
        for group in optimizer.param_groups:
            group["lr"] = args.resume_lr
    if args.steps < starting_step:
        raise ValueError(f"Existing checkpoint is already at step {starting_step}")
    history = args.output / "history.jsonl"
    started = time.monotonic()
    loss_power = LOSS_POWERS[args.loss]
    print(f"TopK SAE width {args.width:,}, block size {args.block_size}, k={args.k}, {len(train_rows):,} fit tokens on {device}", flush=True)

    for step in range(starting_step + 1, args.steps + 1):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
        rows = rng.integers(len(train_rows), size=args.batch_size)
        batch = torch.from_numpy(train_rows[rows]).to(device, dtype=torch.float32)
        inputs = sae.normalized(batch)
        prediction, _, _ = sae(inputs)
        errors = (prediction - inputs).square().mean(dim=1)
        batch_mse = errors.mean()
        loss = batch_mse
        if loss_power:
            magnitude = inputs.square().mean(dim=1).clamp_min(0.05)
            loss = (errors / magnitude.pow(loss_power)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), args.clip_grad)
        optimizer.step()
        with torch.no_grad():
            sae.decoder.copy_(torch.nn.functional.normalize(sae.decoder, dim=1))
        if step % 100 == 0:
            print(f"step {step:,} · {args.loss} loss {loss.item():.5f} · {time.monotonic() - started:.1f}s", flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            result = {"step": step, "train_batch_mse": batch_mse.item(), "train_batch_loss": loss.item(),
                      "learning_rate": optimizer.param_groups[0]["lr"], "clip_grad": args.clip_grad,
                      **evaluate(sae, validation, device)}
            print(json.dumps(result), flush=True)
            with history.open("a") as f:
                f.write(json.dumps(result) + "\n")
            tmp = checkpoint.with_suffix(".tmp")
            torch.save({"step": step, "sae": sae.state_dict(), "optimizer": optimizer.state_dict()}, tmp)
            tmp.replace(checkpoint)

    tensors = {key: value.detach().cpu().contiguous() for key, value in sae.state_dict().items()}
    weight_file = args.output / "topk-sae.safetensors"
    tmp = weight_file.with_suffix(".tmp")
    save_file(tensors, tmp)
    tmp.replace(weight_file)
    (args.output / "validation.json").write_text(json.dumps({"step": args.steps, **evaluate(sae, validation, device)}, indent=2) + "\n")


@torch.inference_mode()
def recode(args: argparse.Namespace) -> None:
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    metadata = json.loads((args.model / "config.json").read_text())
    sae = TopKSAE(metadata["width"], metadata["k"], block_size=metadata.get("block_size", 1))
    sae.load_state_dict(load_file(args.model / "topk-sae.safetensors"))
    sae.eval().to(device)
    destination = args.work_dir / "activations" / args.output_split
    destination.mkdir(parents=True, exist_ok=True)
    revision = {"from_split": args.split, "sae": metadata,
                "weights_file_stats": (args.model / "topk-sae.safetensors").stat().st_mtime_ns}
    manifest = destination / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != revision:
        raise ValueError(f"Existing encodings use different weights: {manifest}")
    manifest.write_text(json.dumps(revision, indent=2) + "\n")
    paths = activation_shards(args.work_dir / "activations" / args.split, args.max_shards)
    for i, path in enumerate(paths):
        cached = destination / path.name
        if cached.exists():
            continue
        with np.load(path, allow_pickle=False) as shard:
            rows = shard["residuals"]
            tokens, documents = shard["tokens"], shard["documents"]
        pointers, values, indices = [], [], []
        for start in range(0, len(rows), 1024):
            batch = torch.from_numpy(rows[start:start + 1024]).to(device, dtype=torch.float32)
            scores, ids = sae.sparse_code(sae.normalized(batch))
            scores, ids = scores.cpu().numpy(), ids.cpu().numpy()
            active = scores > 1e-6
            pointers.append(active.sum(axis=1))
            values.append(scores[active])
            indices.append(ids[active].astype(np.uint16))
        indptr = np.r_[0, np.cumsum(np.concatenate(pointers), dtype=np.int32)]
        tmp = cached.with_suffix(".tmp.npz")
        np.savez(tmp, tokens=tokens, documents=documents, indptr=indptr,
                 values=np.concatenate(values), indices=np.concatenate(indices))
        tmp.replace(cached)
        print(f"Encoded {args.output_split} {i + 1}/{len(paths)}", flush=True)


def group(args: argparse.Namespace) -> None:
    weights = load_file(args.model / "topk-sae.safetensors")
    decoder = weights["decoder"].numpy() * float(weights["scale"])
    np.save(args.model / "decoder.npy", decoder)
    if args.native_blocks:
        block_size = json.loads((args.model / "config.json").read_text()).get("block_size", 1)
        if block_size < 2:
            raise ValueError("Native groups require block-sparse training")
        groups = {i // block_size: list(range(i, i + block_size)) for i in range(0, len(decoder), block_size)}
    else:
        with threadpool_limits(limits=4):
            affinity = kneighbors_graph(decoder, n_neighbors=24, metric="cosine", mode="distance", include_self=False)
            affinity.data = np.maximum(1 - affinity.data, 1e-4)
            affinity = affinity.maximum(affinity.T)
            clustering = SpectralClustering(
                n_clusters=args.groups, affinity="precomputed", assign_labels="cluster_qr", random_state=20261125,
            ).fit_predict(affinity)
        groups = {i: np.flatnonzero(clustering == i).tolist() for i in range(args.groups)}
    (args.model / "groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    print(f"Created {len(groups)} unlabeled decoder groups", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".experiment"))
    cmd = parser.add_subparsers(dest="command", required=True)
    fit = cmd.add_parser("train")
    fit.add_argument("--fit", default="pile-fit")
    fit.add_argument("--validation", default="pile-holdout")
    fit.add_argument("--fit-shards", type=int, default=0)
    fit.add_argument("--validation-shards", type=int, default=0)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--width", type=int, default=3072)
    fit.add_argument("--k", type=int, default=32)
    fit.add_argument("--block-size", type=int, default=1)
    fit.add_argument("--seed", type=int, default=20261125)
    fit.add_argument("--steps", type=int, default=4500)
    fit.add_argument("--checkpoint-every", type=int, default=500)
    fit.add_argument("--batch-size", type=int, default=1024)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--loss", choices=LOSS_POWERS, default="mse")
    fit.add_argument("--resume-lr", type=float)
    fit.add_argument("--clip-grad", type=float, default=0.0)
    encode = cmd.add_parser("encode")
    encode.add_argument("--model", type=Path, required=True)
    encode.add_argument("--split", required=True)
    encode.add_argument("--output-split", required=True)
    encode.add_argument("--max-shards", type=int, default=0)
    cluster = cmd.add_parser("group")
    cluster.add_argument("--model", type=Path, required=True)
    cluster.add_argument("--groups", type=int, default=128)
    cluster.add_argument("--native-blocks", action="store_true")
    for command in (fit, encode):
        command.add_argument("--device", default="auto")
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "train" and (
        not (1 <= args.block_size <= args.k <= args.width <= 65_536)
        or args.width % args.block_size or args.k % args.block_size
        or min(args.steps, args.checkpoint_every, args.batch_size) < 1
        or args.lr <= 0 or args.clip_grad < 0 or (args.resume_lr is not None and args.resume_lr <= 0)
    ):
        parser.error("Width and k must be divisible by block size. Training sizes and steps must be positive.")
    actions = {"train": train, "encode": recode, "group": group}
    actions[args.command](args)


if __name__ == "__main__":
    main()
