from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.data.dataset import PretrainingDataset, singleton_collate
from src.models.model import GeometryLagEEG, parameter_report
from src.training.losses import masked_reconstruction_loss


def learning_rate(step: int, total: int, peak: float, warmup_fraction: float) -> float:
    warmup = max(1, round(total * warmup_fraction))
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def initialize_module(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
        if module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def synchronize(device: torch.device) -> None:
    token = torch.ones(1, device=device)
    dist.all_reduce(token)
    torch.cuda.synchronize(device)


def save_checkpoint(path: Path, model, optimizer, step: int, config: dict, best: float | None, consumed: int):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "best_validation": best,
        "samples_per_rank_consumed": consumed,
    }, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(model, manifest: str, seed: int, rank: int, device: torch.device, batches: int, start: int):
    dataset = PretrainingDataset(manifest, "validation", seed, rank, start_index=start)
    iterator = iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=singleton_collate))
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for _ in range(batches):
        batch = next(iterator)
        patches = batch["patches"].to(device)
        valid = batch["valid"].to(device)
        masked = batch["masked"].to(device)
        template = batch["template"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(patches, masked, valid, template)
            values = masked_reconstruction_loss(prediction, patches, masked, valid)
        totals += torch.tensor([float(value) for value in values], dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    model.train()
    totals /= batches * dist.get_world_size()
    return {"loss": float(totals[0]), "waveform_loss": float(totals[1]), "spectral_loss": float(totals[2])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry-Lag EEG masked-reconstruction pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg, data_cfg = config["training"], config["data"]
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(int(train_cfg["seed"]) + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    synchronize(device)

    with torch.device("meta"):
        model = GeometryLagEEG(bool(config["model"]["activation_checkpointing"]))
    model.to_empty(device=device)
    model.apply(initialize_module)
    torch.nn.init.normal_(model.tokenizer.mask_embedding, std=0.02)
    report = parameter_report(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                gradient_as_bucket_view=True, broadcast_buffers=False, bucket_cap_mb=8)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0, betas=(0.9, 0.95), weight_decay=float(train_cfg["weight_decay"])
    )

    start_step, consumed, best_validation, stale_validations = 0, 0, None, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        consumed = int(checkpoint.get("samples_per_rank_consumed", start_step * train_cfg["gradient_accumulation"]))
        best_validation = checkpoint.get("best_validation")

    dataset = PretrainingDataset(args.train_manifest, "train", int(train_cfg["seed"]), rank, consumed)
    loader = DataLoader(dataset, batch_size=1, num_workers=int(data_cfg["num_workers"]),
                        collate_fn=singleton_collate, pin_memory=True)
    iterator = iter(loader)
    if rank == 0:
        run_config = {"world_size": world, "model": report, "config": config}
        (output / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    total_steps = int(train_cfg["steps"])
    accumulation = int(train_cfg["gradient_accumulation"])
    optimizer.zero_grad(set_to_none=True)
    counts, started = Counter(), time.perf_counter()
    completed_step = start_step
    for step in range(start_step + 1, total_steps + 1):
        completed_step = step
        lr = learning_rate(step, total_steps, float(train_cfg["peak_learning_rate"]),
                           float(train_cfg["warmup_fraction"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        sums = torch.zeros(3, dtype=torch.float64, device=device)
        for micro in range(accumulation):
            batch = next(iterator)
            consumed += 1
            counts.update([batch["dataset"], batch["mask_kind"]])
            patches = batch["patches"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            masked = batch["masked"].to(device, non_blocking=True)
            template = batch["template"].to(device)
            sync_context = contextlib.nullcontext() if micro == accumulation - 1 else model.no_sync()
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(patches, masked, valid, template)
                values = masked_reconstruction_loss(prediction, patches, masked, valid)
                (values[0] / accumulation).backward()
            sums += torch.tensor([float(v.detach()) for v in values], dtype=torch.float64, device=device)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip"]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % int(train_cfg["log_every"]) == 0:
            dist.all_reduce(sums)
            record = {
                "step": step,
                "loss": float(sums[0] / (accumulation * world)),
                "waveform_loss": float(sums[1] / (accumulation * world)),
                "spectral_loss": float(sums[2] / (accumulation * world)),
                "learning_rate": lr,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if rank == 0:
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)

        validation_every = int(train_cfg["validation_every"])
        should_validate = validation_every > 0 and step % validation_every == 0
        if should_validate:
            metrics = validate(
                model, args.validation_manifest, int(train_cfg["seed"]) + 99173,
                rank, device, int(train_cfg["validation_batches"]),
                (step // validation_every - 1) * int(train_cfg["validation_batches"]),
            )
            improved = best_validation is None or metrics["loss"] < best_validation
            best_validation = metrics["loss"] if improved else best_validation
            stale_validations = 0 if improved else stale_validations + 1
            if rank == 0:
                metrics["step"] = step
                with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics) + "\n")
            if improved:
                synchronize(device)
                if rank == 0:
                    save_checkpoint(output / "best_validation.pt", model, optimizer, step, config, best_validation, consumed)
                synchronize(device)

        save_every = int(train_cfg["save_every"])
        if save_every > 0 and step % save_every == 0:
            synchronize(device)
            if rank == 0:
                save_checkpoint(output / "latest.pt", model, optimizer, step, config, best_validation, consumed)
            synchronize(device)

        patience = int(train_cfg["early_stopping_patience"])
        if should_validate and patience > 0 and stale_validations >= patience:
            break

    synchronize(device)
    if rank == 0:
        save_checkpoint(output / "latest.pt", model, optimizer, completed_step, config, best_validation, consumed)
        summary = {"completed_step": completed_step, "best_validation": best_validation, "counts": dict(counts)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    synchronize(device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
