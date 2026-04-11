#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.config import ModelConfig, SamplingConfig
from interventional_tgmm.data import sample_task_batch
from interventional_tgmm.losses import permutation_invariant_loss
from interventional_tgmm.model import TGMMNet
from interventional_tgmm.utils import choose_device, configure_torch_runtime, ensure_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small TGMM-style transformer.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--n-min", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch_runtime(args.num_threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    data_cfg = SamplingConfig(k=args.k, d=args.d, n_min=args.n_min, n_max=args.n_max, sigma=args.sigma)
    model_cfg = ModelConfig(
        d=args.d,
        k=args.k,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )

    model = TGMMNet(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator(device="cpu")
    numpy_rng = __import__("numpy").random.default_rng(args.seed)

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")

    for step in range(1, args.steps + 1):
        batch = sample_task_batch(data_cfg, batch_size=args.batch_size, rng=numpy_rng)
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        true_means = batch["true_means"].to(device)
        true_weights = batch["true_weights"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, mask)
        loss_dict = permutation_invariant_loss(
            pred_means=outputs["means"],
            pred_weight_logits=outputs["weight_logits"],
            true_means=true_means,
            true_weights=true_weights,
        )
        loss = loss_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        record = {
            "step": step,
            "loss": float(loss_dict["loss"].item()),
            "mean_loss": float(loss_dict["mean_loss"].item()),
            "pi_loss": float(loss_dict["pi_loss"].item()),
        }
        history.append(record)

        if record["loss"] < best_loss:
            best_loss = record["loss"]
            checkpoint = {
                "model_state": model.state_dict(),
                "model_config": model_cfg.to_dict(),
                "sampling_config": data_cfg.to_dict(),
                "train_args": vars(args),
                "best_loss": best_loss,
                "step": step,
            }
            torch.save(checkpoint, out_dir / "checkpoint_best.pt")

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            print(
                f"step={step:04d} "
                f"loss={record['loss']:.4f} "
                f"mean_loss={record['mean_loss']:.4f} "
                f"pi_loss={record['pi_loss']:.4f}"
            )

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "sampling_config": data_cfg.to_dict(),
        "train_args": vars(args),
        "best_loss": best_loss,
        "step": args.steps,
    }
    torch.save(checkpoint, out_dir / "checkpoint_last.pt")
    save_json(out_dir / "train_history.json", history)
    save_json(
        out_dir / "config.json",
        {"model_config": model_cfg.to_dict(), "sampling_config": data_cfg.to_dict(), "train_args": vars(args)},
    )
    print(f"Saved checkpoints and logs to {out_dir}")


if __name__ == "__main__":
    main()
