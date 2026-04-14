#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.config import ModelConfig, SamplingConfig
from interventional_tgmm.config_io import apply_config_defaults
from interventional_tgmm.data import sample_task_batch
from interventional_tgmm.losses import permutation_invariant_loss
from interventional_tgmm.model import TGMMNet
from interventional_tgmm.tmp_utils import cleanup_tmp_dir, rewrite_tmp_relative_path
from interventional_tgmm.utils import choose_device, configure_torch_runtime, ensure_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small TGMM-style transformer.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--preset", type=str, default="none", choices=["none", "stage_b_d4", "stage_b_d2"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--use-tqdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-tmp", action="store_true")
    parser.add_argument("--tmp-dir", type=str, default="outputs/tmp")
    parser.add_argument("--dataset", type=str, default="wine")
    parser.add_argument("--dataset-path", type=str, default="wine.csv")
    parser.add_argument("--label-column", type=str, default="Cultivars")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--n-min", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mean-bound", type=float, default=4.0)
    parser.add_argument("--min-separation", type=float, default=3.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=4.0)
    parser.add_argument("--min-weight", type=float, default=0.12)
    parser.add_argument("--max-mean-resamples", type=int, default=200)
    parser.add_argument("--anisotropic-prob", type=float, default=0.0)
    parser.add_argument("--anisotropic-log-scale-min", type=float, default=-1.0)
    parser.add_argument("--anisotropic-log-scale-max", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--predict-scales", action="store_true")
    parser.add_argument("--scale-loss-weight", type=float, default=1.0)
    parser.add_argument("--min-scale", type=float, default=1e-3)

    pre_args, _ = parser.parse_known_args()
    apply_config_defaults(parser, pre_args.config, sections=["dataset", "runtime", "train"])
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "none":
        return
    if args.preset == "stage_b_d4":
        args.k = 3
        args.d = 4
        args.n_min = 16
        args.n_max = 32
        args.min_separation = 1.5
        args.dirichlet_alpha = 2.0
        args.min_weight = 0.03
        return
    if args.preset == "stage_b_d2":
        args.k = 3
        args.d = 2
        args.n_min = 16
        args.n_max = 32
        args.min_separation = 1.0
        args.dirichlet_alpha = 2.0
        args.min_weight = 0.03
        return
    raise ValueError(f"Unknown preset: {args.preset}")


def main() -> None:
    args = parse_args()
    args.output_dir = rewrite_tmp_relative_path(args.output_dir, args.tmp_dir)
    try:
        apply_preset(args)
        configure_torch_runtime(args.num_threads)
        seed_everything(args.seed)
        device = choose_device(args.device)
        out_dir = ensure_dir(args.output_dir)

        data_cfg = SamplingConfig(
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            label_column=args.label_column,
            k=args.k,
            d=args.d,
            n_min=args.n_min,
            n_max=args.n_max,
            sigma=args.sigma,
            mean_bound=args.mean_bound,
            min_separation=args.min_separation,
            dirichlet_alpha=args.dirichlet_alpha,
            min_weight=args.min_weight,
            max_mean_resamples=args.max_mean_resamples,
            anisotropic_prob=args.anisotropic_prob,
            anisotropic_log_scale_min=args.anisotropic_log_scale_min,
            anisotropic_log_scale_max=args.anisotropic_log_scale_max,
        )
        model_cfg = ModelConfig(
            d=args.d,
            k=args.k,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            predict_scales=args.predict_scales,
            min_scale=args.min_scale,
        )

        model = TGMMNet(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        numpy_rng = np.random.default_rng(args.seed)

        history: list[dict[str, float | int]] = []
        best_loss = float("inf")

        progress = tqdm(
            range(1, args.steps + 1),
            total=args.steps,
            desc="train",
            disable=not args.use_tqdm,
        )
        for step in progress:
            batch = sample_task_batch(data_cfg, batch_size=args.batch_size, rng=numpy_rng)
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            true_means = batch["true_means"].to(device)
            true_weights = batch["true_weights"].to(device)
            true_scales = batch["true_scales"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, mask)
            loss_dict = permutation_invariant_loss(
                pred_means=outputs["means"],
                pred_weight_logits=outputs["weight_logits"],
                true_means=true_means,
                true_weights=true_weights,
                pred_log_scales=outputs.get("log_scales"),
                true_scales=true_scales,
                scale_loss_weight=args.scale_loss_weight,
                min_scale=args.min_scale,
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
                "scale_loss": float(loss_dict["scale_loss"].item()),
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
                progress.set_postfix(
                    loss=f"{record['loss']:.4f}",
                    mean=f"{record['mean_loss']:.4f}",
                    pi=f"{record['pi_loss']:.4f}",
                    scale=f"{record['scale_loss']:.4f}",
                    refresh=False,
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
    finally:
        cleanup_tmp_dir(args.tmp_dir, args.show_tmp)


if __name__ == "__main__":
    main()
