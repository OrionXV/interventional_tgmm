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

from interventional_tgmm.clean_benchmark import SplitTaskConfig, load_wine_split_benchmark, sample_split_task_batch
from interventional_tgmm.config import ModelConfig
from interventional_tgmm.config_io import apply_config_defaults
from interventional_tgmm.losses import permutation_invariant_loss
from interventional_tgmm.model import TGMMNet
from interventional_tgmm.tmp_utils import cleanup_tmp_dir, rewrite_tmp_relative_path
from interventional_tgmm.utils import choose_device, configure_torch_runtime, ensure_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TGMM on a held-out split benchmark for cleaner claims.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs_clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--use-tqdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-tmp", action="store_true")
    parser.add_argument("--tmp-dir", type=str, default="outputs/tmp")

    # Benchmark settings
    parser.add_argument("--dataset-path", type=str, default="wine.csv")
    parser.add_argument("--label-column", type=str, default="Cultivars")
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--train-split", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--fit-preprocessor-on-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generator", type=str, choices=["residual", "gaussian_diag"], default="residual")

    # Task distribution
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--n-min", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=4.0)
    parser.add_argument("--min-weight", type=float, default=0.12)

    # Model
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--predict-scales", action="store_true")
    parser.add_argument("--scale-loss-weight", type=float, default=1.0)
    parser.add_argument("--min-scale", type=float, default=1e-3)

    pre_args, _ = parser.parse_known_args()
    apply_config_defaults(parser, pre_args.config, sections=["runtime"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = rewrite_tmp_relative_path(args.output_dir, args.tmp_dir)
    try:
        configure_torch_runtime(args.num_threads)
        seed_everything(args.seed)
        device = choose_device(args.device)
        out_dir = ensure_dir(args.output_dir)

        task_cfg = SplitTaskConfig(
            dataset_path=args.dataset_path,
            label_column=args.label_column,
            k=args.k,
            d=args.d,
            n_min=args.n_min,
            n_max=args.n_max,
            sigma=args.sigma,
            dirichlet_alpha=args.dirichlet_alpha,
            min_weight=args.min_weight,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            fit_preprocessor_on_train=args.fit_preprocessor_on_train,
            generator=args.generator,
        )
        benchmark = load_wine_split_benchmark(
            dataset_path=task_cfg.dataset_path,
            label_column=task_cfg.label_column,
            d=task_cfg.d,
            k=task_cfg.k,
            split_seed=task_cfg.split_seed,
            train_fraction=task_cfg.train_fraction,
            fit_preprocessor_on_train=task_cfg.fit_preprocessor_on_train,
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
            desc="train_clean",
            disable=not args.use_tqdm,
        )
        for step in progress:
            batch = sample_split_task_batch(
                cfg=task_cfg,
                benchmark=benchmark,
                split=args.train_split,
                batch_size=args.batch_size,
                rng=numpy_rng,
            )
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
                    "clean_task_config": task_cfg.to_dict(),
                    "clean_benchmark": {
                        "split_seed": args.split_seed,
                        "train_fraction": args.train_fraction,
                        "train_split": args.train_split,
                        "fit_preprocessor_on_train": args.fit_preprocessor_on_train,
                        "explained_variance_ratio": benchmark.explained_variance_ratio,
                    },
                    "train_args": vars(args),
                    "best_loss": best_loss,
                    "step": step,
                }
                torch.save(checkpoint, out_dir / "checkpoint_best.pt")

            if step % args.eval_every == 0 or step in {1, args.steps}:
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
            "clean_task_config": task_cfg.to_dict(),
            "clean_benchmark": {
                "split_seed": args.split_seed,
                "train_fraction": args.train_fraction,
                "train_split": args.train_split,
                "fit_preprocessor_on_train": args.fit_preprocessor_on_train,
                "explained_variance_ratio": benchmark.explained_variance_ratio,
            },
            "train_args": vars(args),
            "best_loss": best_loss,
            "step": args.steps,
        }
        torch.save(checkpoint, out_dir / "checkpoint_last.pt")
        save_json(out_dir / "train_history.json", history)
        save_json(
            out_dir / "config.json",
            {
                "model_config": model_cfg.to_dict(),
                "clean_task_config": task_cfg.to_dict(),
                "clean_benchmark": checkpoint["clean_benchmark"],
                "train_args": vars(args),
            },
        )
        print(f"Saved clean-claim checkpoints and logs to {out_dir}")
    finally:
        cleanup_tmp_dir(args.tmp_dir, args.show_tmp)


if __name__ == "__main__":
    main()
