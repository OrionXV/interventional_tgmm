#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.clean_benchmark import SplitTaskConfig, load_split_benchmark, sample_split_task, task_hash
from interventional_tgmm.config import InterventionConfig
from interventional_tgmm.tmp_utils import cleanup_tmp_dir, rewrite_tmp_relative_path
from interventional_tgmm.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exact task overlap between train and eval streams.")
    parser.add_argument("--output-dir", type=str, default="outputs_clean/overlap_audit")
    parser.add_argument("--show-tmp", action="store_true")
    parser.add_argument("--tmp-dir", type=str, default="outputs/tmp")
    parser.add_argument("--dataset", type=str, choices=["wine", "iris", "digits"], default="wine")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--label-column", type=str, default="")
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--generator", type=str, choices=["residual", "gaussian_diag"], default="residual")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--n-min", type=int, default=32)
    parser.add_argument("--n-max", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=4.0)
    parser.add_argument("--min-weight", type=float, default=0.12)
    parser.add_argument("--train-split", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--eval-split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--num-train-tasks", type=int, default=200)
    parser.add_argument("--num-eval-tasks", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = rewrite_tmp_relative_path(args.output_dir, args.tmp_dir)
    try:
        out_dir = ensure_dir(args.output_dir)
        task_cfg = SplitTaskConfig(
            dataset=args.dataset,
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
            generator=args.generator,
        )
        benchmark = load_split_benchmark(
            dataset=task_cfg.dataset,
            dataset_path=task_cfg.dataset_path,
            label_column=task_cfg.label_column,
            d=task_cfg.d,
            k=task_cfg.k,
            split_seed=task_cfg.split_seed,
            train_fraction=task_cfg.train_fraction,
            fit_preprocessor_on_train=task_cfg.fit_preprocessor_on_train,
        )
        intervention = InterventionConfig(kind="none")

        train_rng = np.random.default_rng(args.train_seed)
        train_hashes: dict[str, dict[str, object]] = {}
        for idx in range(args.num_train_tasks):
            task = sample_split_task(task_cfg, benchmark, split=args.train_split, rng=train_rng, intervention=intervention)
            train_hashes[task_hash(task)] = {"seed": args.train_seed, "task_idx": idx}

        overlaps: list[dict[str, object]] = []
        eval_total = 0
        for eval_seed in args.eval_seeds:
            eval_rng = np.random.default_rng(eval_seed)
            for idx in range(args.num_eval_tasks):
                task = sample_split_task(task_cfg, benchmark, split=args.eval_split, rng=eval_rng, intervention=intervention)
                digest = task_hash(task)
                eval_total += 1
                if digest in train_hashes:
                    overlaps.append(
                        {
                            "hash": digest,
                            "eval_seed": eval_seed,
                            "eval_task_idx": idx,
                            "train_seed": train_hashes[digest]["seed"],
                            "train_task_idx": train_hashes[digest]["task_idx"],
                        }
                    )

        report = {
            "dataset": benchmark.dataset_name,
            "dataset_path": benchmark.dataset_path,
            "label_column": benchmark.label_column,
            "generator": args.generator,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "train_seed": args.train_seed,
            "eval_seeds": args.eval_seeds,
            "num_train_tasks": args.num_train_tasks,
            "num_eval_tasks_per_seed": args.num_eval_tasks,
            "num_eval_tasks_total": eval_total,
            "num_exact_hash_overlaps": len(overlaps),
            "overlap_rate": float(len(overlaps) / max(1, eval_total)),
            "overlaps": overlaps[:20],
        }
        save_json(out_dir / "overlap_report.json", report)

        md = ["# Task-overlap audit", ""]
        md.append(f"- Dataset: {benchmark.dataset_name} ({benchmark.dataset_path})")
        md.append(f"- Generator: {args.generator}")
        md.append(f"- Train split / eval split: {args.train_split} / {args.eval_split}")
        md.append(f"- Train seed: {args.train_seed}; eval seeds: {args.eval_seeds}")
        md.append(f"- Exact hash overlaps: {len(overlaps)} / {eval_total} ({100.0 * report['overlap_rate']:.2f}%)")
        (out_dir / "overlap_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"Saved overlap audit to {out_dir}")
    finally:
        cleanup_tmp_dir(args.tmp_dir, args.show_tmp)


if __name__ == "__main__":
    main()
