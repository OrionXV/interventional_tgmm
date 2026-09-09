#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clean-claims training/evaluation suite.")
    parser.add_argument("--output-root", type=str, default="outputs_clean_suite")
    parser.add_argument("--dataset", type=str, choices=["wine", "iris", "digits"], default="wine")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--label-column", type=str, default="")
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--generator", type=str, choices=["residual", "gaussian_diag"], default="residual")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--min-weight", type=float, default=0.12)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--show-tmp", action="store_true")
    parser.add_argument("--tmp-dir", type=str, default="outputs/tmp")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    train_out = output_root / "train"
    overlap_out = output_root / "overlap"
    eval_out = output_root / "eval"

    common = ["--show-tmp"] if args.show_tmp else []
    if not args.show_tmp:
        common += ["--tmp-dir", args.tmp_dir]

    run([
        sys.executable,
        "scripts/train_clean_tgmm.py",
        "--output-dir",
        str(train_out),
        "--dataset",
        args.dataset,
        "--dataset-path",
        args.dataset_path,
        "--label-column",
        args.label_column,
        "--split-seed",
        str(args.split_seed),
        "--train-fraction",
        str(args.train_fraction),
        "--generator",
        args.generator,
        "--k",
        str(args.k),
        "--d",
        str(args.d),
        "--min-weight",
        str(args.min_weight),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        *common,
    ])

    run([
        sys.executable,
        "scripts/check_task_overlap.py",
        "--output-dir",
        str(overlap_out),
        "--dataset",
        args.dataset,
        "--dataset-path",
        args.dataset_path,
        "--label-column",
        args.label_column,
        "--split-seed",
        str(args.split_seed),
        "--train-fraction",
        str(args.train_fraction),
        "--generator",
        args.generator,
        "--k",
        str(args.k),
        "--d",
        str(args.d),
        "--min-weight",
        str(args.min_weight),
        "--train-seed",
        str(args.seed),
        "--eval-seeds",
        *[str(seed) for seed in args.eval_seeds],
        *common,
    ])

    run([
        sys.executable,
        "scripts/evaluate_clean_claims.py",
        "--checkpoint",
        str(train_out / "checkpoint_last.pt"),
        "--output-dir",
        str(eval_out),
        "--device",
        args.device,
        "--num-tasks",
        str(args.num_tasks),
        "--seeds",
        *[str(seed) for seed in args.eval_seeds],
        *common,
    ])


if __name__ == "__main__":
    main()
