#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged benchmark progression (A/B/C).")
    parser.add_argument("--stage", choices=["a", "b", "c", "all"], default="all")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint_last.pt")
    parser.add_argument("--output-root", type=str, default="outputs/stages")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage-b-d", type=int, choices=[2, 4], default=4)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def stage_a(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_root) / "stage_a"
    run(
        [
            sys.executable,
            "scripts/evaluate_interventions.py",
            "--checkpoint",
            args.checkpoint,
            "--preset",
            "stage_a",
            "--output-dir",
            str(out_dir),
            "--num-tasks",
            str(args.num_tasks),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--device",
            args.device,
            "--methods",
            "tgmm",
            "em",
            "kmeans",
            "spectral",
        ]
    )


def stage_b(args: argparse.Namespace) -> None:
    preset = "stage_b_d4" if args.stage_b_d == 4 else "stage_b_d2"
    train_out = Path(args.output_root) / f"stage_b_{preset}"
    eval_out = train_out / "eval"

    run(
        [
            sys.executable,
            "scripts/train_tgmm.py",
            "--preset",
            preset,
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            str(train_out),
            "--device",
            args.device,
        ]
    )

    run(
        [
            sys.executable,
            "scripts/evaluate_interventions.py",
            "--checkpoint",
            str(train_out / "checkpoint_last.pt"),
            "--output-dir",
            str(eval_out),
            "--num-tasks",
            str(args.num_tasks),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--methods",
            "tgmm",
            "em",
            "kmeans",
            "spectral",
            "--mechanism-compressions",
            "0.65 0.5 0.35",
            "--sample-sizes",
            "8 16",
            "--dirichlet-alphas",
            "0.2",
            "--device",
            args.device,
        ]
    )


def stage_c(args: argparse.Namespace) -> None:
    # Current TGMM implementation is fixed-K per checkpoint, so Stage C is
    # approximated by training/evaluating a family of checkpoints for K in {2,3,4,5}.
    for k in [2, 3, 4, 5]:
        train_out = Path(args.output_root) / f"stage_c_k{k}"
        eval_out = train_out / "eval"

        run(
            [
                sys.executable,
                "scripts/train_tgmm.py",
                "--k",
                str(k),
                "--d",
                "4",
                "--n-min",
                "16",
                "--n-max",
                "32",
                "--min-separation",
                "1.2",
                "--dirichlet-alpha",
                "2.0",
                "--min-weight",
                "0.02",
                "--predict-scales",
                "--anisotropic-prob",
                "0.25",
                "--steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
                "--output-dir",
                str(train_out),
                "--device",
                args.device,
            ]
        )

        run(
            [
                sys.executable,
                "scripts/evaluate_interventions.py",
                "--checkpoint",
                str(train_out / "checkpoint_last.pt"),
                "--output-dir",
                str(eval_out),
                "--num-tasks",
                str(args.num_tasks),
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--methods",
                "tgmm",
                "em",
                "kmeans",
                "spectral",
                "--mechanism-compressions",
                "0.85 0.7 0.5 0.35",
                "--sample-size-multipliers",
                "0.5 1.0 2.0",
                "--dirichlet-alphas",
                "0.2 0.5 1.0",
                "--noise-factors",
                "1.25 1.5 2.0",
                "--include-anisotropic-noise",
                "--device",
                args.device,
            ]
        )


def main() -> None:
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    if args.stage in {"a", "all"}:
        stage_a(args)
    if args.stage in {"b", "all"}:
        stage_b(args)
    if args.stage in {"c", "all"}:
        stage_c(args)


if __name__ == "__main__":
    main()
