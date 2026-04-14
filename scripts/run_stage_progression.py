#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import yaml
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]


def _load_config_defaults(config_path: str) -> dict[str, object]:
    path = Path(config_path)
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file '{path}' must contain a mapping.")
    defaults: dict[str, object] = {}
    for section_name in ["runtime", "stage_progression"]:
        section = payload.get(section_name, {})
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"Config section '{section_name}' must be a mapping.")
        defaults.update({str(key): value for key, value in section.items()})
    return defaults


def _rewrite_tmp_relative_path(path_value: str, tmp_dir: str) -> str:
    path = Path(path_value)
    if path.is_absolute() or not path.parts:
        return path_value
    if path.parts[0].startswith("tmp"):
        return str(Path(tmp_dir) / path)
    return path_value


def _cleanup_tmp_dir(tmp_dir: str, show_tmp: bool) -> None:
    if show_tmp:
        return
    path = Path(tmp_dir)
    if path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged benchmark progression (A/B/C).")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--stage", choices=["a", "b", "c", "all"], default="all")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint_last.pt")
    parser.add_argument("--output-root", type=str, default="outputs/stages")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-tqdm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-tmp", action="store_true")
    parser.add_argument("--tmp-dir", type=str, default="outputs/tmp")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage-b-d", type=int, choices=[2, 4], default=4)
    pre_args, _ = parser.parse_known_args()
    defaults = _load_config_defaults(pre_args.config)
    if defaults:
        valid_keys = {action.dest for action in parser._actions}
        parser.set_defaults(**{key: value for key, value in defaults.items() if key in valid_keys})
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def with_runtime_flags(cmd: list[str], args: argparse.Namespace) -> list[str]:
    merged = [*cmd, "--config", args.config, "--tmp-dir", args.tmp_dir]
    if args.show_tmp:
        merged.append("--show-tmp")
    return merged


def stage_a(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_root) / "stage_a"
    run(
        with_runtime_flags([
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
        ], args)
    )


def stage_b(args: argparse.Namespace) -> None:
    preset = "stage_b_d4" if args.stage_b_d == 4 else "stage_b_d2"
    train_out = Path(args.output_root) / f"stage_b_{preset}"
    eval_out = train_out / "eval"

    run(
        with_runtime_flags([
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
        ], args)
    )

    run(
        with_runtime_flags([
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
        ], args)
    )


def stage_c(args: argparse.Namespace) -> None:
    # Wine-backed setup is fixed to K=3 classes, so Stage C keeps K fixed and
    # focuses on harder interventions and anisotropic extensions.
    for k in [3]:
        train_out = Path(args.output_root) / f"stage_c_k{k}"
        eval_out = train_out / "eval"

        run(
            with_runtime_flags([
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
            ], args)
        )

        run(
            with_runtime_flags([
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
            ], args)
        )


def main() -> None:
    args = parse_args()
    args.checkpoint = _rewrite_tmp_relative_path(args.checkpoint, args.tmp_dir)
    args.output_root = _rewrite_tmp_relative_path(args.output_root, args.tmp_dir)
    try:
        Path(args.output_root).mkdir(parents=True, exist_ok=True)
        selected_stages: list[tuple[str, Callable[[argparse.Namespace], None]]] = []
        if args.stage in {"a", "all"}:
            selected_stages.append(("a", stage_a))
        if args.stage in {"b", "all"}:
            selected_stages.append(("b", stage_b))
        if args.stage in {"c", "all"}:
            selected_stages.append(("c", stage_c))

        for stage_name, stage_fn in tqdm(selected_stages, desc="stages", disable=not args.use_tqdm):
            stage_fn(args)
    finally:
        _cleanup_tmp_dir(args.tmp_dir, args.show_tmp)


if __name__ == "__main__":
    main()
