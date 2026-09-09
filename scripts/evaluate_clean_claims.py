#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.clean_baselines import fit_method
from interventional_tgmm.clean_benchmark import SplitTaskConfig, load_split_benchmark, sample_split_task
from interventional_tgmm.config import InterventionConfig, ModelConfig
from interventional_tgmm.config_io import apply_config_defaults
from interventional_tgmm.metrics import evaluate_task, summarize_metric_dicts
from interventional_tgmm.model import TGMMNet
from interventional_tgmm.tmp_utils import cleanup_tmp_dir, rewrite_tmp_relative_path
from interventional_tgmm.types import GMMEstimate
from interventional_tgmm.utils import choose_device, configure_torch_runtime, ensure_dir, save_json


@dataclass(slots=True)
class InterventionSpec:
    name: str
    family: str
    strength: float
    config: InterventionConfig


def parse_float_list(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(part) for part in raw.replace(",", " ").split()]


def parse_int_list(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(part) for part in raw.replace(",", " ").split()]


def format_float(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.4g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cleaner claims on held-out split tasks.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs_clean/eval")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--show-tmp", action="store_true")
    parser.add_argument("--tmp-dir", type=str, default="outputs/tmp")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    parser.add_argument("--methods", nargs="+", type=str, default=["tgmm", "em", "kmeans", "spectral", "diag_gmm"])
    parser.add_argument("--eval-split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--generators", nargs="+", choices=["residual", "gaussian_diag"], default=["residual", "gaussian_diag"])
    parser.add_argument("--em-restarts", type=int, default=10)
    parser.add_argument("--spectral-restarts", type=int, default=20)
    parser.add_argument("--curve-metric", type=str, default="parameter_error")

    parser.add_argument("--prior-strengths", type=str, default="1.25")
    parser.add_argument("--mechanism-compressions", type=str, default="0.65")
    parser.add_argument("--noise-factors", type=str, default="1.5")
    parser.add_argument("--sample-sizes", type=str, default="128")

    pre_args, _ = parser.parse_known_args()
    apply_config_defaults(parser, pre_args.config, sections=["runtime"])
    return parser.parse_args()


def load_model_and_task_cfg(checkpoint_path: Path, device: torch.device) -> tuple[TGMMNet, SplitTaskConfig, dict[str, Any], int | None]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "clean_task_config" not in ckpt:
        raise ValueError(
            "Checkpoint does not contain clean_task_config. Train with scripts/train_clean_tgmm.py first."
        )
    model_cfg = ModelConfig(**ckpt["model_config"])
    task_cfg = SplitTaskConfig(**ckpt["clean_task_config"])
    model = TGMMNet(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    train_seed = None
    train_args = ckpt.get("train_args") or {}
    if isinstance(train_args, dict) and "seed" in train_args:
        train_seed = int(train_args["seed"])
    return model, task_cfg, ckpt, train_seed


def model_estimate(model: TGMMNet, x: np.ndarray, device: torch.device, sigma: float) -> GMMEstimate:
    x_t = torch.from_numpy(x[None, ...]).float().to(device)
    mask = torch.ones((1, x.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        outputs = model(x_t, mask)
    means = outputs["means"].cpu().numpy()[0]
    weights = torch.softmax(outputs["weight_logits"], dim=-1).cpu().numpy()[0]
    sigma_diag = None
    if "log_scales" in outputs:
        sigma_diag = model.decode_scales(outputs["log_scales"]).cpu().numpy()[0]
    return GMMEstimate(
        weights=weights.astype(np.float32),
        means=means.astype(np.float32),
        assumed_sigma=float(sigma),
        sigma_diag=None if sigma_diag is None else sigma_diag.astype(np.float32),
    )


def build_interventions(args: argparse.Namespace) -> list[InterventionSpec]:
    interventions: list[InterventionSpec] = [
        InterventionSpec(name="none", family="none", strength=0.0, config=InterventionConfig(kind="none"))
    ]
    for strength in sorted(set(parse_float_list(args.prior_strengths))):
        interventions.append(
            InterventionSpec(
                name=f"prior@{format_float(strength)}",
                family="prior",
                strength=float(strength),
                config=InterventionConfig(kind="prior", target_component=0, prior_strength=float(strength)),
            )
        )
    for compression in sorted(set(parse_float_list(args.mechanism_compressions))):
        interventions.append(
            InterventionSpec(
                name=f"mechanism@{format_float(compression)}",
                family="mechanism",
                strength=float(1.0 - compression),
                config=InterventionConfig(kind="mechanism", mechanism_compression=float(compression)),
            )
        )
    for factor in sorted(set(parse_float_list(args.noise_factors))):
        interventions.append(
            InterventionSpec(
                name=f"noise@{format_float(factor)}",
                family="noise",
                strength=float(factor),
                config=InterventionConfig(kind="noise", noise_factor=float(factor), noise_type="scale"),
            )
        )
    for sample_size in sorted(set(parse_int_list(args.sample_sizes))):
        interventions.append(
            InterventionSpec(
                name=f"sample_size@{sample_size}",
                family="sample_size",
                strength=float(sample_size),
                config=InterventionConfig(kind="sample_size", sample_size=int(sample_size)),
            )
        )
    return interventions


def evaluate_method(
    method: str,
    task,
    model: TGMMNet,
    device: torch.device,
    seed: int,
    em_restarts: int,
    spectral_restarts: int,
) -> tuple[dict[str, float], bool, str | None]:
    try:
        if method == "tgmm":
            estimate = model_estimate(model, task.x, device, sigma=task.params.sigma)
            metrics = evaluate_task(task, estimate)
            return metrics, True, None
        baseline = fit_method(
            method=method,
            x=task.x,
            k=task.params.means.shape[0],
            sigma=task.params.sigma,
            seed=seed,
            em_restarts=em_restarts,
            spectral_restarts=spectral_restarts,
        )
        metrics = evaluate_task(task, baseline.estimate)
        if baseline.avg_log_likelihood is not None:
            metrics["avg_log_likelihood"] = float(baseline.avg_log_likelihood)
        return metrics, True, None
    except Exception as exc:  # noqa: BLE001
        return {
            "parameter_error": float("nan"),
            "mean_mse": float("nan"),
            "weight_mse": float("nan"),
            "scale_mse": float("nan"),
            "cluster_acc": float("nan"),
            "avg_log_likelihood": float("nan"),
        }, False, repr(exc)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    successes: dict[tuple[str, str, str], list[float]] = {}
    failures: dict[tuple[str, str, str], list[str]] = {}

    for record in records:
        key = (str(record["generator"]), str(record["method"]), str(record["intervention"]))
        grouped.setdefault(key, []).append(record["metrics"])
        successes.setdefault(key, []).append(1.0 if bool(record["success"]) else 0.0)
        if not bool(record["success"]) and record.get("error"):
            failures.setdefault(key, []).append(str(record["error"]))

    summary: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for (generator, method, intervention), metric_dicts in grouped.items():
        metric_summary = summarize_metric_dicts(metric_dicts)
        metric_summary["success_rate"] = {
            "mean": float(np.mean(successes[(generator, method, intervention)])),
            "std": 0.0,
        }
        if failures.get((generator, method, intervention)):
            metric_summary["sample_errors"] = {"examples": failures[(generator, method, intervention)][:3]}
        summary.setdefault(generator, {}).setdefault(method, {})[intervention] = metric_summary
    return summary


def build_main_table(summary: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> str:
    lines: list[str] = []
    for generator, gen_summary in summary.items():
        lines.append(f"## Generator: {generator}")
        lines.append("")
        lines.append("| Method | Intervention | Parameter Error | Cluster Acc | Avg Log-Likelihood | Success |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for method, method_summary in gen_summary.items():
            for intervention, metrics in method_summary.items():
                pe = metrics["parameter_error"]
                acc = metrics["cluster_acc"]
                ll = metrics["avg_log_likelihood"]
                success = metrics["success_rate"]
                lines.append(
                    "| "
                    f"{method} | {intervention} | "
                    f"{pe['mean']:.4f} +- {pe['std']:.4f} | "
                    f"{acc['mean']:.4f} +- {acc['std']:.4f} | "
                    f"{ll['mean']:.4f} +- {ll['std']:.4f} | "
                    f"{100.0 * success['mean']:.1f}% |"
                )
        lines.append("")
    return "\n".join(lines)


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _intervention_family_strength(name: str) -> tuple[str, float]:
    if name == "none":
        return "none", 0.0
    if "@" not in name:
        return name, float("inf")
    family, raw = name.split("@", 1)
    try:
        return family, float(raw)
    except ValueError:
        return family, float("inf")


def _ordered_interventions(interventions: list[str]) -> list[str]:
    family_order = {
        "none": 0,
        "prior": 1,
        "mechanism": 2,
        "noise": 3,
        "noise_diag": 4,
        "sample_size": 5,
        "dirichlet": 6,
    }

    def key_fn(name: str) -> tuple[int, float, str]:
        family, strength = _intervention_family_strength(name)
        return family_order.get(family, 99), strength, name

    return sorted(interventions, key=key_fn)


def _intervention_label(name: str, family_counts: dict[str, int]) -> str:
    family, strength = _intervention_family_strength(name)
    if family == "none":
        return "none"
    if family == "sample_size" and math.isfinite(strength):
        return f"n={int(round(strength))}"

    shorthand = {
        "prior": "prior",
        "mechanism": "mech",
        "noise": "noise",
        "noise_diag": "noise_diag",
        "dirichlet": "dirichlet",
    }
    short = shorthand.get(family, family)
    if family_counts.get(family, 0) <= 1:
        return short
    if math.isfinite(strength):
        return f"{short}@{strength:g}"
    return name


def write_method_comparison_bar_svg(
    path: Path,
    summary: dict[str, dict[str, dict[str, dict[str, Any]]]],
    methods: list[str],
) -> None:
    generators = [g for g in ["residual", "gaussian_diag"] if g in summary] or sorted(summary.keys())
    metrics = ["parameter_error", "cluster_acc"]
    if not generators:
        return

    panel_w = 420
    panel_h = 300
    cols = len(generators)
    rows = len(metrics)
    legend_h = 48
    width = panel_w * cols
    height = panel_h * rows + legend_h

    colors = {
        "tgmm": "#2C7FB8",
        "em": "#F28E2B",
        "kmeans": "#59A14F",
        "spectral": "#E15759",
        "diag_gmm": "#B07AA1",
    }
    fallback_palette = ["#2C7FB8", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2", "#EDC948", "#9C755F"]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#f6f7f9"/>',
        '<text x="16" y="24" font-size="16" fill="#1f1f1f">Method Comparison Across Shift Tasks (Bar Charts)</text>',
    ]

    legend_x = 16
    legend_y = 42
    for idx, method in enumerate(methods):
        color = colors.get(method, fallback_palette[idx % len(fallback_palette)])
        x0 = legend_x + idx * 120
        svg.append(f'<rect x="{x0}" y="{legend_y - 10}" width="10" height="10" fill="{color}" rx="1.5"/>')
        svg.append(f'<text x="{x0 + 14}" y="{legend_y - 1}" font-size="11" fill="#333333">{xml_escape(method)}</text>')

    for row_idx, metric in enumerate(metrics):
        for col_idx, generator in enumerate(generators):
            gen_summary = summary.get(generator, {})
            interventions_set: set[str] = set()
            for method in methods:
                interventions_set.update(gen_summary.get(method, {}).keys())
            if not interventions_set:
                continue

            interventions = _ordered_interventions(list(interventions_set))
            family_counts: dict[str, int] = {}
            for name in interventions:
                family, _ = _intervention_family_strength(name)
                family_counts[family] = family_counts.get(family, 0) + 1
            labels = [_intervention_label(name, family_counts) for name in interventions]

            all_values: list[float] = []
            for method in methods:
                for intervention in interventions:
                    stats = gen_summary.get(method, {}).get(intervention, {}).get(metric)
                    if stats is None:
                        continue
                    value = float(stats.get("mean", float("nan")))
                    if math.isfinite(value):
                        all_values.append(value)
            if not all_values:
                continue

            use_log = metric == "parameter_error" and all(value > 0.0 for value in all_values)
            if use_log:
                plot_values = [math.log10(value) for value in all_values]
            else:
                plot_values = all_values

            y_min = min(plot_values)
            y_max = max(plot_values)
            if abs(y_max - y_min) < 1e-9:
                y_min -= 1.0
                y_max += 1.0
            else:
                pad = 0.1 * (y_max - y_min)
                y_min -= pad
                y_max += pad

            ox = col_idx * panel_w
            oy = legend_h + row_idx * panel_h
            left = ox + 70
            right = ox + panel_w - 24
            top = oy + 38
            bottom = oy + panel_h - 58

            title_metric = "parameter error" if metric == "parameter_error" else "cluster accuracy"
            svg.append(f'<text x="{ox + 16}" y="{oy + 22}" font-size="14" fill="#202020">Held-out {xml_escape(generator)} generator: {xml_escape(title_metric)}</text>')
            svg.append(
                f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="#ffffff" stroke="#d1d5db"/>'
            )

            for grid_idx in range(5):
                y = top + grid_idx * (bottom - top) / 4.0
                svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')

            n_groups = max(1, len(interventions))
            group_w = (right - left) / n_groups
            inner_w = group_w * 0.82
            n_methods = max(1, len(methods))
            bar_w = inner_w / n_methods

            def y_map(raw_value: float) -> float:
                yv = math.log10(raw_value) if use_log else raw_value
                return bottom - (yv - y_min) / (y_max - y_min) * (bottom - top)

            for g_idx, intervention in enumerate(interventions):
                gx_left = left + g_idx * group_w + 0.5 * (group_w - inner_w)
                x_tick = left + (g_idx + 0.5) * group_w
                tick_label = labels[g_idx]
                svg.append(
                    f'<text x="{x_tick:.2f}" y="{bottom + 16}" font-size="10" fill="#4b5563" text-anchor="middle">{xml_escape(tick_label)}</text>'
                )

                for m_idx, method in enumerate(methods):
                    stats = gen_summary.get(method, {}).get(intervention, {}).get(metric)
                    if stats is None:
                        continue
                    value = float(stats.get("mean", float("nan")))
                    if not math.isfinite(value):
                        continue
                    if use_log and value <= 0.0:
                        continue
                    x0 = gx_left + m_idx * bar_w
                    y0 = y_map(value)
                    color = colors.get(method, fallback_palette[m_idx % len(fallback_palette)])
                    h = max(0.0, bottom - y0)
                    svg.append(
                        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_w * 0.92:.2f}" height="{h:.2f}" fill="{color}" opacity="0.92"/>'
                    )

            y_mid = (top + bottom) / 2.0
            y_label = "PE (lower is better)" if metric == "parameter_error" else "Accuracy"
            svg.append(
                f'<text x="{left - 50}" y="{y_mid:.2f}" font-size="11" fill="#374151" transform="rotate(-90 {left - 50} {y_mid:.2f})" text-anchor="middle">{xml_escape(y_label)}</text>'
            )
            svg.append(f'<text x="{(left + right)/2:.2f}" y="{bottom + 34}" font-size="11" fill="#374151" text-anchor="middle">Intervention</text>')

            if use_log:
                yt_top_label = f"{10**y_max:.2g}"
                yt_bot_label = f"{10**y_min:.2g}"
            else:
                yt_top_label = f"{y_max:.3f}"
                yt_bot_label = f"{y_min:.3f}"
            svg.append(f'<text x="{left - 60}" y="{top + 4}" font-size="10" fill="#6b7280">{yt_top_label}</text>')
            svg.append(f'<text x="{left - 60}" y="{bottom + 4}" font-size="10" fill="#6b7280">{yt_bot_label}</text>')

    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def compute_claim_checks(
    summary: dict[str, dict[str, dict[str, dict[str, Any]]]],
    eval_seeds: list[int],
    train_seed: int | None,
) -> dict[str, Any]:
    claims: dict[str, Any] = {
        "eval_seeds": list(eval_seeds),
        "train_seed": train_seed,
        "eval_seeds_disjoint_from_train_seed": None if train_seed is None else all(seed != train_seed for seed in eval_seeds),
        "generators": {},
    }
    baseline_methods = ["em", "kmeans", "spectral", "diag_gmm"]
    for generator, gen_summary in summary.items():
        generator_claims: dict[str, Any] = {}
        none_rows = {method: metrics.get("none") for method, metrics in gen_summary.items()}
        if "tgmm" in none_rows and none_rows["tgmm"] is not None:
            tgmm_row = none_rows["tgmm"]
            assert tgmm_row is not None
            valid_baselines = [m for m in baseline_methods if none_rows.get(m) is not None]
            if valid_baselines:
                best_baseline_pe_method = min(
                    valid_baselines,
                    key=lambda m: float(none_rows[m]["parameter_error"]["mean"]),
                )
                best_baseline_acc_method = max(
                    valid_baselines,
                    key=lambda m: float(none_rows[m]["cluster_acc"]["mean"]),
                )
                generator_claims["no_shift"] = {
                    "tgmm_parameter_error": float(tgmm_row["parameter_error"]["mean"]),
                    "tgmm_cluster_acc": float(tgmm_row["cluster_acc"]["mean"]),
                    "best_baseline_parameter_error_method": best_baseline_pe_method,
                    "best_baseline_parameter_error": float(none_rows[best_baseline_pe_method]["parameter_error"]["mean"]),
                    "best_baseline_cluster_acc_method": best_baseline_acc_method,
                    "best_baseline_cluster_acc": float(none_rows[best_baseline_acc_method]["cluster_acc"]["mean"]),
                    "tgmm_beats_best_baseline_on_parameter_error": float(tgmm_row["parameter_error"]["mean"]) < float(none_rows[best_baseline_pe_method]["parameter_error"]["mean"]),
                    "tgmm_beats_best_baseline_on_cluster_acc": float(tgmm_row["cluster_acc"]["mean"]) > float(none_rows[best_baseline_acc_method]["cluster_acc"]["mean"]),
                }
        intervention_gaps: dict[str, Any] = {}
        if "tgmm" in gen_summary:
            tgmm_none = gen_summary["tgmm"].get("none")
            if tgmm_none is not None:
                for intervention, metrics in gen_summary["tgmm"].items():
                    if intervention == "none":
                        continue
                    intervention_gaps[intervention] = {
                        "parameter_error_gap": float(metrics["parameter_error"]["mean"] - tgmm_none["parameter_error"]["mean"]),
                        "cluster_acc_gap": float(tgmm_none["cluster_acc"]["mean"] - metrics["cluster_acc"]["mean"]),
                    }
        generator_claims["tgmm_interventional_gaps"] = intervention_gaps
        claims["generators"][generator] = generator_claims
    return claims


def build_claim_checks_md(claims: dict[str, Any]) -> str:
    lines = ["# Clean-claim checks", ""]
    if claims["train_seed"] is not None:
        lines.append(
            f"- Train seed: {claims['train_seed']}; eval seeds: {claims['eval_seeds']}; disjoint: {claims['eval_seeds_disjoint_from_train_seed']}"
        )
    else:
        lines.append(f"- Eval seeds: {claims['eval_seeds']}")
    lines.append("")
    for generator, payload in claims["generators"].items():
        lines.append(f"## {generator}")
        lines.append("")
        no_shift = payload.get("no_shift")
        if no_shift:
            lines.append(
                "- No shift: TGMM PE = "
                f"{no_shift['tgmm_parameter_error']:.4f}; best baseline PE = {no_shift['best_baseline_parameter_error']:.4f} "
                f"({no_shift['best_baseline_parameter_error_method']}); TGMM wins on PE = {no_shift['tgmm_beats_best_baseline_on_parameter_error']}"
            )
            lines.append(
                "- No shift: TGMM Acc = "
                f"{no_shift['tgmm_cluster_acc']:.4f}; best baseline Acc = {no_shift['best_baseline_cluster_acc']:.4f} "
                f"({no_shift['best_baseline_cluster_acc_method']}); TGMM wins on Acc = {no_shift['tgmm_beats_best_baseline_on_cluster_acc']}"
            )
        gaps = payload.get("tgmm_interventional_gaps", {})
        if gaps:
            lines.append("")
            lines.append("| Intervention | TGMM PE gap | TGMM Acc gap |")
            lines.append("|---|---:|---:|")
            for intervention, gap_payload in gaps.items():
                lines.append(
                    f"| {intervention} | {gap_payload['parameter_error_gap']:.4f} | {gap_payload['cluster_acc_gap']:.4f} |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.checkpoint = rewrite_tmp_relative_path(args.checkpoint, args.tmp_dir)
    args.output_dir = rewrite_tmp_relative_path(args.output_dir, args.tmp_dir)
    try:
        configure_torch_runtime(args.num_threads)
        device = choose_device(args.device)
        out_dir = ensure_dir(args.output_dir)

        model, base_task_cfg, checkpoint_payload, train_seed = load_model_and_task_cfg(Path(args.checkpoint), device)
        benchmark = load_split_benchmark(
            dataset=base_task_cfg.dataset,
            dataset_path=base_task_cfg.dataset_path,
            label_column=base_task_cfg.label_column,
            d=base_task_cfg.d,
            k=base_task_cfg.k,
            split_seed=base_task_cfg.split_seed,
            train_fraction=base_task_cfg.train_fraction,
            fit_preprocessor_on_train=base_task_cfg.fit_preprocessor_on_train,
        )
        interventions = build_interventions(args)

        records: list[dict[str, Any]] = []
        for generator in args.generators:
            task_cfg = SplitTaskConfig(**{**base_task_cfg.to_dict(), "generator": generator})
            for intervention in interventions:
                for seed in args.seeds:
                    rng = np.random.default_rng(seed)
                    for task_idx in range(args.num_tasks):
                        task = sample_split_task(
                            cfg=task_cfg,
                            benchmark=benchmark,
                            split=args.eval_split,
                            rng=rng,
                            intervention=intervention.config,
                        )
                        for method in args.methods:
                            metrics, success, error = evaluate_method(
                                method=method,
                                task=task,
                                model=model,
                                device=device,
                                seed=seed,
                                em_restarts=args.em_restarts,
                                spectral_restarts=args.spectral_restarts,
                            )
                            records.append(
                                {
                                    "generator": generator,
                                    "method": method,
                                    "intervention": intervention.name,
                                    "family": intervention.family,
                                    "strength": intervention.strength,
                                    "seed": seed,
                                    "task_idx": task_idx,
                                    "success": success,
                                    "metrics": metrics,
                                    "error": error,
                                }
                            )

        summary = summarize_records(records)
        claims = compute_claim_checks(summary, eval_seeds=list(args.seeds), train_seed=train_seed)
        main_table = build_main_table(summary)
        claims_md = build_claim_checks_md(claims)

        payload = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "eval_args": vars(args),
            "clean_task_config": base_task_cfg.to_dict(),
            "clean_benchmark": checkpoint_payload.get("clean_benchmark", {}),
            "summary": summary,
            "claims": claims,
        }
        save_json(out_dir / "summary.json", payload)
        save_json(out_dir / "raw_results.json", records)
        save_json(out_dir / "claim_checks.json", claims)
        (out_dir / "main_table.md").write_text(main_table + "\n", encoding="utf-8")
        (out_dir / "claim_checks.md").write_text(claims_md + "\n", encoding="utf-8")
        write_method_comparison_bar_svg(
            out_dir / "method_comparison_bars.svg",
            summary=summary,
            methods=[str(m) for m in args.methods],
        )
        print(f"Saved clean-claim evaluation outputs to {out_dir}")
    finally:
        cleanup_tmp_dir(args.tmp_dir, args.show_tmp)


if __name__ == "__main__":
    main()
