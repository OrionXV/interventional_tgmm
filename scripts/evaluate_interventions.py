#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.baselines import fit_em_known_sigma, fit_kmeans_baseline, fit_spectral_isotropic
from interventional_tgmm.config import InterventionConfig, ModelConfig, SamplingConfig
from interventional_tgmm.data import sample_task
from interventional_tgmm.metrics import evaluate_task, summarize_metric_dicts
from interventional_tgmm.model import TGMMNet
from interventional_tgmm.types import GMMEstimate
from interventional_tgmm.utils import choose_device, configure_torch_runtime, ensure_dir, save_json, seed_everything


@dataclass(slots=True, frozen=True)
class InterventionSpec:
    name: str
    family: str
    strength: float
    config: InterventionConfig
    sampling_overrides: dict[str, float | int] | None = None


def parse_float_list(value: str) -> list[float]:
    tokens = [tok for tok in value.replace(",", " ").split() if tok]
    return [float(tok) for tok in tokens]


def parse_int_list(value: str) -> list[int]:
    tokens = [tok for tok in value.replace(",", " ").split() if tok]
    return [int(tok) for tok in tokens]


def format_float(value: float) -> str:
    text = f"{value:.4f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate interventional robustness with sweeps and report artifacts.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint_last.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--preset", type=str, default="none", choices=["none", "stage_a"])
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=["tgmm", "em", "kmeans", "spectral"])
    parser.add_argument("--em-restarts", type=int, default=10)
    parser.add_argument("--em-max-iters", type=int, default=100)
    parser.add_argument("--solver-sigma", type=float, default=None)
    parser.add_argument("--prior-strengths", type=str, default="1.25")
    parser.add_argument("--mechanism-compressions", type=str, default="0.65")
    parser.add_argument("--noise-factors", type=str, default="1.5")
    parser.add_argument("--sample-size-multipliers", type=str, default="2.0")
    parser.add_argument("--sample-sizes", type=str, default="")
    parser.add_argument("--dirichlet-alphas", type=str, default="")
    parser.add_argument("--dirichlet-min-weight", type=float, default=0.0)
    parser.add_argument("--include-anisotropic-noise", action="store_true")
    parser.add_argument("--anisotropic-log-scale-min", type=float, default=-1.0)
    parser.add_argument("--anisotropic-log-scale-max", type=float, default=1.0)
    parser.add_argument("--curve-metric", type=str, default="parameter_error")
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "none":
        return
    if args.preset == "stage_a":
        args.prior_strengths = ""
        args.mechanism_compressions = "0.5 0.35"
        args.noise_factors = ""
        args.sample_size_multipliers = ""
        args.sample_sizes = "16 8"
        args.dirichlet_alphas = "0.2"
        return
    raise ValueError(f"Unknown preset: {args.preset}")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TGMMNet, SamplingConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["model_config"])
    data_cfg = SamplingConfig(**ckpt["sampling_config"])
    model = TGMMNet(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, data_cfg


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


def build_interventions(data_cfg: SamplingConfig, args: argparse.Namespace) -> list[InterventionSpec]:
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

    noise_factors = sorted(set(parse_float_list(args.noise_factors)))
    for factor in noise_factors:
        interventions.append(
            InterventionSpec(
                name=f"noise@{format_float(factor)}",
                family="noise",
                strength=float(factor),
                config=InterventionConfig(kind="noise", noise_factor=float(factor), noise_type="scale"),
            )
        )

    if args.include_anisotropic_noise:
        for factor in noise_factors:
            interventions.append(
                InterventionSpec(
                    name=f"noise_diag@{format_float(factor)}",
                    family="noise_diag",
                    strength=float(factor),
                    config=InterventionConfig(
                        kind="noise",
                        noise_factor=float(factor),
                        noise_type="anisotropic",
                        anisotropic_log_scale_min=float(args.anisotropic_log_scale_min),
                        anisotropic_log_scale_max=float(args.anisotropic_log_scale_max),
                    ),
                )
            )

    for sample_size in sorted(set(parse_int_list(args.sample_sizes))):
        sample_size = max(2, int(sample_size))
        interventions.append(
            InterventionSpec(
                name=f"sample_size@{sample_size}",
                family="sample_size",
                strength=float(sample_size),
                config=InterventionConfig(kind="sample_size", sample_size=sample_size),
            )
        )

    for multiplier in sorted(set(parse_float_list(args.sample_size_multipliers))):
        sample_size = max(4, int(round(float(multiplier) * data_cfg.n_max)))
        interventions.append(
            InterventionSpec(
                name=f"sample_size@{sample_size}",
                family="sample_size",
                strength=float(sample_size),
                config=InterventionConfig(kind="sample_size", sample_size=sample_size),
            )
        )

    for alpha in sorted(set(parse_float_list(args.dirichlet_alphas))):
        interventions.append(
            InterventionSpec(
                name=f"dirichlet@{format_float(alpha)}",
                family="prior_imbalance",
                strength=float(alpha),
                config=InterventionConfig(kind="none"),
                sampling_overrides={
                    "dirichlet_alpha": float(alpha),
                    "min_weight": float(args.dirichlet_min_weight),
                },
            )
        )

    deduped: list[InterventionSpec] = []
    seen_names: set[str] = set()
    for spec in interventions:
        if spec.name in seen_names:
            continue
        deduped.append(spec)
        seen_names.add(spec.name)
    return deduped


def empty_metrics() -> dict[str, float]:
    return {
        "parameter_error": float("nan"),
        "mean_mse": float("nan"),
        "weight_mse": float("nan"),
        "scale_mse": float("nan"),
        "cluster_acc": float("nan"),
        "avg_log_likelihood": float("nan"),
    }


def safe_metric_eval(task, estimate_fn: Callable[[], GMMEstimate]) -> dict[str, float]:
    try:
        estimate = estimate_fn()
        return evaluate_task(task, estimate)
    except Exception:
        return empty_metrics()


def summarize_records(
    records: list[dict[str, object]],
) -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], dict[str, dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    intervention_meta: dict[str, dict[str, object]] = {}

    for record in records:
        method = str(record["method"])
        intervention = str(record["intervention"])
        grouped.setdefault((method, intervention), []).append(record["metrics"])  # type: ignore[arg-type]
        intervention_meta[intervention] = {
            "family": str(record["family"]),
            "strength": float(record["strength"]),
            "sampling_overrides": record.get("sampling_overrides"),
        }

    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (method, intervention), metrics in grouped.items():
        summary.setdefault(method, {})[intervention] = summarize_metric_dicts(metrics)

    return summary, intervention_meta


def compute_interventional_gaps(
    summary: dict[str, dict[str, dict[str, dict[str, float]]]],
    intervention_meta: dict[str, dict[str, object]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    metrics_where_lower_is_better = {"parameter_error", "mean_mse", "weight_mse", "scale_mse"}
    gaps_by_intervention: dict[str, dict[str, dict[str, float]]] = {}
    gaps_by_family: dict[str, dict[str, dict[str, float]]] = {}

    for method, method_summary in summary.items():
        baseline = method_summary.get("none")
        if baseline is None:
            continue
        gaps_by_intervention[method] = {}
        family_accumulator: dict[str, dict[str, list[float]]] = {}

        for intervention_name, metric_summary in method_summary.items():
            if intervention_name == "none":
                continue

            gap_metrics: dict[str, float] = {}
            for metric_name, shifted_stats in metric_summary.items():
                baseline_stats = baseline.get(metric_name)
                if baseline_stats is None:
                    continue
                baseline_mean = float(baseline_stats["mean"])
                shifted_mean = float(shifted_stats["mean"])
                if metric_name in metrics_where_lower_is_better:
                    gap = shifted_mean - baseline_mean
                else:
                    gap = baseline_mean - shifted_mean
                gap_metrics[f"{metric_name}_gap"] = gap

            gaps_by_intervention[method][intervention_name] = gap_metrics

            family = str(intervention_meta[intervention_name]["family"])
            family_values = family_accumulator.setdefault(family, {})
            for key, value in gap_metrics.items():
                family_values.setdefault(key, []).append(float(value))

        gaps_by_family[method] = {}
        for family, family_values in family_accumulator.items():
            gaps_by_family[method][family] = {
                key: float(np.nanmean(np.asarray(values, dtype=np.float64)))
                for key, values in family_values.items()
            }

    return {"by_intervention": gaps_by_intervention, "by_family": gaps_by_family}


def format_mean_std(stats: dict[str, float]) -> str:
    mean = float(stats.get("mean", float("nan")))
    std = float(stats.get("std", float("nan")))
    if not math.isfinite(mean):
        return "NaN"
    if not math.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +- {std:.4f}"


def write_main_table(
    path: Path,
    summary: dict[str, dict[str, dict[str, dict[str, float]]]],
    interventions: list[InterventionSpec],
    methods: list[str],
) -> None:
    lines = [
        "# Main Quantitative Table",
        "",
        "| Method | Intervention | Parameter Error | Cluster Acc | Avg Log-Likelihood | Scale MSE |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for method in methods:
        method_summary = summary.get(method, {})
        for intervention in interventions:
            metrics = method_summary.get(intervention.name)
            if metrics is None:
                continue
            lines.append(
                "| "
                f"{method} | {intervention.name} | "
                f"{format_mean_std(metrics.get('parameter_error', {}))} | "
                f"{format_mean_std(metrics.get('cluster_acc', {}))} | "
                f"{format_mean_std(metrics.get('avg_log_likelihood', {}))} | "
                f"{format_mean_std(metrics.get('scale_mse', {}))} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_intervention_curve_svg(
    path: Path,
    summary: dict[str, dict[str, dict[str, dict[str, float]]]],
    interventions: list[InterventionSpec],
    methods: list[str],
    metric: str,
) -> None:
    families = sorted({spec.family for spec in interventions if spec.family != "none"})
    if not families:
        return

    panel_w = 360
    panel_h = 240
    cols = 2
    rows = int(math.ceil(len(families) / cols))
    width = panel_w * cols
    height = panel_h * rows
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for family_idx, family in enumerate(families):
        specs = [spec for spec in interventions if spec.family == family]
        specs = [spec for spec in specs if any(spec.name in summary.get(method, {}) for method in methods)]
        if not specs:
            continue
        specs.sort(key=lambda item: item.strength)

        all_y: list[float] = []
        for method in methods:
            for spec in specs:
                stats = summary.get(method, {}).get(spec.name, {}).get(metric)
                if stats is None:
                    continue
                value = float(stats.get("mean", float("nan")))
                if math.isfinite(value):
                    all_y.append(value)

        if not all_y:
            continue

        x_values = [float(spec.strength) for spec in specs]
        x_min = min(x_values)
        x_max = max(x_values)
        if x_max - x_min < 1e-9:
            x_min -= 1.0
            x_max += 1.0

        y_min = min(all_y)
        y_max = max(all_y)
        if y_max - y_min < 1e-9:
            y_min -= 1.0
            y_max += 1.0
        y_pad = 0.08 * (y_max - y_min)
        y_min -= y_pad
        y_max += y_pad

        origin_x = (family_idx % cols) * panel_w
        origin_y = (family_idx // cols) * panel_h

        left = origin_x + 58
        right = origin_x + panel_w - 22
        top = origin_y + 32
        bottom = origin_y + panel_h - 42

        svg.append(f'<text x="{origin_x + 12}" y="{origin_y + 20}" font-size="14" fill="#222222">{xml_escape(family)}</text>')
        svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333333" stroke-width="1"/>')
        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333333" stroke-width="1"/>')

        svg.append(
            f'<text x="{left}" y="{bottom + 16}" font-size="10" fill="#444444">{x_min:.2f}</text>'
        )
        svg.append(
            f'<text x="{right - 24}" y="{bottom + 16}" font-size="10" fill="#444444">{x_max:.2f}</text>'
        )
        svg.append(f'<text x="{left - 46}" y="{top + 4}" font-size="10" fill="#444444">{y_max:.3f}</text>')
        svg.append(f'<text x="{left - 46}" y="{bottom + 4}" font-size="10" fill="#444444">{y_min:.3f}</text>')

        for method_idx, method in enumerate(methods):
            points: list[tuple[float, float]] = []
            for spec in specs:
                stats = summary.get(method, {}).get(spec.name, {}).get(metric)
                if stats is None:
                    continue
                value = float(stats.get("mean", float("nan")))
                if not math.isfinite(value):
                    continue
                x = left + (spec.strength - x_min) / (x_max - x_min) * (right - left)
                y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
                points.append((x, y))

            if not points:
                continue

            color = colors[method_idx % len(colors)]
            if len(points) > 1:
                polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                svg.append(
                    f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>'
                )
            for x, y in points:
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')

            legend_x = origin_x + 12
            legend_y = origin_y + panel_h - 18 - 12 * method_idx
            svg.append(f'<rect x="{legend_x}" y="{legend_y - 8}" width="8" height="8" fill="{color}"/>')
            svg.append(
                f'<text x="{legend_x + 12}" y="{legend_y - 1}" font-size="10" fill="#333333">{xml_escape(method)}</text>'
            )

        svg.append(
            f'<text x="{origin_x + panel_w - 120}" y="{origin_y + 20}" font-size="10" fill="#555555">metric={xml_escape(metric)}</text>'
        )

    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_preset(args)
    configure_torch_runtime(args.num_threads)
    seed_everything(int(args.seeds[0]))
    device = choose_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    checkpoint_path = Path(args.checkpoint)
    model, data_cfg = load_model(checkpoint_path, device)
    solver_sigma = float(data_cfg.sigma if args.solver_sigma is None else args.solver_sigma)

    interventions = build_interventions(data_cfg, args)
    raw_records: list[dict[str, object]] = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        for intervention in interventions:
            for task_idx in range(args.num_tasks):
                task_cfg = data_cfg
                if intervention.sampling_overrides is not None:
                    task_cfg = replace(data_cfg, **intervention.sampling_overrides)
                task = sample_task(task_cfg, rng=rng, intervention=intervention.config)

                if "tgmm" in args.methods:
                    metrics = safe_metric_eval(task, lambda: model_estimate(model, task.x, device, sigma=solver_sigma))
                    raw_records.append(
                        {
                            "method": "tgmm",
                            "intervention": intervention.name,
                            "family": intervention.family,
                            "strength": intervention.strength,
                            "seed": int(seed),
                            "task_index": task_idx,
                            "sampling_overrides": intervention.sampling_overrides,
                            "metrics": metrics,
                        }
                    )

                if "em" in args.methods:
                    metrics = safe_metric_eval(
                        task,
                        lambda: fit_em_known_sigma(
                            task.x,
                            k=task_cfg.k,
                            sigma=solver_sigma,
                            restarts=args.em_restarts,
                            max_iters=args.em_max_iters,
                            seed=int(seed),
                        ),
                    )
                    raw_records.append(
                        {
                            "method": "em",
                            "intervention": intervention.name,
                            "family": intervention.family,
                            "strength": intervention.strength,
                            "seed": int(seed),
                            "task_index": task_idx,
                            "sampling_overrides": intervention.sampling_overrides,
                            "metrics": metrics,
                        }
                    )

                if "kmeans" in args.methods:
                    metrics = safe_metric_eval(
                        task,
                        lambda: fit_kmeans_baseline(task.x, k=task_cfg.k, sigma=solver_sigma, seed=int(seed)),
                    )
                    raw_records.append(
                        {
                            "method": "kmeans",
                            "intervention": intervention.name,
                            "family": intervention.family,
                            "strength": intervention.strength,
                            "seed": int(seed),
                            "task_index": task_idx,
                            "sampling_overrides": intervention.sampling_overrides,
                            "metrics": metrics,
                        }
                    )

                if "spectral" in args.methods:
                    metrics = safe_metric_eval(
                        task,
                        lambda: fit_spectral_isotropic(task.x, k=task_cfg.k, sigma=solver_sigma, seed=int(seed)),
                    )
                    raw_records.append(
                        {
                            "method": "spectral",
                            "intervention": intervention.name,
                            "family": intervention.family,
                            "strength": intervention.strength,
                            "seed": int(seed),
                            "task_index": task_idx,
                            "sampling_overrides": intervention.sampling_overrides,
                            "metrics": metrics,
                        }
                    )

    summary, intervention_meta = summarize_records(raw_records)
    interventional_gaps = compute_interventional_gaps(summary, intervention_meta)

    payload = {
        "metadata": {
            "num_tasks": args.num_tasks,
            "seeds": [int(seed) for seed in args.seeds],
            "methods": args.methods,
            "solver_sigma": solver_sigma,
            "em_restarts": args.em_restarts,
            "em_max_iters": args.em_max_iters,
        },
        "interventions": intervention_meta,
        "summary": summary,
        "interventional_gaps": interventional_gaps,
    }

    save_json(out_dir / "raw_results.json", raw_records)
    save_json(out_dir / "summary.json", payload)
    write_main_table(out_dir / "main_table.md", summary, interventions, methods=args.methods)
    write_intervention_curve_svg(
        out_dir / "intervention_curves.svg",
        summary,
        interventions,
        methods=args.methods,
        metric=args.curve_metric,
    )

    print(f"Saved evaluation outputs to {out_dir}")
    for method in args.methods:
        method_summary = summary.get(method, {})
        baseline = method_summary.get("none")
        if baseline is None:
            continue
        print(f"\n== {method} ==")
        base_err = baseline["parameter_error"]["mean"]
        base_acc = baseline["cluster_acc"]["mean"]
        base_ll = baseline["avg_log_likelihood"]["mean"]
        print(
            "none: "
            f"parameter_error={base_err:.4f}, cluster_acc={base_acc:.4f}, avg_log_likelihood={base_ll:.4f}"
        )
        family_gaps = interventional_gaps.get("by_family", {}).get(method, {})
        for family, gaps in sorted(family_gaps.items()):
            pe_gap = gaps.get("parameter_error_gap", float("nan"))
            acc_gap = gaps.get("cluster_acc_gap", float("nan"))
            ll_gap = gaps.get("avg_log_likelihood_gap", float("nan"))
            print(
                f"{family:>12s}: parameter_error_gap={pe_gap:.4f}, "
                f"cluster_acc_gap={acc_gap:.4f}, avg_log_likelihood_gap={ll_gap:.4f}"
            )


if __name__ == "__main__":
    main()
