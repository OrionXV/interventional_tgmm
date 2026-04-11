#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate interventional robustness.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint_last.pt")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=["tgmm", "em", "kmeans", "spectral"])
    parser.add_argument("--em-restarts", type=int, default=10)
    parser.add_argument("--em-max-iters", type=int, default=100)
    parser.add_argument("--solver-sigma", type=float, default=None)
    return parser.parse_args()


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
    return GMMEstimate(weights=weights.astype(np.float32), means=means.astype(np.float32), assumed_sigma=float(sigma))


def build_interventions(data_cfg: SamplingConfig) -> list[InterventionConfig]:
    return [
        InterventionConfig(kind="none"),
        InterventionConfig(kind="prior", target_component=0, prior_strength=1.25),
        InterventionConfig(kind="mechanism", mechanism_compression=0.65),
        InterventionConfig(kind="noise", noise_factor=1.5),
        InterventionConfig(kind="sample_size", sample_size=max(2 * data_cfg.n_max, data_cfg.n_max + 32)),
    ]


def safe_metric_eval(task, estimate_fn: Callable[[], GMMEstimate]) -> dict[str, float]:
    try:
        estimate = estimate_fn()
        return evaluate_task(task, estimate)
    except Exception:
        return {
            "parameter_error": float("nan"),
            "mean_mse": float("nan"),
            "weight_mse": float("nan"),
            "cluster_acc": float("nan"),
            "avg_log_likelihood": float("nan"),
        }


def main() -> None:
    args = parse_args()
    configure_torch_runtime(args.num_threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    checkpoint_path = Path(args.checkpoint)
    model, data_cfg = load_model(checkpoint_path, device)
    solver_sigma = float(data_cfg.sigma if args.solver_sigma is None else args.solver_sigma)

    interventions = build_interventions(data_cfg)
    rng = np.random.default_rng(args.seed)
    raw_results: dict[str, dict[str, list[dict[str, float]]]] = {
        method: {intervention.kind: [] for intervention in interventions}
        for method in args.methods
    }

    for intervention in interventions:
        for _ in range(args.num_tasks):
            task = sample_task(data_cfg, rng=rng, intervention=intervention)

            if "tgmm" in args.methods:
                raw_results["tgmm"][intervention.kind].append(
                    safe_metric_eval(task, lambda: model_estimate(model, task.x, device, sigma=solver_sigma))
                )
            if "em" in args.methods:
                raw_results["em"][intervention.kind].append(
                    safe_metric_eval(
                        task,
                        lambda: fit_em_known_sigma(
                            task.x,
                            k=data_cfg.k,
                            sigma=solver_sigma,
                            restarts=args.em_restarts,
                            max_iters=args.em_max_iters,
                            seed=args.seed,
                        ),
                    )
                )
            if "kmeans" in args.methods:
                raw_results["kmeans"][intervention.kind].append(
                    safe_metric_eval(task, lambda: fit_kmeans_baseline(task.x, k=data_cfg.k, sigma=solver_sigma, seed=args.seed))
                )
            if "spectral" in args.methods:
                raw_results["spectral"][intervention.kind].append(
                    safe_metric_eval(
                        task,
                        lambda: fit_spectral_isotropic(task.x, k=data_cfg.k, sigma=solver_sigma, seed=args.seed),
                    )
                )

    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for method, method_results in raw_results.items():
        summary[method] = {}
        for intervention_name, metric_list in method_results.items():
            summary[method][intervention_name] = summarize_metric_dicts(metric_list)

    save_json(out_dir / "raw_results.json", raw_results)
    save_json(out_dir / "summary.json", summary)
    print(f"Saved evaluation outputs to {out_dir}")
    for method, method_summary in summary.items():
        print(f"\n== {method} ==")
        for intervention_name, metrics in method_summary.items():
            param_err = metrics["parameter_error"]["mean"]
            acc = metrics["cluster_acc"]["mean"]
            ll = metrics["avg_log_likelihood"]["mean"]
            print(
                f"{intervention_name:>12s}: parameter_error={param_err:.4f}, "
                f"cluster_acc={acc:.4f}, avg_log_likelihood={ll:.4f}"
            )


if __name__ == "__main__":
    main()
