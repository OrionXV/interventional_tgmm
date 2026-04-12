#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.baselines import fit_em_known_sigma, fit_kmeans_baseline, fit_spectral_isotropic
from interventional_tgmm.config import InterventionConfig, ModelConfig, SamplingConfig
from interventional_tgmm.data import sample_task
from interventional_tgmm.metrics import match_estimate_to_truth, predict_labels
from interventional_tgmm.model import TGMMNet
from interventional_tgmm.types import GMMEstimate
from interventional_tgmm.utils import choose_device, configure_torch_runtime, ensure_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2D qualitative visualization panels as SVG.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoint_last.pt")
    parser.add_argument("--output", type=str, default="outputs/eval/viz_2d.svg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--intervention", type=str, default="none", choices=["none", "prior", "mechanism", "noise", "noise_diag", "sample_size"])
    parser.add_argument("--prior-strength", type=float, default=1.25)
    parser.add_argument("--mechanism-compression", type=float, default=0.65)
    parser.add_argument("--noise-factor", type=float, default=1.5)
    parser.add_argument("--methods", nargs="+", default=["tgmm", "em", "kmeans", "spectral"])
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
    sigma_diag = None
    if "log_scales" in outputs:
        sigma_diag = model.decode_scales(outputs["log_scales"]).cpu().numpy()[0]
    return GMMEstimate(
        weights=weights.astype(np.float32),
        means=means.astype(np.float32),
        assumed_sigma=float(sigma),
        sigma_diag=None if sigma_diag is None else sigma_diag.astype(np.float32),
    )


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def color_for_label(label: int) -> str:
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    return palette[int(label) % len(palette)]


def build_intervention(args: argparse.Namespace, n_points: int) -> InterventionConfig:
    if args.intervention == "none":
        return InterventionConfig(kind="none")
    if args.intervention == "prior":
        return InterventionConfig(kind="prior", target_component=0, prior_strength=float(args.prior_strength))
    if args.intervention == "mechanism":
        return InterventionConfig(kind="mechanism", mechanism_compression=float(args.mechanism_compression))
    if args.intervention == "noise":
        return InterventionConfig(kind="noise", noise_factor=float(args.noise_factor), noise_type="scale")
    if args.intervention == "noise_diag":
        return InterventionConfig(kind="noise", noise_factor=float(args.noise_factor), noise_type="anisotropic")
    if args.intervention == "sample_size":
        return InterventionConfig(kind="sample_size", sample_size=int(n_points))
    raise ValueError(f"Unknown intervention kind: {args.intervention}")


def main() -> None:
    args = parse_args()
    configure_torch_runtime(args.num_threads)
    seed_everything(args.seed)
    device = choose_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    model, train_cfg = load_model(checkpoint_path, device)
    rng = np.random.default_rng(args.seed)

    cfg_2d = SamplingConfig(
        k=train_cfg.k,
        d=2,
        n_min=int(args.num_points),
        n_max=int(args.num_points),
        sigma=train_cfg.sigma,
    )
    intervention = build_intervention(args, args.num_points)
    task = sample_task(cfg_2d, rng=rng, intervention=intervention)

    panels: list[dict[str, object]] = [
        {"name": "truth", "labels": task.z.tolist(), "means": task.params.means.tolist()}
    ]

    solver_sigma = float(train_cfg.sigma)
    for method in args.methods:
        try:
            if method == "tgmm":
                estimate = model_estimate(model, task.x, device, sigma=solver_sigma)
            elif method == "em":
                estimate = fit_em_known_sigma(task.x, k=cfg_2d.k, sigma=solver_sigma, restarts=10, max_iters=100, seed=args.seed)
            elif method == "kmeans":
                estimate = fit_kmeans_baseline(task.x, k=cfg_2d.k, sigma=solver_sigma, seed=args.seed)
            elif method == "spectral":
                estimate = fit_spectral_isotropic(task.x, k=cfg_2d.k, sigma=solver_sigma, seed=args.seed)
            else:
                continue

            matched_est, _ = match_estimate_to_truth(estimate, task.params)
            labels = predict_labels(task.x, matched_est)
            panels.append(
                {
                    "name": method,
                    "labels": labels.tolist(),
                    "means": matched_est.means.tolist(),
                }
            )
        except Exception as exc:  # pragma: no cover - visualization should continue on method failure
            panels.append({"name": method, "error": str(exc), "labels": [], "means": []})

    out_path = Path(args.output)
    ensure_dir(out_path.parent)

    x_min = float(task.x[:, 0].min())
    x_max = float(task.x[:, 0].max())
    y_min = float(task.x[:, 1].min())
    y_max = float(task.x[:, 1].max())
    for panel in panels:
        means = np.asarray(panel.get("means", []), dtype=np.float32)
        if means.size > 0:
            x_min = min(x_min, float(means[:, 0].min()))
            x_max = max(x_max, float(means[:, 0].max()))
            y_min = min(y_min, float(means[:, 1].min()))
            y_max = max(y_max, float(means[:, 1].max()))

    x_pad = 0.1 * max(1e-6, x_max - x_min)
    y_pad = 0.1 * max(1e-6, y_max - y_min)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    panel_w = 300
    panel_h = 260
    cols = 3
    rows = int(math.ceil(len(panels) / cols))
    width = cols * panel_w
    height = rows * panel_h

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for idx, panel in enumerate(panels):
        ox = (idx % cols) * panel_w
        oy = (idx // cols) * panel_h
        left = ox + 42
        right = ox + panel_w - 20
        top = oy + 30
        bottom = oy + panel_h - 30

        name = str(panel["name"])
        svg.append(f'<text x="{ox + 12}" y="{oy + 18}" font-size="14" fill="#222">{xml_escape(name)}</text>')
        svg.append(f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="#fafafa" stroke="#cccccc"/>')

        labels = panel.get("labels", [])
        for point, label in zip(task.x, labels, strict=False):
            px = left + (float(point[0]) - x_min) / (x_max - x_min) * (right - left)
            py = bottom - (float(point[1]) - y_min) / (y_max - y_min) * (bottom - top)
            svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.2" fill="{color_for_label(int(label))}" opacity="0.78"/>')

        means = np.asarray(panel.get("means", []), dtype=np.float32)
        for mean in means:
            px = left + (float(mean[0]) - x_min) / (x_max - x_min) * (right - left)
            py = bottom - (float(mean[1]) - y_min) / (y_max - y_min) * (bottom - top)
            svg.append(f'<line x1="{px - 6:.2f}" y1="{py:.2f}" x2="{px + 6:.2f}" y2="{py:.2f}" stroke="#111" stroke-width="1.8"/>')
            svg.append(f'<line x1="{px:.2f}" y1="{py - 6:.2f}" x2="{px:.2f}" y2="{py + 6:.2f}" stroke="#111" stroke-width="1.8"/>')

        true_means = task.params.means
        for mean in true_means:
            px = left + (float(mean[0]) - x_min) / (x_max - x_min) * (right - left)
            py = bottom - (float(mean[1]) - y_min) / (y_max - y_min) * (bottom - top)
            svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="7" fill="none" stroke="#000" stroke-width="1"/>')

        if "error" in panel:
            svg.append(
                f'<text x="{left + 4}" y="{top + 14}" font-size="10" fill="#b22222">error: {xml_escape(str(panel["error"]))}</text>'
            )

    svg.append("</svg>")
    out_path.write_text("\n".join(svg) + "\n", encoding="utf-8")

    save_json(
        out_path.with_suffix(".json"),
        {
            "intervention": args.intervention,
            "num_points": int(args.num_points),
            "points": task.x.tolist(),
            "true_labels": task.z.tolist(),
            "true_means": task.params.means.tolist(),
            "panels": panels,
        },
    )
    print(f"Saved 2D visualization to {out_path}")


if __name__ == "__main__":
    main()
