#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.clean_baselines import fit_method
from interventional_tgmm.clean_benchmark import SplitTaskConfig, load_split_benchmark, sample_split_task
from interventional_tgmm.config import InterventionConfig, ModelConfig
from interventional_tgmm.metrics import evaluate_task, match_estimate_to_truth, predict_labels
from interventional_tgmm.model import TGMMNet
from interventional_tgmm.tmp_utils import cleanup_tmp_dir, rewrite_tmp_relative_path
from interventional_tgmm.types import GMMEstimate
from interventional_tgmm.utils import choose_device, configure_torch_runtime, ensure_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize clustering panels for held-out clean-claims checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs_clean/eval/viz_clean_2d.svg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--show-tmp", action="store_true")
    parser.add_argument("--tmp-dir", type=str, default="outputs/tmp")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument(
        "--intervention",
        type=str,
        default="none",
        choices=["none", "prior", "mechanism", "noise", "noise_diag", "sample_size"],
    )
    parser.add_argument("--prior-strength", type=float, default=1.25)
    parser.add_argument("--mechanism-compression", type=float, default=0.65)
    parser.add_argument("--noise-factor", type=float, default=1.5)
    parser.add_argument("--methods", nargs="+", default=["tgmm", "em", "kmeans", "spectral", "diag_gmm"])
    parser.add_argument("--em-restarts", type=int, default=10)
    parser.add_argument("--spectral-restarts", type=int, default=20)
    return parser.parse_args()


def load_clean_model(checkpoint_path: Path, device: torch.device) -> tuple[TGMMNet, SplitTaskConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "clean_task_config" not in ckpt:
        raise ValueError(
            "Checkpoint does not contain clean_task_config. Use a checkpoint from scripts/train_clean_tgmm.py."
        )

    model_cfg = ModelConfig(**ckpt["model_config"])
    task_cfg = SplitTaskConfig(**ckpt["clean_task_config"])
    model = TGMMNet(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, task_cfg


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


def build_intervention(args: argparse.Namespace, num_points: int) -> InterventionConfig:
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
        return InterventionConfig(kind="sample_size", sample_size=int(num_points))
    raise ValueError(f"Unknown intervention kind: {args.intervention}")


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def color_for_label(label: int) -> str:
    # Colorblind-friendly palette.
    palette = ["#4C78A8", "#E45756", "#54A24B", "#F58518", "#B279A2", "#72B7B2"]
    return palette[int(label) % len(palette)]


def _safe_spectral_sigma(x: np.ndarray, k: int, sigma: float, margin: float = 0.95) -> float | None:
    """Return a reduced sigma that keeps top-k corrected second moments positive."""
    if not (0.0 < margin < 1.0):
        raise ValueError(f"margin must be in (0, 1), got {margin}.")
    n = x.shape[0]
    if n <= 0:
        return None
    raw_m2 = (x.astype(np.float64).T @ x.astype(np.float64)) / float(n)
    eigvals = np.linalg.eigvalsh(raw_m2)
    if eigvals.size < k:
        return None
    kth = float(eigvals[-k])
    if kth <= 1e-10:
        return None
    sigma_cap = math.sqrt(max(1e-10, margin * kth))
    return min(float(sigma), sigma_cap)


def make_projector(x: np.ndarray, means_blocks: list[np.ndarray]) -> callable:
    d = int(x.shape[1])
    if d == 1:
        return lambda arr: np.concatenate([arr.astype(np.float32), np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
    if d == 2:
        return lambda arr: arr[:, :2].astype(np.float32)

    stack_parts = [x.astype(np.float64)]
    for means in means_blocks:
        if means.size > 0:
            stack_parts.append(means.astype(np.float64))
    ref = np.concatenate(stack_parts, axis=0)
    pca = PCA(n_components=2, svd_solver="full", random_state=0)
    pca.fit(ref)
    return lambda arr: pca.transform(arr).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.checkpoint = rewrite_tmp_relative_path(args.checkpoint, args.tmp_dir)
    args.output = rewrite_tmp_relative_path(args.output, args.tmp_dir)
    try:
        configure_torch_runtime(args.num_threads)
        seed_everything(args.seed)
        device = choose_device(args.device)

        model, base_task_cfg = load_clean_model(Path(args.checkpoint), device)
        task_cfg = SplitTaskConfig(
            **{
                **base_task_cfg.to_dict(),
                "n_min": int(args.num_points),
                "n_max": int(args.num_points),
            }
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

        intervention = build_intervention(args, args.num_points)
        rng = np.random.default_rng(args.seed)
        task = sample_split_task(task_cfg, benchmark, split=args.split, rng=rng, intervention=intervention)

        panels: list[dict[str, object]] = [
            {
                "name": "truth",
                "labels": task.z.tolist(),
                "means": task.params.means.tolist(),
            }
        ]

        for method in args.methods:
            try:
                if method == "tgmm":
                    estimate = model_estimate(model, task.x, device, sigma=task.params.sigma)
                else:
                    try:
                        baseline = fit_method(
                            method=method,
                            x=task.x,
                            k=task.params.means.shape[0],
                            sigma=task.params.sigma,
                            seed=args.seed,
                            em_restarts=args.em_restarts,
                            spectral_restarts=args.spectral_restarts,
                        )
                    except ValueError as exc:
                        # For spectral on some Iris tasks, the provided sigma can make the
                        # corrected second moment non-positive. Retry with a safe reduced sigma.
                        if method != "spectral" or "second moment eigenvalues" not in str(exc).lower():
                            raise
                        retry_sigma = _safe_spectral_sigma(
                            task.x,
                            k=int(task.params.means.shape[0]),
                            sigma=float(task.params.sigma),
                        )
                        if retry_sigma is None:
                            raise
                        baseline = fit_method(
                            method=method,
                            x=task.x,
                            k=task.params.means.shape[0],
                            sigma=float(retry_sigma),
                            seed=args.seed,
                            em_restarts=args.em_restarts,
                            spectral_restarts=args.spectral_restarts,
                        )
                        if baseline.estimate.metadata is None:
                            baseline.estimate.metadata = {}
                        baseline.estimate.metadata["sigma_retry"] = float(retry_sigma)
                        baseline.estimate.metadata["sigma_original"] = float(task.params.sigma)
                    estimate = baseline.estimate

                matched_est, _ = match_estimate_to_truth(estimate, task.params)
                labels = predict_labels(task.x, matched_est)
                metrics = evaluate_task(task, matched_est)
                panels.append(
                    {
                        "name": method,
                        "labels": labels.tolist(),
                        "means": matched_est.means.tolist(),
                        "metrics": metrics,
                    }
                )
            except Exception as exc:  # pragma: no cover - continue on method failures
                panels.append({"name": method, "error": str(exc), "labels": [], "means": []})

        means_blocks = [task.params.means.astype(np.float32)]
        for panel in panels:
            means_arr = np.asarray(panel.get("means", []), dtype=np.float32)
            means_blocks.append(means_arr)

        project = make_projector(task.x, means_blocks)
        x_plot = project(task.x)
        true_means_plot = project(task.params.means)

        x_min = float(x_plot[:, 0].min())
        x_max = float(x_plot[:, 0].max())
        y_min = float(x_plot[:, 1].min())
        y_max = float(x_plot[:, 1].max())

        projected_means_by_panel: list[np.ndarray] = []
        for panel in panels:
            means_arr = np.asarray(panel.get("means", []), dtype=np.float32)
            if means_arr.size > 0:
                means_plot = project(means_arr)
                x_min = min(x_min, float(means_plot[:, 0].min()))
                x_max = max(x_max, float(means_plot[:, 0].max()))
                y_min = min(y_min, float(means_plot[:, 1].min()))
                y_max = max(y_max, float(means_plot[:, 1].max()))
            else:
                means_plot = np.zeros((0, 2), dtype=np.float32)
            projected_means_by_panel.append(means_plot)

        x_pad = 0.1 * max(1e-6, x_max - x_min)
        y_pad = 0.1 * max(1e-6, y_max - y_min)
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        panel_w = 360
        panel_h = 310
        cols = 3
        rows = int(math.ceil(len(panels) / cols))
        width = cols * panel_w
        legend_h = 38
        height = rows * panel_h + legend_h

        svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="#f3f5f7"/>',
            '<text x="12" y="22" font-size="16" fill="#111827">Clean Clustering Comparison</text>',
            '<circle cx="250" cy="16" r="4" fill="#4C78A8" stroke="#ffffff" stroke-width="0.8"/>',
            '<text x="260" y="20" font-size="11" fill="#374151">assigned point label</text>',
            '<line x1="430" y1="16" x2="442" y2="16" stroke="#111827" stroke-width="2"/>',
            '<line x1="436" y1="10" x2="436" y2="22" stroke="#111827" stroke-width="2"/>',
            '<text x="448" y="20" font-size="11" fill="#374151">predicted mean</text>',
            '<circle cx="570" cy="16" r="7" fill="none" stroke="#111827" stroke-width="1.3"/>',
            '<text x="582" y="20" font-size="11" fill="#374151">true mean</text>',
        ]

        for idx, panel in enumerate(panels):
            ox = (idx % cols) * panel_w
            oy = legend_h + (idx // cols) * panel_h
            left = ox + 48
            right = ox + panel_w - 22
            top = oy + 48
            bottom = oy + panel_h - 34

            name = str(panel["name"])
            svg.append(
                f'<rect x="{ox + 8}" y="{oy + 8}" width="{panel_w - 16}" height="{panel_h - 14}" fill="#ffffff" stroke="#d1d5db" rx="8"/>'
            )
            svg.append(f'<text x="{ox + 16}" y="{oy + 26}" font-size="15" fill="#111827">{xml_escape(name)}</text>')

            metric_payload = panel.get("metrics")
            if isinstance(metric_payload, dict):
                pe = float(metric_payload.get("parameter_error", float("nan")))
                acc = float(metric_payload.get("cluster_acc", float("nan")))
                if math.isfinite(pe) and math.isfinite(acc):
                    subtitle = f"PE={pe:.3f}, Acc={acc:.3f}"
                    svg.append(f'<text x="{ox + 16}" y="{oy + 40}" font-size="11" fill="#6b7280">{xml_escape(subtitle)}</text>')

            svg.append(
                f'<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="#fcfcfd" stroke="#e5e7eb"/>'
            )

            for grid_idx in range(5):
                gy = top + grid_idx * (bottom - top) / 4.0
                svg.append(f'<line x1="{left}" y1="{gy:.2f}" x2="{right}" y2="{gy:.2f}" stroke="#eef2f7" stroke-width="1"/>')
            for grid_idx in range(5):
                gx = left + grid_idx * (right - left) / 4.0
                svg.append(f'<line x1="{gx:.2f}" y1="{top}" x2="{gx:.2f}" y2="{bottom}" stroke="#eef2f7" stroke-width="1"/>')

            error = panel.get("error")
            if isinstance(error, str) and error:
                err_msg = xml_escape(error)
                if len(err_msg) > 70:
                    err_msg = err_msg[:67] + "..."
                svg.append(
                    f'<text x="{left + 8}" y="{top + 20}" font-size="11" fill="#b22222">error: {err_msg}</text>'
                )

            labels = panel.get("labels", [])
            for point, label in zip(x_plot, labels, strict=False):
                px = left + (float(point[0]) - x_min) / (x_max - x_min) * (right - left)
                py = bottom - (float(point[1]) - y_min) / (y_max - y_min) * (bottom - top)
                svg.append(
                    f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.2" fill="{color_for_label(int(label))}" stroke="#ffffff" stroke-width="0.9" opacity="0.92"/>'
                )

            means_plot = projected_means_by_panel[idx]
            for mean in means_plot:
                px = left + (float(mean[0]) - x_min) / (x_max - x_min) * (right - left)
                py = bottom - (float(mean[1]) - y_min) / (y_max - y_min) * (bottom - top)
                svg.append(f'<line x1="{px - 6:.2f}" y1="{py:.2f}" x2="{px + 6:.2f}" y2="{py:.2f}" stroke="#111827" stroke-width="2"/>')
                svg.append(f'<line x1="{px:.2f}" y1="{py - 6:.2f}" x2="{px:.2f}" y2="{py + 6:.2f}" stroke="#111827" stroke-width="2"/>')

            for mean in true_means_plot:
                px = left + (float(mean[0]) - x_min) / (x_max - x_min) * (right - left)
                py = bottom - (float(mean[1]) - y_min) / (y_max - y_min) * (bottom - top)
                svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="7" fill="none" stroke="#111827" stroke-width="1.3"/>')

        svg.append("</svg>")

        out_path = Path(args.output)
        ensure_dir(out_path.parent)
        out_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
        print(f"Saved clean clustering visualization to {out_path}")
    finally:
        cleanup_tmp_dir(args.tmp_dir, args.show_tmp)


if __name__ == "__main__":
    main()
