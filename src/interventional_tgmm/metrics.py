from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp

from .matching import hungarian_match
from .types import GMMEstimate, GMMParams, GMMTask


_EPS = 1e-8


def match_estimate_to_truth(estimate: GMMEstimate, true_params: GMMParams) -> tuple[GMMEstimate, np.ndarray]:
    perm = hungarian_match(estimate.means, true_params.means)
    matched = GMMEstimate(
        weights=estimate.weights[perm].copy(),
        means=estimate.means[perm].copy(),
        assumed_sigma=estimate.assumed_sigma,
        sigma_diag=None if estimate.sigma_diag is None else estimate.sigma_diag[perm].copy(),
        converged=estimate.converged,
        metadata={**estimate.metadata, "perm": perm.tolist()},
    )
    return matched, perm


def _component_log_probs(x: np.ndarray, estimate: GMMEstimate) -> np.ndarray:
    _, d = x.shape
    scales = np.clip(estimate.component_scales().astype(np.float64, copy=False), _EPS, None)
    centered = x[:, None, :] - estimate.means[None, :, :]
    sq_dist = ((centered / scales[None, :, :]) ** 2).sum(axis=-1)
    log_pi = np.log(np.clip(estimate.weights, _EPS, None))[None, :]
    log_det = np.log(scales).sum(axis=-1)[None, :]
    const = -0.5 * d * np.log(2.0 * np.pi)
    return log_pi + const - log_det - 0.5 * sq_dist


def predict_labels(x: np.ndarray, estimate: GMMEstimate) -> np.ndarray:
    return _component_log_probs(x, estimate).argmax(axis=1)


def parameter_recovery_metrics(estimate: GMMEstimate, true_params: GMMParams) -> dict[str, float]:
    matched, _ = match_estimate_to_truth(estimate, true_params)
    mean_mse = float(((matched.means - true_params.means) ** 2).mean())
    weight_mse = float(((matched.weights - true_params.weights) ** 2).mean())
    matched_scales = matched.component_scales()
    true_scales = true_params.component_scales()
    scale_mse = float(((matched_scales - true_scales) ** 2).mean())
    return {
        "parameter_error": mean_mse + weight_mse + scale_mse,
        "mean_mse": mean_mse,
        "weight_mse": weight_mse,
        "scale_mse": scale_mse,
    }


def clustering_accuracy(task: GMMTask, estimate: GMMEstimate) -> float:
    matched, _ = match_estimate_to_truth(estimate, task.params)
    preds = predict_labels(task.x, matched)
    return float((preds == task.z).mean())


def average_log_likelihood(task: GMMTask, estimate: GMMEstimate) -> float:
    return float(logsumexp(_component_log_probs(task.x, estimate), axis=1).mean())


def evaluate_task(task: GMMTask, estimate: GMMEstimate) -> dict[str, float]:
    metrics = parameter_recovery_metrics(estimate, task.params)
    metrics["cluster_acc"] = clustering_accuracy(task, estimate)
    metrics["avg_log_likelihood"] = average_log_likelihood(task, estimate)
    return metrics


def summarize_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {}
    keys = sorted(metric_dicts[0].keys())
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([metrics[key] for metrics in metric_dicts], dtype=np.float64)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    return summary
