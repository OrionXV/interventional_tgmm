from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import InterventionConfig, SamplingConfig
from .types import GMMParams, GMMTask


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def _pairwise_min_distance(means: np.ndarray) -> float:
    if len(means) < 2:
        return float("inf")
    diffs = means[:, None, :] - means[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    dists += np.eye(len(means)) * 1e9
    return float(dists.min())


def sample_weights(cfg: SamplingConfig, rng: np.random.Generator) -> np.ndarray:
    for _ in range(512):
        weights = rng.dirichlet(np.full(cfg.k, cfg.dirichlet_alpha, dtype=np.float64))
        if float(weights.min()) >= cfg.min_weight:
            return weights.astype(np.float32)
    return weights.astype(np.float32)


def sample_means(cfg: SamplingConfig, rng: np.random.Generator) -> np.ndarray:
    best: np.ndarray | None = None
    best_sep = -np.inf
    for _ in range(cfg.max_mean_resamples):
        means = rng.uniform(-cfg.mean_bound, cfg.mean_bound, size=(cfg.k, cfg.d)).astype(np.float32)
        cur_sep = _pairwise_min_distance(means)
        if cur_sep > best_sep:
            best = means
            best_sep = cur_sep
        if cur_sep >= cfg.min_separation:
            return means
    assert best is not None
    return best


def sample_anisotropic_scales(
    k: int,
    d: int,
    rng: np.random.Generator,
    min_log_scale: float,
    max_log_scale: float,
    base_sigma: float = 1.0,
) -> np.ndarray:
    logits = rng.uniform(min_log_scale, max_log_scale, size=(k, d)).astype(np.float32)
    scales = np.log1p(np.exp(logits)).astype(np.float32)
    return (float(base_sigma) * scales).astype(np.float32)


def apply_intervention(
    weights: np.ndarray,
    means: np.ndarray,
    sigma: float,
    sigma_diag: np.ndarray | None,
    intervention: InterventionConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    new_weights = weights.copy()
    new_means = means.copy()
    new_sigma = float(sigma)
    new_sigma_diag = None if sigma_diag is None else sigma_diag.copy()

    if intervention.kind == "none":
        return new_weights, new_means, new_sigma, new_sigma_diag

    if intervention.kind == "prior":
        idx = int(np.clip(intervention.target_component, 0, len(new_weights) - 1))
        logits = np.log(new_weights + 1e-12)
        logits[idx] += intervention.prior_strength
        new_weights = _softmax_np(logits).astype(np.float32)
        return new_weights, new_means, new_sigma, new_sigma_diag

    if intervention.kind == "mechanism":
        center = new_means.mean(axis=0, keepdims=True)
        new_means = center + intervention.mechanism_compression * (new_means - center)
        if intervention.mechanism_shift_scale > 0.0:
            new_means = new_means + rng.normal(
                loc=0.0,
                scale=intervention.mechanism_shift_scale,
                size=new_means.shape,
            ).astype(np.float32)
        return new_weights.astype(np.float32), new_means.astype(np.float32), new_sigma, new_sigma_diag

    if intervention.kind == "noise":
        if intervention.noise_type == "anisotropic":
            if new_sigma_diag is None:
                base_scales = np.full((len(new_weights), new_means.shape[1]), new_sigma, dtype=np.float32)
            else:
                base_scales = new_sigma_diag.astype(np.float32, copy=False)
            factors = sample_anisotropic_scales(
                k=base_scales.shape[0],
                d=base_scales.shape[1],
                rng=rng,
                min_log_scale=intervention.anisotropic_log_scale_min,
                max_log_scale=intervention.anisotropic_log_scale_max,
                base_sigma=1.0,
            )
            new_sigma_diag = (base_scales * factors * intervention.noise_factor).astype(np.float32)
            new_sigma = float(new_sigma_diag.mean())
            return new_weights.astype(np.float32), new_means.astype(np.float32), new_sigma, new_sigma_diag

        new_sigma = float(new_sigma * intervention.noise_factor)
        if new_sigma_diag is not None:
            new_sigma_diag = (new_sigma_diag * intervention.noise_factor).astype(np.float32)
        return new_weights.astype(np.float32), new_means.astype(np.float32), new_sigma, new_sigma_diag

    if intervention.kind == "sample_size":
        return new_weights.astype(np.float32), new_means.astype(np.float32), new_sigma, new_sigma_diag

    raise ValueError(f"Unknown intervention kind: {intervention.kind}")


def sample_task(
    cfg: SamplingConfig,
    rng: np.random.Generator,
    intervention: InterventionConfig | None = None,
) -> GMMTask:
    intervention = intervention or InterventionConfig(kind="none")
    weights = sample_weights(cfg, rng)
    means = sample_means(cfg, rng)
    sigma = float(cfg.sigma)
    sigma_diag: np.ndarray | None = None
    if cfg.anisotropic_prob > 0.0 and rng.random() < cfg.anisotropic_prob:
        sigma_diag = sample_anisotropic_scales(
            k=cfg.k,
            d=cfg.d,
            rng=rng,
            min_log_scale=cfg.anisotropic_log_scale_min,
            max_log_scale=cfg.anisotropic_log_scale_max,
            base_sigma=cfg.sigma,
        )

    weights, means, sigma, sigma_diag = apply_intervention(weights, means, sigma, sigma_diag, intervention, rng)

    if intervention.kind == "sample_size" and intervention.sample_size is not None:
        n = int(intervention.sample_size)
    else:
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))

    z = rng.choice(cfg.k, size=n, p=weights)
    noise = rng.normal(size=(n, cfg.d)).astype(np.float32)
    if sigma_diag is None:
        x = means[z] + sigma * noise
    else:
        x = means[z] + sigma_diag[z] * noise

    params = GMMParams(
        weights=weights.astype(np.float32),
        means=means.astype(np.float32),
        sigma=sigma,
        sigma_diag=None if sigma_diag is None else sigma_diag.astype(np.float32),
    )
    return GMMTask(x=x.astype(np.float32), z=z.astype(np.int64), params=params, intervention=intervention.kind)


def sample_task_batch(
    cfg: SamplingConfig,
    batch_size: int,
    rng: np.random.Generator,
    intervention: InterventionConfig | None = None,
) -> dict[str, torch.Tensor]:
    tasks = [sample_task(cfg, rng, intervention=intervention) for _ in range(batch_size)]
    max_n = max(task.x.shape[0] for task in tasks)
    x = np.zeros((batch_size, max_n, cfg.d), dtype=np.float32)
    mask = np.zeros((batch_size, max_n), dtype=bool)
    means = np.zeros((batch_size, cfg.k, cfg.d), dtype=np.float32)
    weights = np.zeros((batch_size, cfg.k), dtype=np.float32)
    scales = np.zeros((batch_size, cfg.k, cfg.d), dtype=np.float32)

    for idx, task in enumerate(tasks):
        n = task.x.shape[0]
        x[idx, :n] = task.x
        mask[idx, :n] = True
        means[idx] = task.params.means
        weights[idx] = task.params.weights
        scales[idx] = task.params.component_scales()

    return {
        "x": torch.from_numpy(x),
        "mask": torch.from_numpy(mask),
        "true_means": torch.from_numpy(means),
        "true_weights": torch.from_numpy(weights),
        "true_scales": torch.from_numpy(scales),
    }


def task_to_dict(task: GMMTask) -> dict[str, Any]:
    return {
        "x": task.x.tolist(),
        "z": task.z.tolist(),
        "params": {
            "weights": task.params.weights.tolist(),
            "means": task.params.means.tolist(),
            "sigma": task.params.sigma,
            "sigma_diag": None if task.params.sigma_diag is None else task.params.sigma_diag.tolist(),
            "metadata": task.params.metadata,
        },
        "intervention": task.intervention,
    }
