from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from .config import InterventionConfig, SamplingConfig
from .types import GMMParams, GMMTask

_EPS = 1e-8


@dataclass(slots=True)
class WineDataset:
    path: str
    label_column: str
    features: np.ndarray
    labels: np.ndarray
    class_points: tuple[np.ndarray, ...]
    class_residuals: tuple[np.ndarray, ...]
    class_means: np.ndarray
    class_scales: np.ndarray
    class_weights: np.ndarray
    feature_names: tuple[str, ...]
    original_dim: int


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def _resolve_dataset_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if path.is_file():
        return path.resolve()

    repo_root = Path(__file__).resolve().parents[2]
    candidate = (repo_root / dataset_path).resolve()
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"Could not find dataset at '{dataset_path}'.")


def _resolve_label_column(header: list[str], label_column: str) -> int:
    if label_column in header:
        return header.index(label_column)

    lowered = [name.lower() for name in header]
    target = label_column.lower()
    if target in lowered:
        return lowered.index(target)

    for fallback in ["cultivars", "class", "target", "label"]:
        if fallback in lowered:
            return lowered.index(fallback)

    raise ValueError(f"Label column '{label_column}' was not found in dataset header: {header}")


def _read_wine_csv(dataset_path: Path, label_column: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        raw_header = next(reader, None)
        if raw_header is None:
            raise ValueError(f"Dataset at '{dataset_path}' is empty.")

        header = [cell.strip() for cell in raw_header]
        label_idx = _resolve_label_column(header, label_column)

        feature_indices: list[int] = []
        feature_names: list[str] = []
        for idx, name in enumerate(header):
            if idx == label_idx:
                continue
            lowered = name.lower()
            if idx == 0 and (lowered == "" or lowered.startswith("unnamed") or lowered in {"index", "id"}):
                continue
            feature_indices.append(idx)
            feature_names.append(name if name else f"feature_{idx}")

        rows_x: list[list[float]] = []
        rows_y: list[int] = []
        for row in reader:
            if not row:
                continue
            rows_y.append(int(float(row[label_idx])))
            rows_x.append([float(row[idx]) for idx in feature_indices])

    x = np.asarray(rows_x, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.int64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError(f"Dataset at '{dataset_path}' did not contain tabular numeric features.")
    return x, y, feature_names


def _standardize_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


@lru_cache(maxsize=32)
def _load_wine_dataset_cached(dataset_path: str, label_column: str, d: int, k: int) -> WineDataset:
    resolved_path = _resolve_dataset_path(dataset_path)
    x_raw, y_raw, feature_names = _read_wine_csv(resolved_path, label_column)
    original_dim = int(x_raw.shape[1])

    if d <= 0:
        raise ValueError(f"Sampling dimension d must be positive; got d={d}.")
    if d > original_dim:
        raise ValueError(
            f"Requested d={d} but dataset has only {original_dim} continuous features before PCA."
        )

    unique_labels = sorted(np.unique(y_raw).tolist())
    if len(unique_labels) != k:
        raise ValueError(
            f"Configured k={k} does not match the dataset class count={len(unique_labels)} ({unique_labels})."
        )

    label_map = {int(label): idx for idx, label in enumerate(unique_labels)}
    y = np.asarray([label_map[int(label)] for label in y_raw], dtype=np.int64)

    x = _standardize_features(x_raw)
    if d < original_dim:
        x = PCA(n_components=d, svd_solver="full", random_state=0).fit_transform(x)
    x = x.astype(np.float32)

    class_points: list[np.ndarray] = []
    class_residuals: list[np.ndarray] = []
    class_means = np.zeros((k, d), dtype=np.float32)
    class_scales = np.zeros((k, d), dtype=np.float32)
    class_weights = np.zeros(k, dtype=np.float32)

    for class_idx in range(k):
        points = x[y == class_idx]
        if points.shape[0] == 0:
            raise ValueError(f"Class {class_idx} has no samples in the dataset.")

        mean = points.mean(axis=0).astype(np.float32)
        residuals = (points - mean).astype(np.float32)
        scales = np.clip(residuals.std(axis=0), 1e-3, None).astype(np.float32)

        class_points.append(points.astype(np.float32, copy=False))
        class_residuals.append(residuals)
        class_means[class_idx] = mean
        class_scales[class_idx] = scales
        class_weights[class_idx] = float(points.shape[0] / x.shape[0])

    return WineDataset(
        path=str(resolved_path),
        label_column=label_column,
        features=x,
        labels=y,
        class_points=tuple(class_points),
        class_residuals=tuple(class_residuals),
        class_means=class_means,
        class_scales=class_scales,
        class_weights=class_weights,
        feature_names=tuple(feature_names),
        original_dim=original_dim,
    )


def load_wine_dataset(cfg: SamplingConfig) -> WineDataset:
    if cfg.dataset.lower() != "wine":
        raise ValueError(f"Unsupported dataset '{cfg.dataset}'. This project now expects dataset='wine'.")
    return _load_wine_dataset_cached(cfg.dataset_path, cfg.label_column, cfg.d, cfg.k)


def sample_weights(
    cfg: SamplingConfig,
    rng: np.random.Generator,
    base_weights: np.ndarray | None = None,
) -> np.ndarray:
    if base_weights is None:
        base = np.full(cfg.k, 1.0 / cfg.k, dtype=np.float64)
    else:
        base = np.asarray(base_weights, dtype=np.float64)
        base = np.clip(base, _EPS, None)
        base /= base.sum()

    concentration = max(float(cfg.dirichlet_alpha), 1e-3)
    alpha = np.clip(base * concentration * cfg.k, 1e-3, None)
    weights = base.astype(np.float32, copy=True)
    for _ in range(512):
        weights = rng.dirichlet(alpha).astype(np.float32)
        if float(weights.min()) >= cfg.min_weight:
            return weights
    return weights


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
    dataset = load_wine_dataset(cfg)
    weights = sample_weights(cfg, rng, base_weights=dataset.class_weights)
    means = dataset.class_means.copy()
    sigma = float(cfg.sigma)
    base_sigma_diag = dataset.class_scales.copy()
    base_sigma = float(base_sigma_diag.mean())
    if base_sigma < _EPS:
        base_sigma = 1.0
    base_sigma_diag = (base_sigma_diag * (sigma / base_sigma)).astype(np.float32)
    base_sigma = float(base_sigma_diag.mean())

    sigma_diag: np.ndarray | None = None
    if cfg.anisotropic_prob > 0.0 and rng.random() < cfg.anisotropic_prob:
        sigma_diag = sample_anisotropic_scales(
            k=cfg.k,
            d=cfg.d,
            rng=rng,
            min_log_scale=cfg.anisotropic_log_scale_min,
            max_log_scale=cfg.anisotropic_log_scale_max,
            base_sigma=1.0,
        )
        sigma_diag = (base_sigma_diag * sigma_diag).astype(np.float32)
        sigma = float(sigma_diag.mean())

    weights, means, sigma, sigma_diag = apply_intervention(weights, means, sigma, sigma_diag, intervention, rng)

    if intervention.kind == "sample_size" and intervention.sample_size is not None:
        n = int(intervention.sample_size)
    else:
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))

    z = rng.choice(cfg.k, size=n, p=weights).astype(np.int64)
    x = np.zeros((n, cfg.d), dtype=np.float32)

    if sigma_diag is None:
        scale_factors = np.full((cfg.k, cfg.d), sigma / max(base_sigma, _EPS), dtype=np.float32)
    else:
        scale_factors = (sigma_diag / np.clip(base_sigma_diag, _EPS, None)).astype(np.float32)

    for class_idx in range(cfg.k):
        indices = np.where(z == class_idx)[0]
        if len(indices) == 0:
            continue
        residual_pool = dataset.class_residuals[class_idx]
        sampled_idx = rng.integers(0, residual_pool.shape[0], size=len(indices))
        sampled_residuals = residual_pool[sampled_idx]
        x[indices] = means[class_idx] + sampled_residuals * scale_factors[class_idx]

    params = GMMParams(
        weights=weights.astype(np.float32),
        means=means.astype(np.float32),
        sigma=sigma,
        sigma_diag=None if sigma_diag is None else sigma_diag.astype(np.float32),
        metadata={
            "dataset": dataset.path,
            "dataset_name": cfg.dataset,
            "label_column": dataset.label_column,
            "original_dim": dataset.original_dim,
            "projected_dim": cfg.d,
        },
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
