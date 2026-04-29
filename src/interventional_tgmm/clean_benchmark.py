from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.decomposition import PCA

from .config import InterventionConfig
from .data import _read_wine_csv, _resolve_dataset_path, apply_intervention
from .types import GMMParams, GMMTask


_EPS = 1e-8
SplitName = Literal["train", "test"]
GeneratorKind = Literal["residual", "gaussian_diag"]


@dataclass(slots=True)
class SplitTaskConfig:
    dataset_path: str = "wine.csv"
    label_column: str = "Cultivars"
    k: int = 3
    d: int = 8
    n_min: int = 32
    n_max: int = 64
    sigma: float = 1.0
    dirichlet_alpha: float = 4.0
    min_weight: float = 0.12
    split_seed: int = 11
    train_fraction: float = 0.7
    fit_preprocessor_on_train: bool = True
    generator: GeneratorKind = "residual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "label_column": self.label_column,
            "k": self.k,
            "d": self.d,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "sigma": self.sigma,
            "dirichlet_alpha": self.dirichlet_alpha,
            "min_weight": self.min_weight,
            "split_seed": self.split_seed,
            "train_fraction": self.train_fraction,
            "fit_preprocessor_on_train": self.fit_preprocessor_on_train,
            "generator": self.generator,
        }


@dataclass(slots=True)
class WineSplit:
    indices: tuple[np.ndarray, ...]
    points: tuple[np.ndarray, ...]
    residuals: tuple[np.ndarray, ...]
    means: np.ndarray
    scales: np.ndarray
    covariances: tuple[np.ndarray, ...]
    weights: np.ndarray
    total_points: int


@dataclass(slots=True)
class WineSplitBenchmark:
    dataset_path: str
    label_column: str
    k: int
    d: int
    split_seed: int
    train_fraction: float
    fit_preprocessor_on_train: bool
    original_dim: int
    explained_variance_ratio: float | None
    feature_names: tuple[str, ...]
    train: WineSplit
    test: WineSplit


@dataclass(slots=True)
class SplitGMMEstimate:
    """Lightweight container for method outputs with optional custom metrics.

    This mirrors the pieces we need from the existing GMMEstimate while allowing
    methods such as sklearn GaussianMixture to provide their own average
    log-likelihood when desired.
    """

    weights: np.ndarray
    means: np.ndarray
    assumed_sigma: float = 1.0
    sigma_diag: np.ndarray | None = None
    converged: bool = True
    metadata: dict[str, Any] | None = None


def _standardize_train_reference(
    x_train: np.ndarray,
    x_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train_std = (x_train - mean) / std
    x_all_std = (x_all - mean) / std
    return x_train_std, x_all_std, mean, std


def _split_indices_by_class(labels: np.ndarray, k: int, split_seed: int, train_fraction: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}.")
    rng = np.random.default_rng(split_seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for class_idx in range(k):
        class_indices = np.where(labels == class_idx)[0]
        if class_indices.size < 4:
            raise ValueError(
                f"Class {class_idx} has only {class_indices.size} points; need at least 4 for a clean train/test split."
            )
        shuffled = class_indices.copy()
        rng.shuffle(shuffled)
        n_train = int(round(train_fraction * shuffled.size))
        n_train = int(np.clip(n_train, 2, shuffled.size - 2))
        train_parts.append(np.sort(shuffled[:n_train]))
        test_parts.append(np.sort(shuffled[n_train:]))
    return train_parts, test_parts


@lru_cache(maxsize=32)
def load_wine_split_benchmark(
    dataset_path: str,
    label_column: str,
    d: int,
    k: int,
    split_seed: int,
    train_fraction: float,
    fit_preprocessor_on_train: bool = True,
) -> WineSplitBenchmark:
    resolved = _resolve_dataset_path(dataset_path)
    x_raw, y_raw, feature_names = _read_wine_csv(resolved, label_column)
    original_dim = int(x_raw.shape[1])
    if d <= 0 or d > original_dim:
        raise ValueError(f"Requested d={d} but dataset has original_dim={original_dim}.")

    unique_labels = sorted(np.unique(y_raw).tolist())
    if len(unique_labels) != k:
        raise ValueError(f"Configured k={k} does not match class count={len(unique_labels)}.")
    label_map = {int(label): idx for idx, label in enumerate(unique_labels)}
    y = np.asarray([label_map[int(label)] for label in y_raw], dtype=np.int64)

    train_parts, test_parts = _split_indices_by_class(y, k=k, split_seed=split_seed, train_fraction=train_fraction)
    train_indices_all = np.concatenate(train_parts, axis=0)

    if fit_preprocessor_on_train:
        x_train_ref = x_raw[train_indices_all]
        x_train_std, x_all_std, _, _ = _standardize_train_reference(x_train_ref, x_raw)
        x_proc = x_all_std
        explained_variance_ratio: float | None = None
        if d < original_dim:
            pca = PCA(n_components=d, svd_solver="full", random_state=0)
            pca.fit(x_train_std)
            x_proc = pca.transform(x_all_std)
            explained_variance_ratio = float(np.sum(pca.explained_variance_ratio_))
    else:
        mean = x_raw.mean(axis=0, keepdims=True)
        std = x_raw.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        x_proc = (x_raw - mean) / std
        explained_variance_ratio = None
        if d < original_dim:
            pca = PCA(n_components=d, svd_solver="full", random_state=0)
            x_proc = pca.fit_transform(x_proc)
            explained_variance_ratio = float(np.sum(pca.explained_variance_ratio_))

    x_proc = np.asarray(x_proc, dtype=np.float32)

    def build_split(parts: list[np.ndarray]) -> WineSplit:
        points_per_class: list[np.ndarray] = []
        residuals_per_class: list[np.ndarray] = []
        covariances_per_class: list[np.ndarray] = []
        means = np.zeros((k, d), dtype=np.float32)
        scales = np.zeros((k, d), dtype=np.float32)
        weights = np.zeros(k, dtype=np.float32)
        total_points = int(sum(len(idx) for idx in parts))
        for class_idx, class_indices in enumerate(parts):
            pts = x_proc[class_indices]
            mean = pts.mean(axis=0).astype(np.float32)
            residuals = (pts - mean).astype(np.float32)
            scale = np.clip(residuals.std(axis=0), 1e-3, None).astype(np.float32)
            if pts.shape[0] >= 2:
                cov = np.cov(pts, rowvar=False).astype(np.float32)
            else:
                cov = np.diag(np.square(scale)).astype(np.float32)
            points_per_class.append(pts.astype(np.float32, copy=False))
            residuals_per_class.append(residuals)
            covariances_per_class.append(cov)
            means[class_idx] = mean
            scales[class_idx] = scale
            weights[class_idx] = float(len(class_indices) / total_points)
        return WineSplit(
            indices=tuple(np.asarray(idx, dtype=np.int64) for idx in parts),
            points=tuple(points_per_class),
            residuals=tuple(residuals_per_class),
            means=means,
            scales=scales,
            covariances=tuple(covariances_per_class),
            weights=weights,
            total_points=total_points,
        )

    return WineSplitBenchmark(
        dataset_path=str(resolved),
        label_column=label_column,
        k=k,
        d=d,
        split_seed=split_seed,
        train_fraction=train_fraction,
        fit_preprocessor_on_train=fit_preprocessor_on_train,
        original_dim=original_dim,
        explained_variance_ratio=explained_variance_ratio,
        feature_names=tuple(feature_names),
        train=build_split(train_parts),
        test=build_split(test_parts),
    )


def _sample_weights(base_weights: np.ndarray, alpha: float, min_weight: float, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(base_weights, dtype=np.float64)
    base = np.clip(base, _EPS, None)
    base /= base.sum()
    concentration = max(float(alpha), 1e-3)
    dirichlet = np.clip(base * concentration * len(base), 1e-3, None)
    weights = base.astype(np.float32, copy=True)
    for _ in range(512):
        weights = rng.dirichlet(dirichlet).astype(np.float32)
        if float(weights.min()) >= float(min_weight):
            return weights
    return weights


def _get_split(benchmark: WineSplitBenchmark, split: SplitName) -> WineSplit:
    if split == "train":
        return benchmark.train
    if split == "test":
        return benchmark.test
    raise ValueError(f"Unknown split: {split}")


def sample_split_task(
    cfg: SplitTaskConfig,
    benchmark: WineSplitBenchmark,
    split: SplitName,
    rng: np.random.Generator,
    intervention: InterventionConfig | None = None,
) -> GMMTask:
    if cfg.k != benchmark.k:
        raise ValueError(f"cfg.k={cfg.k} does not match benchmark.k={benchmark.k}.")
    if cfg.d != benchmark.d:
        raise ValueError(f"cfg.d={cfg.d} does not match benchmark.d={benchmark.d}.")

    intervention = intervention or InterventionConfig(kind="none")
    split_data = _get_split(benchmark, split)
    weights = _sample_weights(split_data.weights, cfg.dirichlet_alpha, cfg.min_weight, rng)
    means = split_data.means.copy()

    base_sigma_diag = split_data.scales.copy().astype(np.float32)
    base_sigma = float(base_sigma_diag.mean())
    if base_sigma < _EPS:
        base_sigma = 1.0
    base_sigma_diag = (base_sigma_diag * (float(cfg.sigma) / base_sigma)).astype(np.float32)
    sigma = float(base_sigma_diag.mean())
    sigma_diag: np.ndarray | None = None

    weights, means, sigma, sigma_diag = apply_intervention(
        weights=weights,
        means=means,
        sigma=sigma,
        sigma_diag=sigma_diag,
        intervention=intervention,
        rng=rng,
    )

    if intervention.kind == "sample_size" and intervention.sample_size is not None:
        n = max(2, int(intervention.sample_size))
    else:
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))

    z = rng.choice(cfg.k, size=n, p=weights).astype(np.int64)
    x = np.zeros((n, cfg.d), dtype=np.float32)

    if sigma_diag is None:
        true_scales = np.full((cfg.k, cfg.d), sigma / max(base_sigma, _EPS), dtype=np.float32) * base_sigma_diag
    else:
        true_scales = sigma_diag.astype(np.float32, copy=False)

    for class_idx in range(cfg.k):
        indices = np.where(z == class_idx)[0]
        if indices.size == 0:
            continue
        if cfg.generator == "residual":
            residual_pool = split_data.residuals[class_idx]
            sampled_idx = rng.integers(0, residual_pool.shape[0], size=indices.size)
            sampled_residuals = residual_pool[sampled_idx]
            scale_factor = true_scales[class_idx] / np.clip(base_sigma_diag[class_idx], _EPS, None)
            x[indices] = means[class_idx] + sampled_residuals * scale_factor
        elif cfg.generator == "gaussian_diag":
            x[indices] = rng.normal(
                loc=means[class_idx],
                scale=true_scales[class_idx],
                size=(indices.size, cfg.d),
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown generator kind: {cfg.generator}")

    params = GMMParams(
        weights=weights.astype(np.float32),
        means=means.astype(np.float32),
        sigma=float(sigma),
        sigma_diag=true_scales.astype(np.float32),
        metadata={
            "dataset": benchmark.dataset_path,
            "label_column": benchmark.label_column,
            "split": split,
            "generator": cfg.generator,
            "split_seed": benchmark.split_seed,
            "train_fraction": benchmark.train_fraction,
            "fit_preprocessor_on_train": benchmark.fit_preprocessor_on_train,
            "explained_variance_ratio": benchmark.explained_variance_ratio,
        },
    )
    return GMMTask(x=x.astype(np.float32), z=z.astype(np.int64), params=params, intervention=intervention.kind)


def sample_split_task_batch(
    cfg: SplitTaskConfig,
    benchmark: WineSplitBenchmark,
    split: SplitName,
    batch_size: int,
    rng: np.random.Generator,
    intervention: InterventionConfig | None = None,
) -> dict[str, torch.Tensor]:
    tasks = [sample_split_task(cfg, benchmark, split, rng, intervention=intervention) for _ in range(batch_size)]
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


def task_hash(task: GMMTask, decimals: int = 6) -> str:
    digest = hashlib.sha256()
    arrays = [
        np.round(task.x.astype(np.float64, copy=False), decimals=decimals),
        task.z.astype(np.int64, copy=False),
        np.round(task.params.weights.astype(np.float64, copy=False), decimals=decimals),
        np.round(task.params.means.astype(np.float64, copy=False), decimals=decimals),
        np.round(task.params.component_scales().astype(np.float64, copy=False), decimals=decimals),
    ]
    for arr in arrays:
        arr_c = np.ascontiguousarray(arr)
        digest.update(str(arr_c.shape).encode("utf-8"))
        digest.update(arr_c.tobytes())
    return digest.hexdigest()


def task_to_dict(task: GMMTask) -> dict[str, Any]:
    return {
        "x": task.x.tolist(),
        "z": task.z.tolist(),
        "params": {
            "weights": task.params.weights.tolist(),
            "means": task.params.means.tolist(),
            "sigma": task.params.sigma,
            "sigma_diag": task.params.component_scales().tolist(),
            "metadata": task.params.metadata,
        },
        "intervention": task.intervention,
    }
