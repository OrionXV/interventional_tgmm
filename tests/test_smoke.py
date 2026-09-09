from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.clean_benchmark import load_split_benchmark
from interventional_tgmm.config import InterventionConfig, SamplingConfig
from interventional_tgmm.data import sample_task
from interventional_tgmm.losses import permutation_invariant_loss
from interventional_tgmm.metrics import parameter_recovery_metrics
from interventional_tgmm.types import GMMEstimate, GMMParams


def test_default_sampling_config() -> None:
    cfg = SamplingConfig()
    assert cfg.k == 3
    assert cfg.d == 8


def test_anisotropic_noise_intervention_returns_diagonal_scales() -> None:
    cfg = SamplingConfig(k=3, d=4, n_min=32, n_max=32, sigma=1.0)
    intervention = InterventionConfig(kind="noise", noise_type="anisotropic", noise_factor=1.25)
    task = sample_task(cfg, np.random.default_rng(0), intervention=intervention)
    assert task.params.sigma_diag is not None
    assert task.params.sigma_diag.shape == (cfg.k, cfg.d)
    assert bool((task.params.sigma_diag > 0.0).all())


def test_permutation_loss_supports_scale_head() -> None:
    batch, k, d = 2, 3, 4
    pred_means = torch.randn(batch, k, d)
    pred_weight_logits = torch.randn(batch, k)
    pred_log_scales = torch.randn(batch, k, d)
    true_means = torch.randn(batch, k, d)
    true_weights = torch.softmax(torch.randn(batch, k), dim=-1)
    true_scales = torch.rand(batch, k, d) + 0.5

    loss_dict = permutation_invariant_loss(
        pred_means=pred_means,
        pred_weight_logits=pred_weight_logits,
        true_means=true_means,
        true_weights=true_weights,
        pred_log_scales=pred_log_scales,
        true_scales=true_scales,
    )

    assert "scale_loss" in loss_dict
    assert torch.isfinite(loss_dict["scale_loss"])


def test_permutation_loss_supports_larger_k_without_factorial_blowup() -> None:
    batch, k, d = 2, 10, 4
    pred_means = torch.randn(batch, k, d)
    pred_weight_logits = torch.randn(batch, k)
    true_means = torch.randn(batch, k, d)
    true_weights = torch.softmax(torch.randn(batch, k), dim=-1)

    loss_dict = permutation_invariant_loss(
        pred_means=pred_means,
        pred_weight_logits=pred_weight_logits,
        true_means=true_means,
        true_weights=true_weights,
    )

    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["permutation"].shape == (batch, k)


def test_parameter_recovery_tracks_scale_error() -> None:
    true_params = GMMParams(
        weights=np.array([0.3, 0.3, 0.4], dtype=np.float32),
        means=np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        sigma_diag=np.array([[0.5, 0.8], [0.6, 1.0], [0.7, 0.9]], dtype=np.float32),
        sigma=1.0,
    )
    estimate = GMMEstimate(
        weights=true_params.weights.copy(),
        means=true_params.means.copy(),
        sigma_diag=np.full((3, 2), 1.2, dtype=np.float32),
        assumed_sigma=1.0,
    )

    metrics = parameter_recovery_metrics(estimate, true_params)
    assert metrics["scale_mse"] > 0.0
    assert metrics["parameter_error"] >= metrics["scale_mse"]


def test_builtin_iris_sampling_works() -> None:
    cfg = SamplingConfig(
        dataset="iris",
        dataset_path="",
        label_column="",
        k=3,
        d=4,
        n_min=24,
        n_max=24,
        sigma=1.0,
    )
    task = sample_task(cfg, np.random.default_rng(0))
    assert task.x.shape == (24, 4)
    assert task.z.shape == (24,)
    assert task.params.metadata is not None
    assert task.params.metadata["dataset_name"] == "iris"


def test_split_benchmark_supports_builtin_iris() -> None:
    benchmark = load_split_benchmark(
        dataset="iris",
        dataset_path="",
        label_column="",
        d=4,
        k=3,
        split_seed=11,
        train_fraction=0.7,
        fit_preprocessor_on_train=True,
    )
    assert benchmark.dataset_name == "iris"
    assert benchmark.train.total_points + benchmark.test.total_points == 150


def test_builtin_digits_sampling_works() -> None:
    cfg = SamplingConfig(
        dataset="digits",
        dataset_path="",
        label_column="",
        k=10,
        d=8,
        n_min=40,
        n_max=40,
        sigma=1.0,
        min_weight=0.03,
    )
    task = sample_task(cfg, np.random.default_rng(0))
    assert task.x.shape == (40, 8)
    assert task.z.shape == (40,)
    assert task.params.metadata is not None
    assert task.params.metadata["dataset_name"] == "digits"


def test_split_benchmark_supports_builtin_digits() -> None:
    benchmark = load_split_benchmark(
        dataset="digits",
        dataset_path="",
        label_column="",
        d=8,
        k=10,
        split_seed=11,
        train_fraction=0.7,
        fit_preprocessor_on_train=True,
    )
    assert benchmark.dataset_name == "digits"
    assert benchmark.train.total_points + benchmark.test.total_points == 1797
