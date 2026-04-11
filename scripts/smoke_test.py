#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interventional_tgmm.baselines import fit_em_known_sigma, fit_kmeans_baseline, fit_spectral_isotropic
from interventional_tgmm.config import InterventionConfig, ModelConfig, SamplingConfig
from interventional_tgmm.data import sample_task, sample_task_batch
from interventional_tgmm.losses import permutation_invariant_loss
from interventional_tgmm.metrics import evaluate_task
from interventional_tgmm.model import TGMMNet
from interventional_tgmm.utils import configure_torch_runtime, seed_everything


def main() -> None:
    configure_torch_runtime(1)
    seed_everything(0)
    rng = np.random.default_rng(0)
    data_cfg = SamplingConfig(k=3, d=8, n_min=32, n_max=40, sigma=1.0)
    task = sample_task(data_cfg, rng, intervention=InterventionConfig(kind="none"))

    model_cfg = ModelConfig(d=8, k=3, hidden_dim=64, n_layers=2, n_heads=4)
    model = TGMMNet(model_cfg)
    batch = sample_task_batch(data_cfg, batch_size=4, rng=rng)
    outputs = model(batch["x"].float(), batch["mask"])
    loss_dict = permutation_invariant_loss(
        pred_means=outputs["means"],
        pred_weight_logits=outputs["weight_logits"],
        true_means=batch["true_means"].float(),
        true_weights=batch["true_weights"].float(),
    )
    assert torch.isfinite(loss_dict["loss"])

    em_est = fit_em_known_sigma(task.x, k=data_cfg.k, sigma=data_cfg.sigma, restarts=3, max_iters=20, seed=0)
    km_est = fit_kmeans_baseline(task.x, k=data_cfg.k, sigma=data_cfg.sigma, seed=0)
    sp_est = fit_spectral_isotropic(task.x, k=data_cfg.k, sigma=data_cfg.sigma, seed=0)

    em_metrics = evaluate_task(task, em_est)
    km_metrics = evaluate_task(task, km_est)
    sp_metrics = evaluate_task(task, sp_est)

    print("model forward/loss ok")
    print("EM metrics", em_metrics)
    print("k-means metrics", km_metrics)
    print("spectral metrics", sp_metrics)
    print("smoke test passed")


if __name__ == "__main__":
    main()
