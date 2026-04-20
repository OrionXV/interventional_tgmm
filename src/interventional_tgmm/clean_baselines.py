from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture

from .baselines import fit_em_known_sigma, fit_kmeans_baseline, fit_spectral_isotropic
from .types import GMMEstimate


@dataclass(slots=True)
class BaselineResult:
    estimate: GMMEstimate
    avg_log_likelihood: float | None = None
    extra: dict[str, Any] | None = None


def fit_diag_gmm_sklearn(
    x: np.ndarray,
    k: int,
    restarts: int = 10,
    max_iters: int = 200,
    reg_covar: float = 1e-6,
    seed: int = 0,
) -> BaselineResult:
    gm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        n_init=restarts,
        max_iter=max_iters,
        reg_covar=reg_covar,
        random_state=seed,
        init_params="kmeans",
    )
    gm.fit(x)
    sigma_diag = np.sqrt(np.clip(gm.covariances_, reg_covar, None)).astype(np.float32)
    estimate = GMMEstimate(
        weights=gm.weights_.astype(np.float32),
        means=gm.means_.astype(np.float32),
        assumed_sigma=float(np.mean(sigma_diag)),
        sigma_diag=sigma_diag,
        converged=bool(gm.converged_),
        metadata={
            "method": "diag_gmm",
            "covariance_type": "diag",
            "n_iter": int(gm.n_iter_),
            "lower_bound": float(gm.lower_bound_),
        },
    )
    return BaselineResult(
        estimate=estimate,
        avg_log_likelihood=float(gm.score(x)),
        extra={"bic": float(gm.bic(x)), "aic": float(gm.aic(x))},
    )


def fit_method(
    method: str,
    x: np.ndarray,
    k: int,
    sigma: float,
    seed: int,
    em_restarts: int = 10,
    spectral_restarts: int = 20,
) -> BaselineResult:
    if method == "em":
        estimate = fit_em_known_sigma(x, k=k, sigma=sigma, restarts=em_restarts, seed=seed)
        ll = estimate.metadata.get("avg_log_likelihood") if estimate.metadata is not None else None
        return BaselineResult(estimate=estimate, avg_log_likelihood=None if ll is None else float(ll))
    if method == "kmeans":
        estimate = fit_kmeans_baseline(x, k=k, sigma=sigma, seed=seed)
        ll = estimate.metadata.get("avg_log_likelihood") if estimate.metadata is not None else None
        return BaselineResult(estimate=estimate, avg_log_likelihood=None if ll is None else float(ll))
    if method == "spectral":
        estimate = fit_spectral_isotropic(x, k=k, sigma=sigma, seed=seed, num_restarts=spectral_restarts)
        ll = estimate.metadata.get("avg_log_likelihood") if estimate.metadata is not None else None
        return BaselineResult(estimate=estimate, avg_log_likelihood=None if ll is None else float(ll))
    if method == "diag_gmm":
        return fit_diag_gmm_sklearn(x, k=k, seed=seed, restarts=em_restarts)
    raise ValueError(f"Unknown method: {method}")
