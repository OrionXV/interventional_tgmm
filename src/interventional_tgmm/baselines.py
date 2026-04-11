from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from .types import GMMEstimate


_EPS = 1e-8


def _component_log_probs(x: np.ndarray, weights: np.ndarray, means: np.ndarray, sigma: float) -> np.ndarray:
    n, d = x.shape
    sq_dist = ((x[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1)
    log_pi = np.log(np.clip(weights, _EPS, None))[None, :]
    const = -0.5 * d * np.log(2.0 * np.pi * sigma * sigma)
    return log_pi + const - 0.5 * sq_dist / (sigma * sigma)


def average_log_likelihood_np(x: np.ndarray, weights: np.ndarray, means: np.ndarray, sigma: float) -> float:
    comp = _component_log_probs(x, weights, means, sigma)
    return float(logsumexp(comp, axis=1).mean())


def _init_em(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "kmeans":
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(x)
        counts = np.bincount(labels, minlength=k).astype(np.float64) + 1.0
        weights = counts / counts.sum()
        means = km.cluster_centers_.astype(np.float64)
        return weights, means

    if method == "random_points":
        indices = rng.choice(len(x), size=k, replace=False)
        means = x[indices].astype(np.float64)
        weights = np.full(k, 1.0 / k, dtype=np.float64)
        return weights, means

    raise ValueError(f"Unknown EM init method: {method}")


def fit_em_known_sigma(
    x: np.ndarray,
    k: int,
    sigma: float = 1.0,
    restarts: int = 10,
    max_iters: int = 100,
    tol: float = 1e-4,
    init: str = "kmeans",
    seed: int = 0,
) -> GMMEstimate:
    best_ll = -np.inf
    best_weights: np.ndarray | None = None
    best_means: np.ndarray | None = None
    best_converged = False

    master_rng = np.random.default_rng(seed)
    for _ in range(restarts):
        local_seed = int(master_rng.integers(0, 10_000_000))
        local_rng = np.random.default_rng(local_seed)
        weights, means = _init_em(x, k, local_rng, init, local_seed)
        prev_ll = -np.inf
        converged = False

        for _ in range(max_iters):
            comp = _component_log_probs(x, weights, means, sigma)
            log_norm = logsumexp(comp, axis=1, keepdims=True)
            resp = np.exp(comp - log_norm)
            nk = resp.sum(axis=0) + _EPS
            weights = nk / nk.sum()
            means = (resp.T @ x) / nk[:, None]
            cur_ll = average_log_likelihood_np(x, weights, means, sigma)
            if abs(cur_ll - prev_ll) < tol:
                converged = True
                break
            prev_ll = cur_ll

        final_ll = average_log_likelihood_np(x, weights, means, sigma)
        if final_ll > best_ll:
            best_ll = final_ll
            best_weights = weights.copy()
            best_means = means.copy()
            best_converged = converged

    assert best_weights is not None and best_means is not None
    return GMMEstimate(
        weights=best_weights.astype(np.float32),
        means=best_means.astype(np.float32),
        assumed_sigma=float(sigma),
        converged=best_converged,
        metadata={"avg_log_likelihood": float(best_ll), "method": "em"},
    )


def fit_kmeans_baseline(x: np.ndarray, k: int, sigma: float = 1.0, seed: int = 0) -> GMMEstimate:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(x)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    weights = np.clip(counts / counts.sum(), _EPS, None)
    weights = weights / weights.sum()
    means = km.cluster_centers_.astype(np.float32)
    ll = average_log_likelihood_np(x, weights, means, sigma)
    return GMMEstimate(
        weights=weights.astype(np.float32),
        means=means.astype(np.float32),
        assumed_sigma=float(sigma),
        converged=True,
        metadata={"avg_log_likelihood": float(ll), "method": "kmeans"},
    )


def _compute_moments(x: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    mean_x = x.mean(axis=0)
    m2 = (x.T @ x) / float(n) - (sigma * sigma) * np.eye(d, dtype=np.float64)
    m3_raw = np.einsum("ni,nj,nk->ijk", x, x, x) / float(n)
    eye = np.eye(d, dtype=np.float64)
    correction = (sigma * sigma) * (
        np.einsum("a,bc->abc", mean_x, eye)
        + np.einsum("b,ac->abc", mean_x, eye)
        + np.einsum("c,ab->abc", mean_x, eye)
    )
    return m2, m3_raw - correction


def _tensor_vec_vec(tensor: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("ijk,j,k->i", tensor, v, v)


def _tensor_cubic(tensor: np.ndarray, v: np.ndarray) -> float:
    return float(np.einsum("ijk,i,j,k", tensor, v, v, v))


def _robust_tensor_power(
    tensor: np.ndarray,
    rank: int,
    rng: np.random.Generator,
    num_restarts: int = 20,
    num_iters: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    dim = tensor.shape[0]
    working = tensor.copy()
    lambdas: list[float] = []
    vectors: list[np.ndarray] = []

    for _ in range(rank):
        best_score = -np.inf
        best_v: np.ndarray | None = None
        for _ in range(num_restarts):
            v = rng.normal(size=dim)
            v /= np.linalg.norm(v) + _EPS
            for _ in range(num_iters):
                v_next = _tensor_vec_vec(working, v)
                norm = np.linalg.norm(v_next)
                if norm < 1e-10:
                    break
                v = v_next / norm
            score = abs(_tensor_cubic(working, v))
            if score > best_score:
                best_score = score
                best_v = v.copy()

        if best_v is None:
            raise RuntimeError("Tensor power method failed to find a candidate vector.")

        v = best_v
        for _ in range(num_iters):
            v_next = _tensor_vec_vec(working, v)
            norm = np.linalg.norm(v_next)
            if norm < 1e-10:
                break
            v = v_next / norm

        lam = _tensor_cubic(working, v)
        if lam < 0:
            v = -v
            lam = -lam
        if lam < 1e-8:
            raise RuntimeError("Tensor power method produced a degenerate eigenvalue.")

        lambdas.append(float(lam))
        vectors.append(v.copy())
        working = working - lam * np.einsum("i,j,k->ijk", v, v, v)

    return np.asarray(lambdas, dtype=np.float64), np.asarray(vectors, dtype=np.float64)


def fit_spectral_isotropic(
    x: np.ndarray,
    k: int,
    sigma: float = 1.0,
    seed: int = 0,
    num_restarts: int = 20,
    num_iters: int = 100,
) -> GMMEstimate:
    n, d = x.shape
    if k > d:
        raise ValueError("Spectral baseline requires k <= d in this implementation.")

    m2, m3 = _compute_moments(x.astype(np.float64), sigma=float(sigma))
    eigvals, eigvecs = np.linalg.eigh(m2)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if np.any(eigvals[:k] <= 1e-8):
        raise ValueError("Top-k empirical second moment eigenvalues are not strictly positive.")

    u = eigvecs[:, :k]
    s = eigvals[:k]
    w = u @ np.diag(1.0 / np.sqrt(s))
    b = u @ np.diag(np.sqrt(s))
    whitened_tensor = np.einsum("abc,ai,bj,ck->ijk", m3, w, w, w)

    rng = np.random.default_rng(seed)
    lambdas, vecs = _robust_tensor_power(
        whitened_tensor,
        rank=k,
        rng=rng,
        num_restarts=num_restarts,
        num_iters=num_iters,
    )

    weights = 1.0 / np.clip(lambdas * lambdas, _EPS, None)
    weights = weights / weights.sum()
    means = np.stack([lam * (b @ vec) for lam, vec in zip(lambdas, vecs, strict=True)], axis=0)
    ll = average_log_likelihood_np(x, weights, means, sigma)
    return GMMEstimate(
        weights=weights.astype(np.float32),
        means=means.astype(np.float32),
        assumed_sigma=float(sigma),
        converged=True,
        metadata={"avg_log_likelihood": float(ll), "method": "spectral", "eigvals": s.tolist()},
    )
