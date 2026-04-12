from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SamplingConfig:
    k: int = 3
    d: int = 8
    n_min: int = 32
    n_max: int = 64
    sigma: float = 1.0
    mean_bound: float = 4.0
    min_separation: float = 3.0
    dirichlet_alpha: float = 4.0
    min_weight: float = 0.12
    max_mean_resamples: int = 200
    anisotropic_prob: float = 0.0
    anisotropic_log_scale_min: float = -1.0
    anisotropic_log_scale_max: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InterventionConfig:
    kind: str = "none"
    target_component: int = 0
    prior_strength: float = 1.25
    mechanism_compression: float = 0.65
    mechanism_shift_scale: float = 0.0
    noise_factor: float = 1.5
    noise_type: str = "scale"
    anisotropic_log_scale_min: float = -1.0
    anisotropic_log_scale_max: float = 1.0
    sample_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelConfig:
    d: int = 8
    k: int = 3
    hidden_dim: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.0
    predict_scales: bool = False
    min_scale: float = 1e-3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
