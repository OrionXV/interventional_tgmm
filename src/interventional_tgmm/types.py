from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass(slots=True)
class GMMParams:
    weights: Array
    means: Array
    sigma: float = 1.0
    sigma_diag: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def component_scales(self) -> Array:
        if self.sigma_diag is not None:
            return self.sigma_diag.astype(np.float32, copy=False)
        return np.full_like(self.means, float(self.sigma), dtype=np.float32)


@dataclass(slots=True)
class GMMTask:
    x: Array
    z: Array
    params: GMMParams
    intervention: str = "none"


@dataclass(slots=True)
class GMMEstimate:
    weights: Array
    means: Array
    assumed_sigma: float = 1.0
    sigma_diag: Array | None = None
    converged: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def component_scales(self) -> Array:
        if self.sigma_diag is not None:
            return self.sigma_diag.astype(np.float32, copy=False)
        return np.full_like(self.means, float(self.assumed_sigma), dtype=np.float32)
