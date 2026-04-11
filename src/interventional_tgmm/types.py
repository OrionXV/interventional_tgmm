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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    converged: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
