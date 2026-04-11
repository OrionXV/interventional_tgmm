from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment


@lru_cache(maxsize=None)
def all_permutations(k: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.permutations(range(k)))


def hungarian_match(pred_means: np.ndarray, true_means: np.ndarray) -> np.ndarray:
    cost = ((pred_means[:, None, :] - true_means[None, :, :]) ** 2).sum(axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros(len(row_ind), dtype=np.int64)
    perm[col_ind] = row_ind
    return perm
