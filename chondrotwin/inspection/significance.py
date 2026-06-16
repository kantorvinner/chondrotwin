from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np


class BootstrapResult(NamedTuple):
    estimate: float
    low: float
    high: float
    p_value: float


def paired_bootstrap(
    reference: np.ndarray,
    candidate: np.ndarray,
    metric: Callable[[np.ndarray], float],
    iterations: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    n = reference.shape[0]
    observed = metric(candidate) - metric(reference)
    deltas = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        deltas[i] = metric(candidate[idx]) - metric(reference[idx])
    low = float(np.quantile(deltas, alpha / 2))
    high = float(np.quantile(deltas, 1 - alpha / 2))
    p_value = float(2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return BootstrapResult(estimate=float(observed), low=low, high=high, p_value=min(1.0, p_value))


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    rejected = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected
