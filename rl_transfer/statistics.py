import random
from typing import Sequence

import numpy as np


def paired_permutation_pvalue(left: Sequence[float], right: Sequence[float], permutations: int = 10_000, seed: int = 0) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("paired non-empty samples are required")
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    observed = abs(float(differences.mean()))
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = np.asarray([value if rng.random() < 0.5 else -value for value in differences])
        exceedances += abs(float(permuted.mean())) >= observed
    return (exceedances + 1) / (permutations + 1)


def bootstrap_interval(values: Sequence[float], samples: int = 2_000, seed: int = 0, confidence: float = 0.95) -> tuple[float, float]:
    if not values:
        raise ValueError("values cannot be empty")
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    estimates = np.asarray([rng.choice(array, size=len(array), replace=True).mean() for _ in range(samples)])
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))
