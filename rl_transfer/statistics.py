import random
import itertools
import math
from typing import Hashable, Mapping, Sequence

import numpy as np


def exact_paired_sign_flip_pvalue(
    differences: Sequence[float],
) -> float:
    """Return an exact two-sided paired randomization p-value.

    Each value must represent one independent top-level replicate. Nested
    victim and image observations must be aggregated before calling this
    function.
    """

    values = tuple(float(value) for value in differences)
    if not values or len(values) > 20:
        raise ValueError(
            "exact sign-flip inference requires 1 to 20 replicates"
        )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("paired differences must be finite")
    observed = abs(float(np.mean(values)))
    assignments = 2 ** len(values)
    exceedances = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(
            float(
                np.mean(
                    tuple(
                        sign * value
                        for sign, value in zip(signs, values)
                    )
                )
            )
        )
        exceedances += permuted >= observed - 1e-15
    return exceedances / assignments


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


def hierarchical_paired_bootstrap_interval(
    cells: Mapping[
        Hashable,
        Mapping[Hashable, Sequence[float]],
    ],
    *,
    samples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap paired differences through seed, victim, and image levels."""

    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 100:
        raise ValueError("hierarchical bootstrap requires at least 100 samples")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    normalized: dict[Hashable, dict[Hashable, np.ndarray]] = {}
    for seed_key, victim_cells in cells.items():
        if not victim_cells:
            raise ValueError("every seed requires at least one victim cell")
        normalized[seed_key] = {}
        for victim_key, values in victim_cells.items():
            array = np.asarray(tuple(values), dtype=float)
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError("every victim cell requires finite paired differences")
            normalized[seed_key][victim_key] = array
    if not normalized:
        raise ValueError("hierarchical bootstrap cells cannot be empty")

    rng = np.random.default_rng(seed)
    seed_keys = tuple(normalized)
    estimates = np.empty(samples, dtype=float)
    for sample_index in range(samples):
        sampled_seed_keys = rng.choice(
            len(seed_keys),
            size=len(seed_keys),
            replace=True,
        )
        seed_estimates: list[float] = []
        for selected_seed_index in sampled_seed_keys:
            victim_cells = normalized[seed_keys[int(selected_seed_index)]]
            victim_keys = tuple(victim_cells)
            sampled_victim_keys = rng.choice(
                len(victim_keys),
                size=len(victim_keys),
                replace=True,
            )
            victim_estimates: list[float] = []
            for selected_victim_index in sampled_victim_keys:
                values = victim_cells[victim_keys[int(selected_victim_index)]]
                victim_estimates.append(
                    float(rng.choice(values, size=len(values), replace=True).mean())
                )
            seed_estimates.append(float(np.mean(victim_estimates)))
        estimates[sample_index] = float(np.mean(seed_estimates))
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1 - alpha)),
    )
