"""Paired descriptive statistics for source-only D1 evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
import math

from .research_metrics import AttackOutcome, asr_at_budgets, asr_query_auc
from .results import ResearchResultRow
from .statistics import (
    bootstrap_interval,
)


def _individual_query_auc(row: ResearchResultRow) -> float:
    if not row.clean_correct:
        raise ValueError("query-AUC contributions require an eligible row")
    curve = asr_at_budgets(
        (AttackOutcome(True, row.query_to_success),),
        tuple(range(row.query_budget + 1)),
    )
    return asr_query_auc(curve)


def _interval(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("paired bootstrap values must be finite and non-empty")
    return bootstrap_interval(values, samples=samples, seed=seed)


def _paired_cells(
    rows: Iterable[ResearchResultRow],
    *,
    control_method: str,
    learned_method: str,
) -> dict[
    tuple[str, str, str],
    tuple[ResearchResultRow, ResearchResultRow],
]:
    selected = tuple(rows)
    if not selected:
        raise ValueError("paired source statistics require raw rows")
    allowed = {control_method, learned_method}
    grouped: dict[
        tuple[str, str, str],
        dict[str, ResearchResultRow],
    ] = defaultdict(dict)
    for row in selected:
        if not isinstance(row, ResearchResultRow) or row.method not in allowed:
            raise ValueError("paired source statistics received an unexpected method")
        key = (row.victim_family, row.victim_id, row.sample_id)
        if row.method in grouped[key]:
            raise ValueError("paired source statistics received duplicate rows")
        grouped[key][row.method] = row
    if any(set(methods) != allowed for methods in grouped.values()):
        raise ValueError("paired source cohorts do not match")
    pairs = {
        key: (methods[control_method], methods[learned_method])
        for key, methods in grouped.items()
    }
    for control, learned in pairs.values():
        if (
            control.clean_correct != learned.clean_correct
            or control.query_budget != learned.query_budget
        ):
            raise ValueError("paired source rows disagree on cohort or budget")
    return pairs


def _cell_summary(
    pairs: Sequence[tuple[ResearchResultRow, ResearchResultRow]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    eligible = tuple(
        (control, learned) for control, learned in pairs if control.clean_correct
    )
    if not eligible:
        raise ValueError("paired source condition has no eligible images")
    asr_differences = tuple(
        float(learned.success) - float(control.success) for control, learned in eligible
    )
    auc_differences = tuple(
        _individual_query_auc(learned) - _individual_query_auc(control)
        for control, learned in eligible
    )
    return {
        "paired_eligible": len(eligible),
        "asr_difference": sum(asr_differences) / len(asr_differences),
        "asr_difference_ci95": _interval(
            asr_differences,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "query_auc_difference": sum(auc_differences) / len(auc_differences),
        "query_auc_difference_ci95": _interval(
            auc_differences,
            samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "asr_wins": sum(value > 0 for value in asr_differences),
        "asr_ties": sum(value == 0 for value in asr_differences),
        "asr_losses": sum(value < 0 for value in asr_differences),
        "inference_scope": "fixed-victim paired image bootstrap",
    }


def paired_source_statistics(
    rows: Iterable[ResearchResultRow],
    *,
    learned_method: str,
    control_method: str = "score_greedy",
    bootstrap_samples: int = 10_000,
    seed: int = 17,
) -> dict[str, object]:
    """Compute paired image-bootstrap intervals without seed-level claims."""

    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < 100
    ):
        raise ValueError("paired source bootstrap requires at least 100 samples")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not control_method
        or not learned_method
        or control_method == learned_method
    ):
        raise ValueError("paired source statistic controls are invalid")
    pairs = _paired_cells(
        rows,
        control_method=control_method,
        learned_method=learned_method,
    )
    by_family_pairs: dict[
        str,
        list[tuple[ResearchResultRow, ResearchResultRow]],
    ] = defaultdict(list)
    by_victim_pairs: dict[
        tuple[str, str],
        list[tuple[ResearchResultRow, ResearchResultRow]],
    ] = defaultdict(list)
    for (family, victim_id, _), pair in pairs.items():
        by_family_pairs[family].append(pair)
        by_victim_pairs[(family, victim_id)].append(pair)

    by_family = {
        family: _cell_summary(
            family_pairs,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 100 * offset,
        )
        for offset, (family, family_pairs) in enumerate(sorted(by_family_pairs.items()))
    }
    by_victim = {
        f"{family}/{victim_id}": _cell_summary(
            victim_pairs,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 10_000 + 100 * offset,
        )
        for offset, ((family, victim_id), victim_pairs) in enumerate(
            sorted(by_victim_pairs.items())
        )
    }
    return {
        "control_method": control_method,
        "learned_method": learned_method,
        "bootstrap_samples": bootstrap_samples,
        "by_family": by_family,
        "by_victim": by_victim,
        "pooled_confidence_intervals": None,
        "pooled_intervals_omitted_reason": (
            "Families reuse CIFAR indices, so independent family resampling "
            "would not preserve cross-family pairing."
        ),
        "supports_seed_level_inference": False,
        "inference_limitation": (
            "Intervals condition on one fixed policy seed and fixed visible "
            "source victims; they are descriptive and not confirmatory."
        ),
    }


__all__ = ("paired_source_statistics",)
