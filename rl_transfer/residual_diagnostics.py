"""Aggregation helpers for source-balanced residual-ranker diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math

import torch


_SOURCE_MARKER = "bc-gradient-source:"
_RATE_FIELDS = (
    "hybrid_top1_accuracy",
    "gated_top1_accuracy",
    "prior_top1_accuracy",
    "hybrid_top5_accuracy",
    "soft_cross_entropy",
    "prior_soft_cross_entropy",
    "residual_use_fraction",
)


def trajectory_source_family(trajectory_id: str) -> str:
    """Extract source-family provenance, retaining synthetic test support."""

    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("residual trajectory identity is missing")
    if _SOURCE_MARKER not in trajectory_id:
        return "unattributed"
    family = trajectory_id.split(_SOURCE_MARKER, 1)[1].split(":", 1)[0]
    if not family:
        raise ValueError("residual trajectory source family is missing")
    return family


def equal_family_tensor_mean(
    terms: Sequence[tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """Average tensors equally by trajectory and then source family."""

    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for family, term in terms:
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(
                term,
                torch.Tensor,
            )
        ):
            raise TypeError("family-weighted tensor terms are invalid")
        grouped[family].append(term)
    if not grouped:
        raise ValueError("family-weighted tensor terms cannot be empty")
    return torch.stack(
        tuple(torch.stack(tuple(grouped[family])).mean() for family in sorted(grouped))
    ).mean()


def equal_family_scalar_mean(
    values: Sequence[tuple[str, float]],
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for family, value in values:
        numeric = float(value)
        if not family or not math.isfinite(numeric):
            raise ValueError("family-weighted scalar values are invalid")
        grouped[family].append(numeric)
    if not grouped:
        raise ValueError("family-weighted scalar values cannot be empty")
    return sum(sum(grouped[family]) / len(grouped[family]) for family in grouped) / len(
        grouped
    )


def summarize_competence_trajectories(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize steps equally within trajectory and trajectories by family."""

    by_trajectory: dict[str, dict[str, object]] = {}
    for record in records:
        trajectory_id = record.get("trajectory_id")
        accepted = record.get("accepted_steps")
        if (
            not isinstance(trajectory_id, str)
            or not trajectory_id
            or isinstance(accepted, bool)
            or not isinstance(accepted, int)
            or accepted < 1
            or trajectory_id in by_trajectory
        ):
            raise ValueError("competence trajectory record is invalid")
        family = trajectory_source_family(trajectory_id)
        totals = {
            field: float(record[field])
            for field in (
                "hybrid_correct",
                "gated_correct",
                "prior_correct",
                "top5_correct",
                "soft_ce_total",
                "prior_soft_ce_total",
                "residual_uses",
            )
        }
        if any(not math.isfinite(value) for value in totals.values()):
            raise ValueError("competence trajectory metrics must be finite")
        by_trajectory[trajectory_id] = {
            "source_family": family,
            "accepted_steps": accepted,
            "hybrid_top1_accuracy": totals["hybrid_correct"] / accepted,
            "gated_top1_accuracy": totals["gated_correct"] / accepted,
            "prior_top1_accuracy": totals["prior_correct"] / accepted,
            "hybrid_top5_accuracy": totals["top5_correct"] / accepted,
            "soft_cross_entropy": totals["soft_ce_total"] / accepted,
            "prior_soft_cross_entropy": (totals["prior_soft_ce_total"] / accepted),
            "residual_use_fraction": totals["residual_uses"] / accepted,
        }
    if not by_trajectory:
        raise ValueError("competence trajectories cannot be empty")

    family_records: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for metrics in by_trajectory.values():
        family_records[str(metrics["source_family"])].append(metrics)
    by_family = {
        family: {
            "trajectories": len(items),
            "accepted_steps": sum(int(item["accepted_steps"]) for item in items),
            **{
                field: sum(float(item[field]) for item in items) / len(items)
                for field in _RATE_FIELDS
            },
        }
        for family, items in sorted(family_records.items())
    }
    by_family = {
        family: {
            **metrics,
            "accuracy_gain_vs_prior": (
                float(metrics["gated_top1_accuracy"])
                - float(metrics["prior_top1_accuracy"])
            ),
            "soft_ce_improvement_vs_prior": (
                float(metrics["prior_soft_cross_entropy"])
                - float(metrics["soft_cross_entropy"])
            ),
        }
        for family, metrics in by_family.items()
    }
    macro = {
        field: sum(float(metrics[field]) for metrics in by_family.values())
        / len(by_family)
        for field in _RATE_FIELDS
    }
    macro = {
        **macro,
        "accuracy_gain_vs_prior": (
            macro["gated_top1_accuracy"] - macro["prior_top1_accuracy"]
        ),
        "soft_ce_improvement_vs_prior": (
            macro["prior_soft_cross_entropy"] - macro["soft_cross_entropy"]
        ),
    }
    return {
        "aggregation": "equal_trajectory_then_family",
        "by_trajectory": by_trajectory,
        "by_source_family": by_family,
        "equal_family_macro": macro,
        "worst_family": {
            "accuracy_gain_vs_prior": min(
                float(metrics["accuracy_gain_vs_prior"])
                for metrics in by_family.values()
            ),
            "soft_ce_improvement_vs_prior": min(
                float(metrics["soft_ce_improvement_vs_prior"])
                for metrics in by_family.values()
            ),
        },
    }


def summarize_threshold_choices(
    observations: Sequence[tuple[str, int, int, float, int]],
    selected_actions: Sequence[int],
) -> dict[str, object]:
    """Score threshold choices equally by trajectory and source family."""

    if len(observations) != len(selected_actions) or not observations:
        raise ValueError("threshold choices do not match validation observations")
    trajectories: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for (trajectory_id, baseline, _, _, label), selected in zip(
        observations,
        selected_actions,
    ):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (baseline, label, selected)
        ):
            raise ValueError("threshold actions must be non-negative integers")
        trajectories[trajectory_id].append((selected == label, selected != baseline))
    by_trajectory = {
        trajectory_id: {
            "source_family": trajectory_source_family(trajectory_id),
            "accepted_steps": len(values),
            "accuracy": sum(correct for correct, _ in values) / len(values),
            "residual_use_fraction": sum(used for _, used in values) / len(values),
        }
        for trajectory_id, values in trajectories.items()
    }
    families: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for metrics in by_trajectory.values():
        families[str(metrics["source_family"])].append(metrics)
    by_family = {
        family: {
            "trajectories": len(values),
            "accepted_steps": sum(int(value["accepted_steps"]) for value in values),
            "accuracy": sum(float(value["accuracy"]) for value in values) / len(values),
            "residual_use_fraction": sum(
                float(value["residual_use_fraction"]) for value in values
            )
            / len(values),
        }
        for family, values in sorted(families.items())
    }
    return {
        "aggregation": "equal_trajectory_then_family",
        "accuracy": sum(float(value["accuracy"]) for value in by_family.values())
        / len(by_family),
        "residual_use_fraction": sum(
            float(value["residual_use_fraction"]) for value in by_family.values()
        )
        / len(by_family),
        "pooled_accuracy": sum(
            selected == observation[4]
            for observation, selected in zip(observations, selected_actions)
        )
        / len(observations),
        "pooled_residual_use_fraction": sum(
            selected != observation[1]
            for observation, selected in zip(observations, selected_actions)
        )
        / len(observations),
        "by_trajectory": by_trajectory,
        "by_source_family": by_family,
    }


__all__ = (
    "equal_family_scalar_mean",
    "equal_family_tensor_mean",
    "summarize_competence_trajectories",
    "summarize_threshold_choices",
    "trajectory_source_family",
)
