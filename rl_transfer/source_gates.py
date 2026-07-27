"""Fail-closed source-competence validation for transfer experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from .research_metrics import asr_query_auc


LEARNED_METHOD = "groupdro_recurrent_ppo_stochastic"
HYBRID_LEARNED_METHOD = "gradient_bc_groupdro_ppo_stochastic"
MATCHED_CONTROLS = ("random_action", "bandit_action", "score_greedy")
REQUIRED_SLICES = ("exact_source", "seen_family_new_instance")


@dataclass(frozen=True)
class SourceGateThresholds:
    minimum_asr_gain: float = 0.05
    minimum_auc_gain: float = 0.02
    entropy_min: float = 0.10
    entropy_max: float = 0.95

    def __post_init__(self) -> None:
        gains = (self.minimum_asr_gain, self.minimum_auc_gain)
        if any(not math.isfinite(value) or not 0 < value <= 1 for value in gains):
            raise ValueError("source-gate gains must be finite and in (0, 1]")
        if not 0 <= self.entropy_min < self.entropy_max <= 1:
            raise ValueError("source-gate entropy bounds are invalid")


def _numeric_metric(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{key} must be a finite probability")
    return numeric


def _final_asr(metrics: Mapping[str, object]) -> float:
    curve = metrics.get("asr_at_budgets")
    if not isinstance(curve, Mapping) or not curve:
        raise ValueError("ASR curve is required")
    normalized = dict(
        sorted((int(key), float(value)) for key, value in curve.items())
    )
    if any(
        budget < 0 or not math.isfinite(value) or not 0 <= value <= 1
        for budget, value in normalized.items()
    ):
        raise ValueError("ASR curves must contain finite probabilities")
    if any(
        later < earlier
        for earlier, later in zip(
            normalized.values(),
            tuple(normalized.values())[1:],
        )
    ):
        raise ValueError("ASR curves must be non-decreasing")
    return float(normalized[max(normalized)])


def _validate_family(
    methods: Mapping[str, object],
    thresholds: SourceGateThresholds,
) -> dict[str, object]:
    learned_method = (
        HYBRID_LEARNED_METHOD
        if HYBRID_LEARNED_METHOD in methods
        else LEARNED_METHOD
    )
    required = {learned_method, *MATCHED_CONTROLS}
    if not required.issubset(methods):
        raise ValueError("source slice is missing a learned or matched control method")
    selected = {
        method: methods[method]
        for method in (learned_method, *MATCHED_CONTROLS)
    }
    if any(not isinstance(metrics, Mapping) for metrics in selected.values()):
        raise ValueError("method metrics must be mappings")
    alignments = set()
    for metrics in selected.values():
        eligible = metrics.get("eligible")
        query_budget = metrics.get("query_budget")
        if (
            not isinstance(eligible, int)
            or isinstance(eligible, bool)
            or eligible <= 0
            or not isinstance(query_budget, int)
            or isinstance(query_budget, bool)
            or query_budget < 2
        ):
            raise ValueError("source evaluation requires positive eligibility and budget")
        successes = metrics.get("successes")
        max_calls = metrics.get("max_total_target_calls")
        if (
            not isinstance(successes, int)
            or isinstance(successes, bool)
            or not 0 <= successes <= eligible
            or not isinstance(max_calls, int)
            or isinstance(max_calls, bool)
            or not 1 <= max_calls <= query_budget
        ):
            raise ValueError("source success and call counts are invalid")
        final_asr = _final_asr(metrics)
        if not math.isclose(final_asr, successes / eligible, abs_tol=1e-12):
            raise ValueError("source final ASR is inconsistent with its counts")
        curve = {
            int(key): float(value)
            for key, value in metrics["asr_at_budgets"].items()
        }
        if max(curve) != query_budget:
            raise ValueError("source curve does not end at the query budget")
        if not math.isclose(
            _numeric_metric(metrics, "asr_query_auc"),
            asr_query_auc(curve),
            abs_tol=1e-12,
        ):
            raise ValueError("source AUC is inconsistent with its curve")
        if metrics.get("initialization_included") is not True:
            raise ValueError("source query budget must include initialization")
        if metrics.get("frozen") is not True:
            raise ValueError("source evaluation policies must be frozen")
        before = metrics.get("policy_digest_before")
        after = metrics.get("policy_digest_after")
        if not isinstance(before, str) or not before or before != after:
            raise ValueError("source evaluation requires stable policy digests")
        alignment = (
            eligible,
            metrics.get("eligible_sample_ids_sha256"),
            query_budget,
            metrics.get("operator_digest"),
        )
        if any(
            not isinstance(value, str) or not value
            for value in (alignment[1], alignment[3])
        ):
            raise ValueError("source evaluation identity digests are required")
        alignments.add(alignment)
    if len(alignments) != 1:
        raise ValueError("source methods must share samples, budgets, and operator")

    learned = selected[learned_method]
    learned_asr = _final_asr(learned)
    learned_auc = _numeric_metric(learned, "asr_query_auc")
    entropy = _numeric_metric(learned, "normalized_action_entropy")
    comparisons: dict[str, object] = {}
    passed = thresholds.entropy_min <= entropy <= thresholds.entropy_max
    for control in MATCHED_CONTROLS:
        control_metrics = selected[control]
        asr_gain = learned_asr - _final_asr(control_metrics)
        auc_gain = learned_auc - _numeric_metric(
            control_metrics,
            "asr_query_auc",
        )
        control_passed = (
            asr_gain >= thresholds.minimum_asr_gain
            and auc_gain >= thresholds.minimum_auc_gain
        )
        passed = passed and control_passed
        comparisons[control] = {
            "asr_gain": asr_gain,
            "auc_gain": auc_gain,
            "passed": control_passed,
        }
    return {
        "passed": bool(passed),
        "learned_asr": learned_asr,
        "learned_auc": learned_auc,
        "learned_action_entropy": entropy,
        "learned_method": learned_method,
        "comparisons": comparisons,
    }


def summarize_source_competence(
    evaluation: Mapping[str, object],
    thresholds: SourceGateThresholds | None = None,
) -> dict[str, object]:
    """Validate both source slices and return evidence instead of raising."""

    selected_thresholds = thresholds or SourceGateThresholds()
    details: dict[str, object] = {}
    errors: list[str] = []
    for slice_name in REQUIRED_SLICES:
        families = evaluation.get(slice_name)
        if not isinstance(families, Mapping) or not families:
            errors.append(f"missing non-empty source slice: {slice_name}")
            continue
        slice_details: dict[str, object] = {}
        for family, methods in families.items():
            try:
                if not isinstance(methods, Mapping):
                    raise ValueError("family method block must be a mapping")
                slice_details[str(family)] = _validate_family(
                    methods,
                    selected_thresholds,
                )
            except (TypeError, ValueError) as error:
                errors.append(f"{slice_name}/{family}: {error}")
        details[slice_name] = {
            "passed": bool(
                slice_details
                and all(
                    bool(result["passed"])
                    for result in slice_details.values()
                    if isinstance(result, Mapping)
                )
                and len(slice_details) == len(families)
            ),
            "families": slice_details,
        }
    passed = bool(
        not errors
        and set(details) == set(REQUIRED_SLICES)
        and all(bool(value["passed"]) for value in details.values())
    )
    return {
        "passed": passed,
        "thresholds": asdict(selected_thresholds),
        "slices": details,
        "errors": errors,
    }
