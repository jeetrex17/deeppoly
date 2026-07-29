"""Conservative source-development selection gates for D1b."""

from __future__ import annotations

from collections.abc import Mapping
import math
import statistics

from .phase2_residual_d1 import (
    D1_SOURCE_FAMILIES,
    validate_source_only_payload as _source_only,
)
from .phase2_residual_d1_evaluation import (
    D1_MIN_DEPLOYMENT_OVERRIDE_FRACTION,
)
from .phase2_residual_d1b import D1B_METHODS


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _number(
    value: object,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
        or (maximum is not None and float(value) > maximum)
    ):
        raise ValueError(f"{label} must be a finite metric in range")
    return float(value)


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _competence_gate(
    competence: Mapping[str, object],
) -> tuple[bool, dict[str, float]]:
    if competence.get("target_mode") != "all_soft":
        raise ValueError("D1b competence must use all-soft teacher targets")
    if _count(competence.get("accepted_steps"), "D1b accepted steps") < 1:
        raise ValueError("D1b competence requires accepted steps")
    by_family = _mapping(
        competence.get("by_source_family"),
        "D1b competence by family",
    )
    if set(by_family) != set(D1_SOURCE_FAMILIES):
        raise ValueError("D1b competence must contain exactly the source families")
    values: dict[str, tuple[float, float, float]] = {}
    for family in D1_SOURCE_FAMILIES:
        metrics = _mapping(by_family[family], f"D1b {family} competence")
        values[family] = (
            _number(
                metrics.get("accuracy_gain_vs_prior"),
                f"D1b {family} accuracy gain",
                minimum=-1.0,
                maximum=1.0,
            ),
            _number(
                metrics.get("soft_ce_improvement_vs_prior"),
                f"D1b {family} soft-CE gain",
                minimum=-1_000_000.0,
            ),
            _number(
                metrics.get("residual_use_fraction"),
                f"D1b {family} residual use",
                maximum=1.0,
            ),
        )
    macro = _mapping(
        competence.get("equal_family_macro"),
        "D1b equal-family competence",
    )
    worst = _mapping(
        competence.get("worst_family"),
        "D1b worst-family competence",
    )
    computed = {
        "macro_accuracy_gain_vs_prior": statistics.fmean(
            value[0] for value in values.values()
        ),
        "macro_soft_ce_improvement_vs_prior": statistics.fmean(
            value[1] for value in values.values()
        ),
        "macro_residual_use_fraction": statistics.fmean(
            value[2] for value in values.values()
        ),
        "worst_family_accuracy_gain_vs_prior": min(
            value[0] for value in values.values()
        ),
        "worst_family_soft_ce_improvement_vs_prior": min(
            value[1] for value in values.values()
        ),
    }
    recorded = {
        "macro_accuracy_gain_vs_prior": _number(
            macro.get("accuracy_gain_vs_prior"),
            "D1b macro accuracy gain",
            minimum=-1.0,
            maximum=1.0,
        ),
        "macro_soft_ce_improvement_vs_prior": _number(
            macro.get("soft_ce_improvement_vs_prior"),
            "D1b macro soft-CE gain",
            minimum=-1_000_000.0,
        ),
        "macro_residual_use_fraction": _number(
            macro.get("residual_use_fraction"),
            "D1b macro residual use",
            maximum=1.0,
        ),
        "worst_family_accuracy_gain_vs_prior": _number(
            worst.get("accuracy_gain_vs_prior"),
            "D1b worst-family accuracy gain",
            minimum=-1.0,
            maximum=1.0,
        ),
        "worst_family_soft_ce_improvement_vs_prior": _number(
            worst.get("soft_ce_improvement_vs_prior"),
            "D1b worst-family soft-CE gain",
            minimum=-1_000_000.0,
        ),
    }
    if any(
        not math.isclose(recorded[key], value, abs_tol=1e-12)
        for key, value in computed.items()
    ):
        raise ValueError("D1b recorded competence aggregation is inconsistent")
    passed = (
        computed["macro_accuracy_gain_vs_prior"] > 0
        and computed["macro_soft_ce_improvement_vs_prior"] > 0
        and computed["macro_residual_use_fraction"] > 0
        and computed["worst_family_accuracy_gain_vs_prior"] > 0
        and computed["worst_family_soft_ce_improvement_vs_prior"] > 0
    )
    return passed, computed


def _method_metrics(
    value: object,
    *,
    family: str,
    method: str,
) -> dict[str, float | int]:
    metrics = _mapping(value, f"D1b {family} {method} metrics")
    eligible = _count(metrics.get("eligible"), f"D1b {family} eligible")
    successes = _count(metrics.get("successes"), f"D1b {family} successes")
    if eligible < 1 or successes > eligible:
        raise ValueError("D1b eligible cohort or success count is invalid")
    result: dict[str, float | int] = {
        "eligible": eligible,
        "successes": successes,
        "final_asr": successes / eligible,
        "asr_query_auc": _number(
            metrics.get("asr_query_auc"),
            f"D1b {family} {method} query AUC",
            maximum=1.0,
        ),
    }
    if method != "score_greedy":
        result["learned_override_decisions"] = _count(
            metrics.get("learned_override_decisions"),
            f"D1b {family} {method} learned decisions",
        )
        result["score_fallback_decisions"] = _count(
            metrics.get("score_fallback_decisions"),
            f"D1b {family} {method} fallback decisions",
        )
    return result


def residual_d1b_selection_decision(
    competence: Mapping[str, object],
    threshold: Mapping[str, object],
    conditions: Mapping[str, object],
) -> dict[str, object]:
    """Choose PPO only after conservative source-only development gates pass."""

    competence = _mapping(competence, "D1b competence")
    threshold = _mapping(threshold, "D1b threshold")
    conditions = _mapping(conditions, "D1b evaluation conditions")
    _source_only(competence, "D1b competence")
    _source_only(threshold, "D1b threshold")
    _source_only(conditions, "D1b conditions")
    if threshold.get("selection_role") != "d1b_threshold_selection_only":
        raise ValueError("D1b threshold must use only the threshold role")
    _number(threshold.get("threshold"), "D1b confidence threshold")
    if _count(threshold.get("accepted_steps"), "D1b threshold steps") < 1:
        raise ValueError("D1b threshold selection requires accepted steps")
    if not isinstance(threshold.get("overrides_enabled"), bool):
        raise ValueError("D1b threshold override flag is invalid")
    if set(conditions) != set(D1_SOURCE_FAMILIES):
        raise ValueError("D1b evaluation conditions must match source families")

    competence_passed, competence_summary = _competence_gate(competence)
    observed: dict[str, object] = {}
    bc_non_decrease: list[bool] = []
    ppo_vs_score: list[bool] = []
    ppo_vs_bc: list[bool] = []
    ppo_learned = 0
    ppo_fallback = 0
    for family in D1_SOURCE_FAMILIES:
        condition = _mapping(conditions[family], f"D1b {family} condition")
        audit = _mapping(condition.get("audit"), f"D1b {family} audit")
        if audit.get("passed") is not True:
            raise ValueError("D1b condition contains a failed evaluation audit")
        methods = _mapping(condition.get("methods"), f"D1b {family} methods")
        if set(methods) != set(D1B_METHODS):
            raise ValueError("D1b condition methods violate the locked comparison")
        metrics = {
            method: _method_metrics(
                methods[method],
                family=family,
                method=method,
            )
            for method in D1B_METHODS
        }
        if len({int(item["eligible"]) for item in metrics.values()}) != 1:
            raise ValueError("D1b compared methods use mismatched eligible cohorts")
        score = metrics["score_greedy"]
        bc = metrics["residual_ranker_bc"]
        ppo = metrics["residual_ranker_bc_ppo"]
        bc_family = float(bc["final_asr"]) >= float(score["final_asr"]) and float(
            bc["asr_query_auc"]
        ) >= float(score["asr_query_auc"])
        ppo_score_family = float(ppo["final_asr"]) >= float(
            score["final_asr"]
        ) and float(ppo["asr_query_auc"]) >= float(score["asr_query_auc"])
        ppo_bc_family = float(ppo["final_asr"]) >= float(bc["final_asr"]) and float(
            ppo["asr_query_auc"]
        ) >= float(bc["asr_query_auc"])
        bc_non_decrease.append(bc_family)
        ppo_vs_score.append(ppo_score_family)
        ppo_vs_bc.append(ppo_bc_family)
        ppo_learned += int(ppo["learned_override_decisions"])
        ppo_fallback += int(ppo["score_fallback_decisions"])
        observed[family] = {
            "score_greedy": score,
            "residual_ranker_bc": bc,
            "residual_ranker_bc_ppo": ppo,
            "bc_vs_score_asr_gap": (float(bc["final_asr"]) - float(score["final_asr"])),
            "bc_vs_score_auc_gap": (
                float(bc["asr_query_auc"]) - float(score["asr_query_auc"])
            ),
            "ppo_vs_score_asr_gap": (
                float(ppo["final_asr"]) - float(score["final_asr"])
            ),
            "ppo_vs_score_auc_gap": (
                float(ppo["asr_query_auc"]) - float(score["asr_query_auc"])
            ),
            "ppo_vs_bc_asr_gap": (float(ppo["final_asr"]) - float(bc["final_asr"])),
            "ppo_vs_bc_auc_gap": (
                float(ppo["asr_query_auc"]) - float(bc["asr_query_auc"])
            ),
            "bc_observed_non_decrease_vs_score": bc_family,
            "ppo_observed_non_decrease_vs_score": ppo_score_family,
            "ppo_observed_non_decrease_vs_bc": ppo_bc_family,
        }

    deployment_total = ppo_learned + ppo_fallback
    deployment_fraction = ppo_learned / deployment_total if deployment_total else 0.0
    threshold_separate = (
        threshold.get("selection_role") == "d1b_threshold_selection_only"
    )
    deployment_passed = (
        threshold.get("overrides_enabled") is True
        and deployment_fraction >= D1_MIN_DEPLOYMENT_OVERRIDE_FRACTION
    )
    bc_passed = all(bc_non_decrease)
    ppo_score_passed = all(ppo_vs_score)
    ppo_bc_passed = all(ppo_vs_bc)
    ppo_viable = (
        competence_passed
        and threshold_separate
        and deployment_passed
        and ppo_score_passed
    )
    selected = (
        "residual_ranker_bc_ppo"
        if bc_passed and ppo_viable and ppo_bc_passed
        else "residual_ranker_bc"
        if bc_passed
        else None
    )
    if selected == "residual_ranker_bc_ppo":
        ppo_rejection_reason = None
    elif not competence_passed or not deployment_passed:
        ppo_rejection_reason = "ppo_source_competence_or_deployment_gate_failed"
    elif not ppo_score_passed:
        ppo_rejection_reason = "ppo_observed_regression_against_score_greedy"
    elif not ppo_bc_passed:
        ppo_rejection_reason = "ppo_observed_regression_against_frozen_bc"
    else:
        ppo_rejection_reason = "frozen_bc_reproduction_gate_failed"
    return {
        "passed": bc_passed,
        "selected_method": selected,
        "failure_reason": (
            None if bc_passed else "frozen_bc_did_not_reproduce_on_reserved_d1b_cohort"
        ),
        "ppo_rejection_reason": ppo_rejection_reason,
        "bc_reproduction_gate_passed": bc_passed,
        "ppo_competence_gate_passed": competence_passed,
        "ppo_threshold_role_gate_passed": threshold_separate,
        "ppo_deployment_gate_passed": deployment_passed,
        "ppo_vs_score_gate_passed": ppo_score_passed,
        "ppo_vs_bc_gate_passed": ppo_bc_passed,
        "ppo_deployment_override_fraction": deployment_fraction,
        "minimum_deployment_override_fraction": (D1_MIN_DEPLOYMENT_OVERRIDE_FRACTION),
        "competence_summary": competence_summary,
        "observed_point_estimate_gaps": observed,
        "inference_scope": (
            "fixed-seed source-development point estimates; observed "
            "non-decrease is not an inferential non-inferiority test"
        ),
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


__all__ = ("residual_d1b_selection_decision",)
