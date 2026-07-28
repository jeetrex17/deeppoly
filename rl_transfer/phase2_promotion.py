"""Exploratory source-only promotion rule for Phase 2 screening."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .phase2_config import FAMILIES, Phase2ScreenConfig


SCREEN_LEARNED_METHODS = (
    "soft_gradient_bc_action_conditioned_groupdro_ppo_stochastic",
    "groupdro_recurrent_ppo_stochastic",
    "gradient_bc_groupdro_ppo_stochastic",
)
SCREEN_CONTROL = "score_greedy"
SOURCE_SLICES = ("exact_source", "seen_family_new_instance")


def _require_number(
    value: object,
    *,
    label: str,
    probability: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite and numeric")
    result = float(value)
    if probability and not 0 <= result <= 1:
        raise ValueError(f"{label} must be a probability")
    return result


def _final_asr(metrics: Mapping[str, object], label: str) -> float:
    curve = metrics.get("asr_at_budgets")
    if not isinstance(curve, Mapping) or not curve:
        raise ValueError(f"{label} requires a non-empty ASR curve")
    normalized = {
        int(budget): _require_number(
            value,
            label=f"{label} ASR",
            probability=True,
        )
        for budget, value in curve.items()
    }
    return normalized[max(normalized)]


def validate_source_run_semantics(
    run: Mapping[str, object],
    *,
    family: str,
    seed: int,
) -> tuple[float, float, list[tuple[float, float]], str]:
    """Validate one source cell and return its diagnostic effect values."""

    run_id = f"{family}/seed-{seed}"
    if (
        run.get("status") != "source_complete"
        or run.get("target_evaluation_performed") is not False
        or run.get("target_calls") != 0
    ):
        raise ValueError(
            f"{run_id}: source-only completion contract failed"
        )
    if (
        run.get("seed") != seed
        or run.get("target_family") != family
        or run.get("validation_roles_disjoint") is not True
    ):
        raise ValueError(f"{run_id}: source cell identity mismatch")
    source_families = {
        item for item in FAMILIES if item != family
    }
    victim_audit = run.get("victim_access_audit")
    constructed_families = (
        victim_audit.get("constructed_families")
        if isinstance(victim_audit, Mapping)
        else None
    )
    model_counts = (
        victim_audit.get("model_instances_by_family")
        if isinstance(victim_audit, Mapping)
        else None
    )
    validation_counts = (
        victim_audit.get("validation_evaluations_by_family")
        if isinstance(victim_audit, Mapping)
        else None
    )
    if (
        not isinstance(victim_audit, Mapping)
        or not isinstance(constructed_families, list)
        or any(
            not isinstance(item, str)
            for item in constructed_families
        )
        or not isinstance(model_counts, Mapping)
        or not isinstance(validation_counts, Mapping)
        or victim_audit.get("source_victims_only") is not True
        or victim_audit.get("victim_cache_only") is not True
        or victim_audit.get("passed") is not True
        or set(constructed_families) != source_families
        or victim_audit.get("untouched_families") != [family]
        or victim_audit.get("heldout_family") != family
        or victim_audit.get("heldout_family_model_calls") != 0
        or victim_audit.get("heldout_family_validation_calls") != 0
        or model_counts.get(family) != 0
        or validation_counts.get(family) != 0
    ):
        raise ValueError(
            f"{run_id}: held-out-family isolation audit failed"
        )
    victim_gate = run.get("victim_accuracy_gate")
    if (
        not isinstance(victim_gate, Mapping)
        or victim_gate.get("passed") is not True
    ):
        raise ValueError(f"{run_id}: victim accuracy gate failed")
    audits = run.get("source_evaluation_audits")
    evaluation = run.get("source_evaluation")
    if (
        not isinstance(audits, Mapping)
        or not isinstance(evaluation, Mapping)
        or set(audits) != set(SOURCE_SLICES)
        or set(evaluation) != set(SOURCE_SLICES)
    ):
        raise ValueError(f"{run_id}: source slices are incomplete")
    gains: list[tuple[float, float]] = []
    for slice_name in SOURCE_SLICES:
        slice_audits = audits[slice_name]
        slice_evaluation = evaluation[slice_name]
        if (
            not isinstance(slice_audits, Mapping)
            or not isinstance(slice_evaluation, Mapping)
            or set(slice_audits) != source_families
            or set(slice_evaluation) != source_families
        ):
            raise ValueError(
                f"{run_id}/{slice_name}: source-family grid mismatch"
            )
        for source_family in sorted(source_families):
            audit = slice_audits[source_family]
            methods = slice_evaluation[source_family]
            if (
                not isinstance(audit, Mapping)
                or audit.get("passed") is not True
                or audit.get("expected_cohort_verified") is not True
            ):
                raise ValueError(
                    f"{run_id}/{slice_name}/{source_family}: "
                    "raw source audit failed"
                )
            if not isinstance(methods, Mapping):
                raise ValueError(
                    f"{run_id}/{slice_name}/{source_family}: "
                    "method block is invalid"
                )
            learned_name = next(
                (
                    method
                    for method in SCREEN_LEARNED_METHODS
                    if method in methods
                ),
                None,
            )
            learned = (
                methods.get(learned_name)
                if learned_name is not None
                else None
            )
            control = methods.get(SCREEN_CONTROL)
            if (
                not isinstance(learned, Mapping)
                or not isinstance(control, Mapping)
            ):
                raise ValueError(
                    f"{run_id}/{slice_name}/{source_family}: "
                    "learned or score-greedy metrics are missing"
                )
            learned_asr = _final_asr(
                learned,
                f"{run_id}/{slice_name}/{source_family}/learned",
            )
            control_asr = _final_asr(
                control,
                f"{run_id}/{slice_name}/{source_family}/control",
            )
            learned_auc = _require_number(
                learned.get("asr_query_auc"),
                label="learned AUC",
                probability=True,
            )
            control_auc = _require_number(
                control.get("asr_query_auc"),
                label="control AUC",
                probability=True,
            )
            gains.append(
                (
                    learned_asr - control_asr,
                    learned_auc - control_auc,
                )
            )
    policy = run.get("policy")
    training = (
        policy.get("training")
        if isinstance(policy, Mapping)
        else None
    )
    cloning = (
        training.get("behavior_cloning")
        if isinstance(training, Mapping)
        else None
    )
    validation = (
        cloning.get("validation")
        if isinstance(cloning, Mapping)
        else None
    )
    if (
        not isinstance(cloning, Mapping)
        or cloning.get("enabled") is not True
        or not isinstance(validation, Mapping)
    ):
        raise ValueError(
            f"{run_id}: behavior-cloning diagnostics are missing"
        )
    if (
        validation.get("baseline_provenance")
        != "evaluated_labels_validation_oracle"
        or validation.get("baseline_estimator")
        != "empirical_best_constant_no_smoothing"
    ):
        raise ValueError(
            f"{run_id}: validation-oracle provenance is invalid"
        )
    target_mode = validation.get("target_mode")
    if target_mode in {"soft", "mixed_soft_and_hard"}:
        accuracy_gain = _require_number(
            validation.get("top5_accuracy"),
            label="BC soft-target top-5 accuracy",
            probability=True,
        ) - _require_number(
            validation.get("validation_oracle_top5_accuracy"),
            label="BC validation-oracle top-5 accuracy",
            probability=True,
        )
        nll_improvement = _require_number(
            validation.get("validation_oracle_soft_cross_entropy"),
            label="BC validation-oracle soft cross-entropy",
        ) - _require_number(
            validation.get("soft_cross_entropy"),
            label="BC validation soft cross-entropy",
        )
    else:
        accuracy_gain = _require_number(
            validation.get("accuracy"),
            label="BC validation accuracy",
            probability=True,
        ) - _require_number(
            validation.get("validation_oracle_top1_accuracy"),
            label="BC validation-oracle top-1 accuracy",
            probability=True,
        )
        nll_improvement = _require_number(
            validation.get("validation_oracle_nll"),
            label="BC validation-oracle NLL",
        ) - _require_number(
            validation.get("nll"),
            label="BC validation NLL",
        )
    diagnostic_mode = (
        "soft_gradient_distillation"
        if target_mode in {"soft", "mixed_soft_and_hard"}
        else "hard_action_classification"
    )
    return accuracy_gain, nll_improvement, gains, diagnostic_mode


def screen_promotion_decision(
    runs: Sequence[Mapping[str, object]],
    config: Phase2ScreenConfig,
) -> dict[str, object]:
    """Apply an exploratory source-screen rule, never a publication gate."""

    expected = {
        (family, seed)
        for family in config.target_families
        for seed in config.seeds
    }
    observed: dict[tuple[str, int], Mapping[str, object]] = {}
    for run in runs:
        family = str(run.get("target_family"))
        raw_seed = run.get("seed")
        if (
            not isinstance(raw_seed, int)
            or isinstance(raw_seed, bool)
        ):
            raise ValueError("source screen run seed is invalid")
        key = (family, raw_seed)
        if key not in expected:
            raise ValueError(f"unexpected source screen cell: {key}")
        if key in observed:
            raise ValueError(f"duplicate source screen cell: {key}")
        observed[key] = run
    accuracy_gains: list[float] = []
    nll_improvements: list[float] = []
    condition_gains: list[tuple[float, float]] = []
    diagnostic_modes: set[str] = set()
    strict_source_gate_passes = 0
    for family, seed in sorted(observed):
        run = observed[(family, seed)]
        accuracy_gain, nll_improvement, gains, diagnostic_mode = (
            validate_source_run_semantics(
                run,
                family=family,
                seed=seed,
            )
        )
        accuracy_gains.append(accuracy_gain)
        nll_improvements.append(nll_improvement)
        condition_gains.extend(gains)
        diagnostic_modes.add(diagnostic_mode)
        source_gate = run.get("source_competence_gate")
        strict_source_gate_passes += int(
            isinstance(source_gate, Mapping)
            and source_gate.get("passed") is True
        )
    grid_complete = set(observed) == expected
    mean_accuracy_gain = (
        sum(accuracy_gains) / len(accuracy_gains)
        if accuracy_gains
        else None
    )
    mean_nll_improvement = (
        sum(nll_improvements) / len(nll_improvements)
        if nll_improvements
        else None
    )
    mean_asr_gain = (
        sum(gain[0] for gain in condition_gains)
        / len(condition_gains)
        if condition_gains
        else None
    )
    mean_auc_gain = (
        sum(gain[1] for gain in condition_gains)
        / len(condition_gains)
        if condition_gains
        else None
    )
    positive_fraction = (
        sum(
            asr_gain > 0 and auc_gain > 0
            for asr_gain, auc_gain in condition_gains
        )
        / len(condition_gains)
        if condition_gains
        else 0.0
    )
    requirements = {
        "grid_complete": grid_complete,
        "mean_bc_accuracy_gain": bool(
            mean_accuracy_gain is not None
            and mean_accuracy_gain
            >= config.minimum_mean_bc_accuracy_gain
        ),
        "mean_bc_nll_improvement": bool(
            mean_nll_improvement is not None
            and mean_nll_improvement
            >= config.minimum_mean_bc_nll_improvement
        ),
        "mean_score_asr_gain": bool(
            mean_asr_gain is not None
            and mean_asr_gain
            >= config.minimum_mean_score_asr_gain
        ),
        "mean_score_auc_gain": bool(
            mean_auc_gain is not None
            and mean_auc_gain
            >= config.minimum_mean_score_auc_gain
        ),
        "positive_condition_fraction": bool(
            positive_fraction
            >= config.minimum_positive_condition_fraction
        ),
    }
    passed = bool(all(requirements.values()))
    return {
        "passed": passed,
        "eligible_for_stage_2b": passed,
        "decision_scope": "source-only screening",
        "publication_candidate": False,
        "target_evaluation_authorized": False,
        "grid_complete": grid_complete,
        "expected_cells": len(expected),
        "completed_cells": len(observed),
        "condition_count": len(condition_gains),
        "bc_diagnostic_modes": sorted(diagnostic_modes),
        "strict_publication_source_gate_passes": (
            strict_source_gate_passes
        ),
        "metrics": {
            "mean_bc_top5_or_hard_accuracy_gain": (
                mean_accuracy_gain
            ),
            "mean_bc_loss_improvement_over_validation_oracle": (
                mean_nll_improvement
            ),
            "mean_asr_gain_over_score_greedy": mean_asr_gain,
            "mean_auc_gain_over_score_greedy": mean_auc_gain,
            "positive_asr_and_auc_condition_fraction": (
                positive_fraction
            ),
        },
        "thresholds": {
            "minimum_mean_bc_accuracy_gain": (
                config.minimum_mean_bc_accuracy_gain
            ),
            "minimum_mean_bc_nll_improvement": (
                config.minimum_mean_bc_nll_improvement
            ),
            "minimum_mean_score_asr_gain": (
                config.minimum_mean_score_asr_gain
            ),
            "minimum_mean_score_auc_gain": (
                config.minimum_mean_score_auc_gain
            ),
            "minimum_positive_condition_fraction": (
                config.minimum_positive_condition_fraction
            ),
        },
        "requirements": requirements,
        "interpretation": (
            "This exploratory gate selects whether to spend compute on a "
            "larger source-only screen. It is not the preregistered final "
            "publication gate and cannot authorize target access."
        ),
    }
