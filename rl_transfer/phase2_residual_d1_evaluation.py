"""Paired held-out-source evaluation and decision gates for Phase 2 D1."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import math
import re
import time

import torch
from torch import nn

from .cifar_evaluation import _method_summary, _victim_performance
from .config import AttackConfig
from .evaluation_audit import audit_evaluation
from .phase2_residual_d1 import (
    D1_HELDOUT_FAMILY,
    D1_SOURCE_FAMILIES,
    ResidualD1Request,
    residual_d1_promotion_decision,
    validate_d1_attack_contract,
)
from .phase2_residual_d1_source import D1SourceContext, _mapping
from .research_metrics import AttackOutcome
from .research_protocol import FrozenEpisodeResult, run_score_greedy_episode
from .residual_ranker import ResidualRankerPolicy, run_residual_ranker_episode
from .results import ResearchResultRow


D1_MIN_DEPLOYMENT_OVERRIDE_FRACTION = 0.01
_METHOD_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_VICTIM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SCORE_GREEDY = "score_greedy"
_SOURCE_SLICES = frozenset({"exact_source", "seen_family_new_instance"})


def _validated_policies(
    policies: Mapping[str, ResidualRankerPolicy], attack: AttackConfig
) -> dict[str, ResidualRankerPolicy]:
    if not isinstance(policies, Mapping):
        raise TypeError("D1 residual policies must be a mapping")
    selected = dict(policies)
    if not selected:
        raise ValueError("D1 evaluation requires one or more residual policies")
    for method, policy in selected.items():
        if (
            not isinstance(method, str)
            or _METHOD_NAME.fullmatch(method) is None
            or method == _SCORE_GREEDY
        ):
            raise ValueError(
                "D1 residual policy method names must use safe identifiers"
            )
        if not isinstance(policy, ResidualRankerPolicy):
            raise TypeError("D1 evaluation requires residual-ranker policies")
        if (
            policy.action_dim != attack.action_dim
            or policy.backbone.observation_dim != attack.recurrent_observation_dim
        ):
            raise ValueError(
                "D1 residual policy dimensions do not match the attack contract"
            )
    return selected


def _validated_victims(
    victims: Sequence[tuple[str, nn.Module]], *, family: str
) -> tuple[tuple[str, nn.Module], ...]:
    if family not in D1_SOURCE_FAMILIES:
        raise ValueError("D1 evaluation accepts only locked source families")
    try:
        selected = tuple(victims)
    except TypeError as error:
        raise ValueError("D1 evaluation requires named source victims") from error
    if not selected:
        raise ValueError("D1 evaluation requires named source victims")
    victim_ids: set[str] = set()
    for item in selected:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or _VICTIM_ID.fullmatch(item[0]) is None
            or not isinstance(item[1], nn.Module)
            or item[0] in victim_ids
        ):
            raise ValueError("D1 source victims require unique, safe IDs and modules")
        victim_ids.add(item[0])
    return selected


def _validated_cohort(
    samples: Sequence[tuple[torch.Tensor, int]],
    indices: Sequence[int],
    *,
    attack: AttackConfig,
) -> tuple[tuple[tuple[torch.Tensor, int], ...], tuple[int, ...]]:
    try:
        selected_samples = tuple(samples)
        selected_indices = tuple(indices)
    except TypeError as error:
        raise ValueError("D1 evaluation requires an explicit cohort") from error
    if (
        not selected_samples
        or len(selected_samples) != len(selected_indices)
        or len(set(selected_indices)) != len(selected_indices)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in selected_indices
        )
    ):
        raise ValueError(
            "D1 samples and unique non-negative indices must align exactly"
        )
    for sample in selected_samples:
        if not isinstance(sample, tuple) or len(sample) != 2:
            raise ValueError("D1 evaluation samples must be image-label tuples")
        image, label = sample
        if (
            not isinstance(image, torch.Tensor)
            or image.ndim != 3
            or image.shape[0] != 3
            or not image.is_floating_point()
            or not bool(torch.isfinite(image).all())
            or image.shape[1] < attack.grid_size
            or image.shape[2] < attack.grid_size
            or image.shape[1] % attack.grid_size
            or image.shape[2] % attack.grid_size
            or isinstance(label, bool)
            or not isinstance(label, int)
            or not 0 <= label <= 9
        ):
            raise ValueError(
                "D1 samples require finite grid-aligned RGB tensors and labels"
            )
    return selected_samples, selected_indices


def _validate_episode_result(
    result: FrozenEpisodeResult,
    *,
    sample_id: str,
    victim_id: str,
    family: str,
    attack: AttackConfig,
    expected_policy_digest: str | None,
) -> None:
    if not isinstance(result, FrozenEpisodeResult):
        raise TypeError("D1 evaluation returned an invalid episode result")
    if (
        result.sample_id != sample_id
        or result.victim_id != victim_id
        or result.family != family
    ):
        raise ValueError("D1 episode result changed the explicit cohort identity")
    if (
        isinstance(result.total_target_calls, bool)
        or not isinstance(result.total_target_calls, int)
        or not 1 <= result.total_target_calls <= attack.max_queries
        or len(result.actions) != result.total_target_calls - 1
        or len(result.query_trace) != result.total_target_calls
        or any(
            isinstance(action, bool)
            or not isinstance(action, int)
            or not 0 <= action < attack.action_dim
            for action in result.actions
        )
    ):
        raise ValueError("D1 episode result violates the locked query budget")
    if (
        not math.isfinite(result.linf)
        or not math.isfinite(result.l2)
        or result.linf < 0
        or result.l2 < 0
        or result.linf > attack.epsilon + 1e-6
    ):
        raise ValueError("D1 episode result violates the perturbation contract")
    if (
        not isinstance(result.policy_digest_before, str)
        or not result.policy_digest_before
        or result.policy_digest_before != result.policy_digest_after
        or (
            expected_policy_digest is not None
            and result.policy_digest_before != expected_policy_digest
        )
    ):
        raise ValueError("D1 evaluation policy changed or has the wrong identity")
    for offset, event in enumerate(result.query_trace, start=1):
        if (
            not isinstance(event, Mapping)
            or event.get("call_index") != offset
            or event.get("sample_id") != sample_id
            or event.get("victim_id") != victim_id
            or event.get("feedback") != "scores"
            or event.get("error") is not None
            or not isinstance(event.get("purpose"), str)
            or not event.get("purpose")
            or (offset == 1 and event.get("purpose") != "initialization")
            or (offset > 1 and event.get("purpose") == "initialization")
        ):
            raise ValueError("D1 evaluation query trace is incomplete or invalid")


def _result_row(
    result: FrozenEpisodeResult,
    *,
    method: str,
    family: str,
    seed: int,
    query_budget: int,
) -> ResearchResultRow:
    return ResearchResultRow(
        sample_id=result.sample_id,
        victim_id=result.victim_id,
        victim_family=family,
        method=method,
        threat_model="T1",
        seed=seed,
        query_budget=query_budget,
        clean_correct=result.clean_correct,
        success=result.success,
        query_to_success=result.query_to_success,
        total_target_calls=result.total_target_calls,
        linf=result.linf,
        l2=result.l2,
        policy_digest=result.policy_digest_after,
        action_trace=result.actions,
    )


def _trace_record(
    result: FrozenEpisodeResult,
    *,
    method: str,
    family: str,
    heldout_family: str,
    source_slice: str,
) -> dict[str, object]:
    return {
        "method": method,
        **result.as_dict(),
        "actions": list(result.actions),
        "query_trace": [dict(event) for event in result.query_trace],
        "victim_family": family,
        "heldout_family": heldout_family,
        "source_slice": source_slice,
        "target_calls": 0,
        "hidden_target_calls": 0,
    }


def _cohort_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(sample_ids)).encode()).hexdigest()


def _trace_and_cohort_audit(
    rows: Sequence[ResearchResultRow],
    traces: Sequence[Mapping[str, object]],
    *,
    methods: Sequence[str],
    expected_sample_ids: set[str],
) -> dict[str, object]:
    row_identities = Counter(
        (row.method, row.sample_id, row.victim_id, row.victim_family) for row in rows
    )
    trace_identities = Counter(
        (
            trace.get("method"),
            trace.get("sample_id"),
            trace.get("victim_id"),
            trace.get("victim_family"),
        )
        for trace in traces
    )
    if row_identities != trace_identities or len(rows) != len(traces):
        raise ValueError("D1 full traces do not match the result-row multiset")
    method_cohorts = {
        method: tuple(row.sample_id for row in rows if row.method == method)
        for method in methods
    }
    if any(
        len(sample_ids) != len(expected_sample_ids)
        or set(sample_ids) != expected_sample_ids
        for sample_ids in method_cohorts.values()
    ):
        raise ValueError("D1 methods do not match the exact supplied cohort")
    method_cohort_sha256 = {
        method: _cohort_digest(sample_ids)
        for method, sample_ids in method_cohorts.items()
    }
    expected_digest = _cohort_digest(tuple(expected_sample_ids))
    if set(method_cohort_sha256.values()) != {expected_digest}:
        raise ValueError("D1 method cohort digests do not match")
    if any(
        trace.get("target_calls") != 0
        or trace.get("hidden_target_calls") != 0
        or trace.get("family") != trace.get("victim_family")
        or len(tuple(trace.get("query_trace", ()))) != trace.get("total_target_calls")
        for trace in traces
    ):
        raise ValueError("D1 traces violate the source-only query contract")
    return {
        "trace_result_identity_multiset_matched": True,
        "full_trace_count": len(traces),
        "expected_sample_ids_sha256": expected_digest,
        "method_cohort_sha256": method_cohort_sha256,
        "method_row_counts": {
            method: len(ids) for method, ids in method_cohorts.items()
        },
        "hidden_target_calls": 0,
    }


def _summarize_method(
    rows: list[ResearchResultRow],
    traces: Sequence[Mapping[str, object]],
    *,
    attack: AttackConfig,
    policy: ResidualRankerPolicy | None,
    digest_before: str,
    victim_ids: Sequence[str],
) -> dict[str, object]:
    summary = _method_summary(
        rows,
        [AttackOutcome(row.clean_correct, row.query_to_success) for row in rows],
        [
            bool(
                trace["query_trace"]
                and trace["query_trace"][0]["purpose"] == "initialization"
            )
            for trace in traces
        ],
        attack,
        policy,
        digest_before,
        True,
        len(victim_ids),
    )
    summary["by_victim"] = {
        victim_id: _victim_performance(
            [row for row in rows if row.victim_id == victim_id],
            attack,
        )
        for victim_id in victim_ids
    }
    return summary


def evaluate_residual_policy_cohort(
    *,
    policies: Mapping[str, ResidualRankerPolicy],
    victims: Sequence[tuple[str, nn.Module]],
    samples: Sequence[tuple[torch.Tensor, int]],
    indices: Sequence[int],
    attack: AttackConfig,
    family: str,
    seed: int,
    heldout_family: str,
    source_slice: str,
    deadline_check: Callable[[], None],
    progress: Callable[[str], None],
) -> tuple[dict[str, object], list[ResearchResultRow], list[dict[str, object]]]:
    """Evaluate score-greedy and named residual policies on one exact cohort."""

    started = time.monotonic()
    validate_d1_attack_contract(attack, D1_SOURCE_FAMILIES)
    if (
        heldout_family != D1_HELDOUT_FAMILY
        or source_slice not in _SOURCE_SLICES
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("D1 evaluation source-only metadata is invalid")
    if not callable(deadline_check) or not callable(progress):
        raise TypeError("D1 evaluation callbacks must be callable")
    selected_policies = _validated_policies(policies, attack)
    selected_victims = _validated_victims(victims, family=family)
    selected_samples, selected_indices = _validated_cohort(
        samples,
        indices,
        attack=attack,
    )
    methods = (_SCORE_GREEDY, *selected_policies)
    rows_by_method: dict[str, list[ResearchResultRow]] = {
        method: [] for method in methods
    }
    traces_by_method: dict[str, list[dict[str, object]]] = {
        method: [] for method in methods
    }
    policy_digests = {
        name: policy.persistent_digest() for name, policy in selected_policies.items()
    }
    learned_decisions = {method: 0 for method in selected_policies}
    fallback_decisions = {method: 0 for method in selected_policies}
    score_digest: str | None = None
    progress(
        f"[d1] paired held-out-source evaluation on {family}: "
        f"{len(selected_samples)} images x {len(selected_victims)} victim "
        f"x {len(methods)} methods"
    )
    for victim_id, victim in selected_victims:
        for (image, label), index in zip(
            selected_samples,
            selected_indices,
        ):
            sample_id = f"cifar10:{family}:{victim_id}:{index}"
            deadline_check()
            score = run_score_greedy_episode(
                victim,
                image,
                label,
                sample_id,
                victim_id,
                family,
                attack,
                seed,
                deadline_check=deadline_check,
            )
            _validate_episode_result(
                score,
                sample_id=sample_id,
                victim_id=victim_id,
                family=family,
                attack=attack,
                expected_policy_digest=score_digest,
            )
            score_digest = score_digest or score.policy_digest_before
            rows_by_method[_SCORE_GREEDY].append(
                _result_row(
                    score,
                    method=_SCORE_GREEDY,
                    family=family,
                    seed=seed,
                    query_budget=attack.max_queries,
                )
            )
            traces_by_method[_SCORE_GREEDY].append(
                _trace_record(
                    score,
                    method=_SCORE_GREEDY,
                    family=family,
                    heldout_family=heldout_family,
                    source_slice=source_slice,
                )
            )
            for method, policy in selected_policies.items():
                deadline_check()
                learned = run_residual_ranker_episode(
                    policy,
                    victim,
                    image,
                    label,
                    sample_id,
                    victim_id,
                    family,
                    attack,
                    score_prior_seed=seed,
                    deadline_check=deadline_check,
                )
                _validate_episode_result(
                    learned,
                    sample_id=sample_id,
                    victim_id=victim_id,
                    family=family,
                    attack=attack,
                    expected_policy_digest=policy_digests[method],
                )
                if score.clean_correct != learned.clean_correct:
                    raise ValueError("D1 paired methods disagree on clean eligibility")
                rows_by_method[method].append(
                    _result_row(
                        learned,
                        method=method,
                        family=family,
                        seed=seed,
                        query_budget=attack.max_queries,
                    )
                )
                traces_by_method[method].append(
                    _trace_record(
                        learned,
                        method=method,
                        family=family,
                        heldout_family=heldout_family,
                        source_slice=source_slice,
                    )
                )
                learned_decisions[method] += sum(
                    event["purpose"] == "residual-ranker-learned"
                    for event in learned.query_trace
                )
                fallback_decisions[method] += sum(
                    event["purpose"] == "residual-ranker-fallback"
                    for event in learned.query_trace
                )
            deadline_check()
    if score_digest is None:
        raise ValueError("D1 paired evaluation produced no score rows")

    victim_ids = tuple(victim_id for victim_id, _ in selected_victims)
    summaries: dict[str, dict[str, object]] = {
        _SCORE_GREEDY: {
            **_summarize_method(
                rows_by_method[_SCORE_GREEDY],
                traces_by_method[_SCORE_GREEDY],
                attack=attack,
                policy=None,
                digest_before=score_digest,
                victim_ids=victim_ids,
            ),
            "source_model_calls": sum(
                row.total_target_calls for row in rows_by_method[_SCORE_GREEDY]
            ),
            "hidden_target_calls": 0,
        }
    }
    for method, policy in selected_policies.items():
        decisions = learned_decisions[method] + fallback_decisions[method]
        summaries[method] = {
            **_summarize_method(
                rows_by_method[method],
                traces_by_method[method],
                attack=attack,
                policy=policy,
                digest_before=policy_digests[method],
                victim_ids=victim_ids,
            ),
            "learned_override_decisions": learned_decisions[method],
            "score_fallback_decisions": fallback_decisions[method],
            "deployment_override_fraction": (
                learned_decisions[method] / decisions if decisions else 0.0
            ),
            "source_model_calls": sum(
                row.total_target_calls for row in rows_by_method[method]
            ),
            "hidden_target_calls": 0,
        }
    eligible_digests = {
        str(summary["eligible_sample_ids_sha256"]) for summary in summaries.values()
    }
    if len(eligible_digests) != 1:
        raise ValueError("D1 paired methods have different eligible cohorts")

    rows = [row for method in methods for row in rows_by_method[method]]
    traces = [trace for method in methods for trace in traces_by_method[method]]
    expected_ids = {
        f"cifar10:{family}:{victim_id}:{index}"
        for victim_id, _ in selected_victims
        for index in selected_indices
    }
    raw_audit = audit_evaluation(
        rows,
        summaries,
        attack,
        expected_sample_ids=expected_ids,
        expected_victim_ids=set(victim_ids),
    )
    trace_audit = _trace_and_cohort_audit(
        rows,
        traces,
        methods=methods,
        expected_sample_ids=expected_ids,
    )
    audit = {**raw_audit, **trace_audit}
    if (
        audit.get("passed") is not True
        or audit.get("expected_cohort_verified") is not True
        or audit.get("trace_result_identity_multiset_matched") is not True
    ):
        raise ValueError("D1 raw paired evaluation audit failed")
    return (
        {
            "seed": seed,
            "source_slice": source_slice,
            "victim_family": family,
            "victim_ids": list(victim_ids),
            "methods": summaries,
            "audit": audit,
            "source_model_calls": sum(row.total_target_calls for row in rows),
            "elapsed_seconds": time.monotonic() - started,
            "target_calls": 0,
            "hidden_target_calls": 0,
        },
        rows,
        traces,
    )


def evaluate_residual_d1(
    request: ResidualD1Request,
    context: D1SourceContext,
    policy: ResidualRankerPolicy,
    *,
    deadline_check: Callable[[], None],
    progress: Callable[[str], None],
) -> tuple[dict[str, object], list[ResearchResultRow], list[dict[str, object]]]:
    """Run both paired methods on every held-out source condition."""

    conditions: dict[str, object] = {}
    rows: list[ResearchResultRow] = []
    traces: list[dict[str, object]] = []
    for offset, family in enumerate(context.source_families):
        condition, family_rows, family_traces = evaluate_residual_policy_cohort(
            policies={"residual_ranker_bc": policy},
            victims=context.evaluation_victims[family],
            samples=context.evaluation_samples,
            indices=context.evaluation_indices,
            attack=context.config.attack_config(),
            family=family,
            seed=request.seed + 900_000 + offset,
            heldout_family=request.heldout_family,
            source_slice="seen_family_new_instance",
            deadline_check=deadline_check,
            progress=progress,
        )
        conditions[family] = condition
        rows.extend(family_rows)
        traces.extend(family_traces)
    return conditions, rows, traces


def _asr(metrics: Mapping[str, object]) -> float:
    eligible = int(metrics.get("eligible", 0))
    if eligible <= 0:
        raise ValueError("D1 evaluation condition has no eligible images")
    return int(metrics.get("successes", 0)) / eligible


def _finite_metric(metrics: Mapping[str, object], key: str, *, label: str) -> float:
    value = metrics.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} is missing or non-finite")
    return float(value)


def _decision(
    competence: Mapping[str, object],
    threshold: Mapping[str, object],
    conditions: Mapping[str, object],
) -> dict[str, object]:
    if competence.get("target_mode") != "all_soft":
        raise ValueError("D1 competence metrics are incomplete")
    by_family = _mapping(
        competence.get("by_source_family"),
        label="D1 competence by source family",
    )
    if set(by_family) != set(D1_SOURCE_FAMILIES):
        raise ValueError("D1 competence source-family keys are incomplete")
    for family in D1_SOURCE_FAMILIES:
        family_metrics = _mapping(
            by_family[family],
            label=f"D1 {family} competence",
        )
        _finite_metric(
            family_metrics,
            "accuracy_gain_vs_prior",
            label=f"D1 {family} accuracy gain",
        )
        _finite_metric(
            family_metrics,
            "soft_ce_improvement_vs_prior",
            label=f"D1 {family} soft-CE gain",
        )
    macro = _mapping(
        competence.get("equal_family_macro"),
        label="D1 equal-family competence",
    )
    worst = _mapping(
        competence.get("worst_family"),
        label="D1 worst-family competence",
    )
    macro_gated_accuracy = _finite_metric(
        macro,
        "gated_top1_accuracy",
        label="D1 macro gated accuracy",
    )
    macro_prior_accuracy = _finite_metric(
        macro,
        "prior_top1_accuracy",
        label="D1 macro prior accuracy",
    )
    competence_accuracy_gain = _finite_metric(
        macro,
        "accuracy_gain_vs_prior",
        label="D1 macro accuracy gain",
    )
    competence_ce_gain = _finite_metric(
        macro,
        "soft_ce_improvement_vs_prior",
        label="D1 macro soft-CE gain",
    )
    competence_residual_use = _finite_metric(
        macro,
        "residual_use_fraction",
        label="D1 macro residual-use fraction",
    )
    worst_accuracy_gain = _finite_metric(
        worst,
        "accuracy_gain_vs_prior",
        label="D1 worst-family accuracy gain",
    )
    worst_ce_gain = _finite_metric(
        worst,
        "soft_ce_improvement_vs_prior",
        label="D1 worst-family soft-CE gain",
    )
    worst_family_passed = worst_accuracy_gain > 0 and worst_ce_gain > 0
    threshold_separate = threshold.get("selection_role") == "bc_validation_only"
    score_asrs: list[float] = []
    learned_asrs: list[float] = []
    score_aucs: list[float] = []
    learned_aucs: list[float] = []
    observed_gaps: dict[str, dict[str, float | bool]] = {}
    deployment_decisions = 0
    deployment_overrides = 0
    for family in D1_SOURCE_FAMILIES:
        condition = _mapping(conditions[family], label=f"D1 {family} condition")
        if _mapping(condition["audit"], label="D1 audit").get("passed") is not True:
            raise ValueError("D1 decision received a failed evaluation audit")
        methods = _mapping(condition["methods"], label="D1 methods")
        score = _mapping(methods["score_greedy"], label="D1 score metrics")
        learned = _mapping(
            methods["residual_ranker_bc"],
            label="D1 learned metrics",
        )
        score_asr = _asr(score)
        learned_asr = _asr(learned)
        score_auc = float(score["asr_query_auc"])
        learned_auc = float(learned["asr_query_auc"])
        score_asrs.append(score_asr)
        learned_asrs.append(learned_asr)
        score_aucs.append(score_auc)
        learned_aucs.append(learned_auc)
        deployment_overrides += int(learned["learned_override_decisions"])
        deployment_decisions += int(learned["learned_override_decisions"]) + int(
            learned["score_fallback_decisions"]
        )
        observed_gaps[family] = {
            "asr_point_estimate_gap": learned_asr - score_asr,
            "auc_point_estimate_gap": learned_auc - score_auc,
            "asr_observed_non_decrease": learned_asr >= score_asr,
            "auc_observed_non_decrease": learned_auc >= score_auc,
        }
    deployment_override_fraction = (
        deployment_overrides / deployment_decisions if deployment_decisions else 0.0
    )
    base = residual_d1_promotion_decision(
        bc_validation_score=macro_gated_accuracy,
        prior_validation_score=macro_prior_accuracy,
        score_greedy_asr=sum(score_asrs) / len(score_asrs),
        score_greedy_auc=sum(score_aucs) / len(score_aucs),
        learned_asr=sum(learned_asrs) / len(learned_asrs),
        learned_auc=sum(learned_aucs) / len(learned_aucs),
    )
    all_observed_non_decrease = all(
        bool(gaps["asr_observed_non_decrease"])
        and bool(gaps["auc_observed_non_decrease"])
        for gaps in observed_gaps.values()
    )
    deployment_material = (
        deployment_override_fraction >= D1_MIN_DEPLOYMENT_OVERRIDE_FRACTION
    )
    passed = (
        competence_accuracy_gain > 0
        and competence_ce_gain > 0
        and competence_residual_use > 0
        and worst_family_passed
        and threshold_separate
        and all_observed_non_decrease
        and deployment_material
    )
    return {
        **base,
        "passed": passed,
        "competence_accuracy_gain_vs_prior": competence_accuracy_gain,
        "competence_soft_ce_improvement_vs_prior": competence_ce_gain,
        "competence_residual_use_fraction": competence_residual_use,
        "worst_family_accuracy_gain_vs_prior": worst_accuracy_gain,
        "worst_family_soft_ce_improvement_vs_prior": worst_ce_gain,
        "worst_family_competence_gate_passed": worst_family_passed,
        "threshold_selected_on_separate_role": threshold_separate,
        "observed_point_estimate_gaps": observed_gaps,
        "deployment_override_fraction": deployment_override_fraction,
        "minimum_deployment_override_fraction": D1_MIN_DEPLOYMENT_OVERRIDE_FRACTION,
        "deployment_override_gate_passed": deployment_material,
        "eligible_for_d1b_source_only_ppo": passed,
        "authorizes_hidden_target_evaluation": False,
        "limitations": (
            "one fold, one seed, 50 source images, held-out source instances; "
            "paired image-bootstrap CIs are descriptive for fixed seed/victims only"
        ),
    }
