"""Fail-closed audits over raw frozen-transfer result rows."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Mapping, Sequence

from .config import AttackConfig
from .research_metrics import (
    AttackOutcome,
    asr_at_budgets,
    asr_query_auc,
)
from .results import ResearchResultRow


def audit_evaluation(
    rows: Sequence[ResearchResultRow],
    evaluation: Mapping[str, object],
    attack: AttackConfig,
    *,
    expected_sample_ids: Sequence[str] | set[str] | None = None,
    expected_victim_ids: Sequence[str] | set[str] | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    expected_cohort = (
        set(expected_sample_ids)
        if expected_sample_ids is not None
        else None
    )
    expected_victims = (
        set(expected_victim_ids)
        if expected_victim_ids is not None
        else None
    )
    if expected_cohort is not None and (
        not expected_cohort
        or any(
            not isinstance(sample_id, str) or not sample_id
            for sample_id in expected_cohort
        )
    ):
        raise ValueError("expected sample IDs must be non-empty strings")
    if expected_victims is not None and (
        not expected_victims
        or any(
            not isinstance(victim_id, str) or not victim_id
            for victim_id in expected_victims
        )
    ):
        raise ValueError("expected victim IDs must be non-empty strings")
    if (expected_cohort is None) != (expected_victims is None):
        raise ValueError(
            "expected sample and victim IDs must be provided together"
        )
    if not rows:
        errors.append("no raw evaluation rows")
    methods = tuple(sorted(str(method) for method in evaluation))
    row_methods = {row.method for row in rows}
    if row_methods != set(methods):
        errors.append("raw rows and method summaries do not align")
    cohorts: dict[str, set[str]] = {method: set() for method in methods}
    for row in rows:
        if (
            not math.isfinite(row.linf)
            or not math.isfinite(row.l2)
            or row.linf < 0
            or row.l2 < 0
            or row.linf > attack.epsilon + 1e-6
        ):
            errors.append(
                f"{row.method}/{row.sample_id}: invalid perturbation norm"
            )
        if (
            row.query_budget != attack.max_queries
            or not 1 <= row.total_target_calls <= attack.max_queries
        ):
            errors.append(
                f"{row.method}/{row.sample_id}: invalid query accounting"
            )
        if row.success and not row.clean_correct:
            errors.append(
                f"{row.method}/{row.sample_id}: success on an ineligible image"
            )
        if (
            len(row.action_trace) != row.total_target_calls - 1
            or any(
                not isinstance(action, int)
                or isinstance(action, bool)
                or not 0 <= action < attack.action_dim
                for action in row.action_trace
            )
        ):
            errors.append(
                f"{row.method}/{row.sample_id}: invalid action trace"
            )
        cohorts.setdefault(row.method, set()).add(row.sample_id)
    if cohorts and len({frozenset(value) for value in cohorts.values()}) != 1:
        errors.append("methods do not share the same raw image-victim cohort")
    if expected_cohort is not None:
        for method, cohort in cohorts.items():
            method_rows = tuple(
                row for row in rows if row.method == method
            )
            if (
                cohort != expected_cohort
                or len(method_rows) != len(expected_cohort)
            ):
                errors.append(
                    f"{method}: raw rows do not match the expected cohort"
                )
    operator_digests: set[str] = set()
    for method, raw_metrics in evaluation.items():
        if not isinstance(raw_metrics, Mapping):
            errors.append(f"{method}: metrics are not a mapping")
            continue
        if raw_metrics.get("frozen") is not True:
            errors.append(f"{method}: persistent policy state changed")
        before = raw_metrics.get("policy_digest_before")
        after = raw_metrics.get("policy_digest_after")
        if not isinstance(before, str) or not before or before != after:
            errors.append(f"{method}: invalid frozen-policy digest")
        operator_digest = raw_metrics.get("operator_digest")
        if not isinstance(operator_digest, str) or not operator_digest:
            errors.append(f"{method}: missing operator digest")
        else:
            operator_digests.add(operator_digest)
        if raw_metrics.get("initialization_included") is not True:
            errors.append(f"{method}: initialization was not counted")
        if raw_metrics.get("max_total_target_calls", 0) > attack.max_queries:
            errors.append(f"{method}: summary exceeds the query budget")
        method_rows = tuple(row for row in rows if row.method == method)
        method_victims = {row.victim_id for row in method_rows}
        if (
            expected_victims is not None
            and method_victims != expected_victims
        ):
            errors.append(
                f"{method}: raw rows do not match expected victims"
            )
        outcomes = tuple(
            AttackOutcome(row.clean_correct, row.query_to_success)
            for row in method_rows
        )
        eligible = sum(outcome.clean_correct for outcome in outcomes)
        curve = (
            asr_at_budgets(
                outcomes,
                tuple(range(attack.max_queries + 1)),
            )
            if eligible
            else {}
        )
        recorded_curve = raw_metrics.get("asr_at_budgets")
        normalized_curve = (
            {
                int(budget): float(value)
                for budget, value in recorded_curve.items()
            }
            if isinstance(recorded_curve, Mapping)
            else None
        )
        eligible_ids = sorted(
            row.sample_id
            for row in method_rows
            if row.clean_correct
        )
        eligible_digest = hashlib.sha256(
            "\n".join(eligible_ids).encode("utf-8")
        ).hexdigest()
        if (
            raw_metrics.get("eligible") != eligible
            or raw_metrics.get("successes")
            != sum(row.success for row in method_rows)
            or normalized_curve != curve
            or (
                eligible
                and not math.isclose(
                    float(raw_metrics.get("asr_query_auc", -1)),
                    asr_query_auc(curve),
                    abs_tol=1e-12,
                )
            )
            or raw_metrics.get("eligible_sample_ids_sha256")
            != eligible_digest
            or raw_metrics.get("max_total_target_calls")
            != max(
                (
                    row.total_target_calls
                    for row in method_rows
                ),
                default=0,
            )
        ):
            errors.append(f"{method}: summary does not match raw rows")
        action_counts = Counter(
            action
            for row in method_rows
            for action in row.action_trace
        )
        action_total = sum(action_counts.values())
        entropy = 0.0
        if action_total and attack.action_dim > 1:
            entropy = -sum(
                (count / action_total)
                * math.log(count / action_total)
                for count in action_counts.values()
            ) / math.log(attack.action_dim)
        if not math.isclose(
            float(
                raw_metrics.get(
                    "normalized_action_entropy",
                    -1,
                )
            ),
            entropy,
            abs_tol=1e-12,
        ):
            errors.append(f"{method}: action entropy mismatch")
        by_victim = raw_metrics.get("by_victim")
        victim_ids = {row.victim_id for row in method_rows}
        if (
            not isinstance(by_victim, Mapping)
            or set(by_victim) != victim_ids
        ):
            errors.append(f"{method}: per-victim summaries are missing")
        elif (
            expected_victims is not None
            and (
                victim_ids != expected_victims
                or raw_metrics.get("victim_count")
                != len(expected_victims)
            )
        ):
            errors.append(
                f"{method}: per-victim summaries do not match expected victims"
            )
        else:
            for victim_id in sorted(victim_ids):
                victim_rows = tuple(
                    row
                    for row in method_rows
                    if row.victim_id == victim_id
                )
                victim_metrics = by_victim[victim_id]
                if not isinstance(victim_metrics, Mapping):
                    errors.append(
                        f"{method}/{victim_id}: invalid victim summary"
                    )
                    continue
                victim_outcomes = tuple(
                    AttackOutcome(
                        row.clean_correct,
                        row.query_to_success,
                    )
                    for row in victim_rows
                )
                victim_eligible = sum(
                    outcome.clean_correct
                    for outcome in victim_outcomes
                )
                victim_curve = (
                    asr_at_budgets(
                        victim_outcomes,
                        tuple(range(attack.max_queries + 1)),
                    )
                    if victim_eligible
                    else {}
                )
                recorded_victim_curve = victim_metrics.get(
                    "asr_at_budgets"
                )
                normalized_victim_curve = (
                    {
                        int(budget): float(value)
                        for budget, value in recorded_victim_curve.items()
                    }
                    if isinstance(
                        recorded_victim_curve,
                        Mapping,
                    )
                    else None
                )
                victim_ids_digest = hashlib.sha256(
                    "\n".join(
                        sorted(
                            row.sample_id
                            for row in victim_rows
                            if row.clean_correct
                        )
                    ).encode("utf-8")
                ).hexdigest()
                if (
                    victim_metrics.get("eligible")
                    != victim_eligible
                    or victim_metrics.get("successes")
                    != sum(row.success for row in victim_rows)
                    or normalized_victim_curve != victim_curve
                    or victim_metrics.get(
                        "eligible_sample_ids_sha256"
                    )
                    != victim_ids_digest
                    or (
                        victim_eligible
                        and not math.isclose(
                            float(
                                victim_metrics.get(
                                    "asr_query_auc",
                                    -1,
                                )
                            ),
                            asr_query_auc(victim_curve),
                            abs_tol=1e-12,
                        )
                    )
                ):
                    errors.append(
                        f"{method}/{victim_id}: summary mismatch"
                    )
    if (
        attack.rollback_on_non_improvement
        and len(operator_digests) != 1
    ):
        errors.append("publication methods do not share one operator digest")
    expected_cohort_verified = bool(
        expected_cohort is not None
        and expected_victims is not None
        and cohorts
        and all(
            cohort == expected_cohort
            and len(
                tuple(row for row in rows if row.method == method)
            )
            == len(expected_cohort)
            and {
                row.victim_id
                for row in rows
                if row.method == method
            }
            == expected_victims
            for method, cohort in cohorts.items()
        )
    )
    return {
        "passed": not errors,
        "errors": errors,
        "row_count": len(rows),
        "method_count": len(methods),
        "epsilon": attack.epsilon,
        "query_budget_including_initialization": attack.max_queries,
        "raw_cohort_aligned": bool(
            cohorts
            and len({frozenset(value) for value in cohorts.values()}) == 1
        ),
        "operator_aligned": len(operator_digests) == 1,
        "expected_cohort_verified": expected_cohort_verified,
        "expected_sample_count": (
            len(expected_cohort)
            if expected_cohort is not None
            else None
        ),
        "expected_victim_count": (
            len(expected_victims)
            if expected_victims is not None
            else None
        ),
    }
