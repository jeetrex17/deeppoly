"""Frozen-policy CIFAR evaluation with shared attack accounting."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from .baselines import (
    BanditActionPolicy,
    FixedActionPolicy,
    RandomActionPolicy,
)
from .config import AttackConfig
from .operator import AttackOperatorContract
from .recurrent import RecurrentAttackPolicy
from .research_metrics import AttackOutcome, asr_at_budgets, asr_query_auc
from .research_protocol import run_frozen_episode, run_score_greedy_episode
from .results import ResearchResultRow


def _named_targets(
    target: tuple[str, nn.Module] | Sequence[tuple[str, nn.Module]],
) -> tuple[tuple[str, nn.Module], ...]:
    if (
        isinstance(target, tuple)
        and len(target) == 2
        and isinstance(target[0], str)
        and isinstance(target[1], nn.Module)
    ):
        targets = (target,)
    else:
        targets = tuple(target)
    if not targets or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not isinstance(item[1], nn.Module)
        for item in targets
    ):
        raise ValueError("evaluation requires one or more named victim modules")
    return targets


def _method_summary(
    rows: list[ResearchResultRow],
    outcomes: list[AttackOutcome],
    initialization_flags: list[bool],
    attack: AttackConfig,
    attack_policy: object | None,
    policy_digest_before: str | None,
    deterministic: bool,
    victim_count: int,
) -> dict[str, object]:
    budgets = tuple(range(attack.max_queries + 1))
    eligible = sum(outcome.clean_correct for outcome in outcomes)
    curve = asr_at_budgets(outcomes, budgets) if eligible else {}
    auc = asr_query_auc(curve) if eligible else None
    action_counts = Counter(action for row in rows for action in row.action_trace)
    action_total = sum(action_counts.values())
    action_entropy = 0.0
    if action_total and attack.action_dim > 1:
        action_entropy = -sum(
            (count / action_total) * math.log(count / action_total)
            for count in action_counts.values()
        ) / math.log(attack.action_dim)
    policy_digest_after = (
        attack_policy.persistent_digest()
        if attack_policy is not None
        else rows[-1].policy_digest
    )
    operator = (
        AttackOperatorContract.from_config(attack)
        if attack_policy is not None or attack.rollback_on_non_improvement
        else AttackOperatorContract(
            epsilon=attack.epsilon,
            step_size=attack.epsilon,
            grid_size=attack.grid_size,
            rollback_on_non_improvement=True,
        )
    )
    eligible_ids = sorted(row.sample_id for row in rows if row.clean_correct)
    eligible_digest = hashlib.sha256(
        "\n".join(eligible_ids).encode("utf-8")
    ).hexdigest()
    return {
        "eligible": eligible,
        "successes": sum(row.success for row in rows),
        "asr_at_budgets": curve,
        "asr_query_auc": auc,
        "query_budget": attack.max_queries,
        "max_total_target_calls": max(
            (row.total_target_calls for row in rows),
            default=0,
        ),
        "initialization_included": all(initialization_flags),
        "eligible_sample_ids_sha256": eligible_digest,
        "policy_digest_before": policy_digest_before,
        "policy_digest_after": policy_digest_after,
        "frozen": policy_digest_before == policy_digest_after,
        "victim_count": victim_count,
        "operator": operator.as_dict(),
        "operator_digest": operator.digest(),
        "deterministic_actions": deterministic,
        "sampling_temperature": (
            float(getattr(attack_policy, "temperature", 1.0))
            if isinstance(attack_policy, RecurrentAttackPolicy)
            else None
        ),
        "action_histogram": {
            str(action): count for action, count in sorted(action_counts.items())
        },
        "normalized_action_entropy": action_entropy,
    }


def _victim_performance(
    rows: Sequence[ResearchResultRow],
    attack: AttackConfig,
) -> dict[str, object]:
    outcomes = tuple(
        AttackOutcome(row.clean_correct, row.query_to_success) for row in rows
    )
    eligible = sum(outcome.clean_correct for outcome in outcomes)
    budgets = tuple(range(attack.max_queries + 1))
    curve = asr_at_budgets(outcomes, budgets) if eligible else {}
    eligible_ids = sorted(row.sample_id for row in rows if row.clean_correct)
    return {
        "eligible": eligible,
        "successes": sum(row.success for row in rows),
        "asr_at_budgets": curve,
        "asr_query_auc": (asr_query_auc(curve) if eligible else None),
        "eligible_sample_ids_sha256": hashlib.sha256(
            "\n".join(eligible_ids).encode("utf-8")
        ).hexdigest(),
    }


def evaluate_methods(
    policy: RecurrentAttackPolicy,
    target: tuple[str, nn.Module] | Sequence[tuple[str, nn.Module]],
    samples: tuple[tuple[torch.Tensor, int], ...],
    indices: Sequence[int],
    attack: AttackConfig,
    seed: int,
    target_family: str,
    progress: Callable[[str], None],
    trace_samples_per_method: int = -1,
    additional_policies: Mapping[
        str,
        tuple[RecurrentAttackPolicy, bool],
    ]
    | None = None,
    main_method_prefix: str = "groupdro_recurrent_ppo",
) -> tuple[
    list[ResearchResultRow],
    list[dict[str, object]],
    dict[str, object],
]:
    methods: dict[str, tuple[object | None, bool]]
    if additional_policies:
        methods = {
            f"{main_method_prefix}_stochastic": (policy, False),
            "random_action": (
                RandomActionPolicy(attack.action_dim, seed),
                False,
            ),
            "bandit_action": (
                BanditActionPolicy(attack.action_dim, seed),
                True,
            ),
            "score_greedy": (None, True),
        }
    else:
        methods = {
            main_method_prefix: (policy, True),
            f"{main_method_prefix}_stochastic": (policy, False),
            "fixed_action": (
                FixedActionPolicy(0, attack.action_dim),
                True,
            ),
            "random_action": (
                RandomActionPolicy(attack.action_dim, seed),
                False,
            ),
            "bandit_action": (
                BanditActionPolicy(attack.action_dim, seed),
                True,
            ),
            "score_greedy": (None, True),
        }
    if additional_policies:
        overlap = set(methods) & set(additional_policies)
        if overlap:
            raise ValueError(f"additional policy names overlap: {sorted(overlap)}")
        methods = {**methods, **additional_policies}
    return evaluate_method_set(
        methods,
        target,
        samples,
        indices,
        attack,
        seed,
        target_family,
        progress,
        trace_samples_per_method=trace_samples_per_method,
    )


def evaluate_method_set(
    methods: Mapping[str, tuple[object | None, bool]],
    target: tuple[str, nn.Module] | Sequence[tuple[str, nn.Module]],
    samples: tuple[tuple[torch.Tensor, int], ...],
    indices: Sequence[int],
    attack: AttackConfig,
    seed: int,
    target_family: str,
    progress: Callable[[str], None],
    trace_samples_per_method: int = -1,
    stochastic_seed_namespace: str | None = None,
) -> tuple[
    list[ResearchResultRow],
    list[dict[str, object]],
    dict[str, object],
]:
    """Evaluate an explicit frozen method set on a shared cohort.

    ``stochastic_seed_namespace`` makes stochastic policies consume the same
    per-episode uniform draws. This supports paired policy diagnostics such as
    temperature sweeps without changing the legacy evaluator's seed contract.
    """

    targets = _named_targets(target)
    selected_methods = dict(methods)
    if not selected_methods:
        raise ValueError("evaluation requires one or more methods")
    if any(not isinstance(name, str) or not name.strip() for name in selected_methods):
        raise ValueError("evaluation method names must be non-empty strings")
    if any(
        not isinstance(specification, tuple)
        or len(specification) != 2
        or not isinstance(specification[1], bool)
        for specification in selected_methods.values()
    ):
        raise ValueError("each method must map to a (policy, deterministic) tuple")
    if len(samples) != len(indices):
        raise ValueError("samples and indices must have identical lengths")
    if trace_samples_per_method < -1:
        raise ValueError("trace sample limit must be -1 or non-negative")
    if stochastic_seed_namespace is not None and not stochastic_seed_namespace.strip():
        raise ValueError("stochastic seed namespace must be a non-empty string")
    all_rows: list[ResearchResultRow] = []
    traces: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for method_offset, (
        method,
        (attack_policy, deterministic),
    ) in enumerate(selected_methods.items()):
        progress(
            f"evaluating {method} on {len(samples)} images across "
            f"{len(targets)} victim instance(s)"
        )
        torch.manual_seed(seed + 100_000 + method_offset)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed + 100_000 + method_offset)
        digest_before = (
            attack_policy.persistent_digest() if attack_policy is not None else None
        )
        outcomes: list[AttackOutcome] = []
        method_rows: list[ResearchResultRow] = []
        initialization_flags: list[bool] = []
        rows_by_victim: dict[str, list[ResearchResultRow]] = {
            victim_id: [] for victim_id, _ in targets
        }
        captured_traces = 0
        for victim_id, victim in targets:
            for (image, label), dataset_index in zip(samples, indices):
                sample_id = f"cifar10:{target_family}:{victim_id}:{dataset_index}"
                if attack_policy is None:
                    result = run_score_greedy_episode(
                        victim,
                        image,
                        label,
                        sample_id,
                        victim_id,
                        target_family,
                        attack,
                        seed + method_offset,
                    )
                    digest_before = digest_before or result.policy_digest_before
                else:
                    seed_method = (
                        stochastic_seed_namespace
                        if stochastic_seed_namespace is not None and not deterministic
                        else method
                    )
                    result = run_frozen_episode(
                        attack_policy,
                        victim,
                        image,
                        label,
                        sample_id,
                        victim_id,
                        target_family,
                        attack,
                        deterministic=deterministic,
                        episode_seed=int.from_bytes(
                            hashlib.sha256(
                                (
                                    f"frozen-eval-v1:{seed}:{seed_method}:"
                                    f"{victim_id}:{sample_id}"
                                ).encode()
                            ).digest()[:8],
                            "big",
                        ),
                    )
                outcomes.append(
                    AttackOutcome(
                        result.clean_correct,
                        result.query_to_success,
                    )
                )
                row = ResearchResultRow(
                    sample_id=sample_id,
                    victim_id=victim_id,
                    victim_family=target_family,
                    method=method,
                    threat_model="T1",
                    seed=seed,
                    query_budget=attack.max_queries,
                    clean_correct=result.clean_correct,
                    success=result.success,
                    query_to_success=result.query_to_success,
                    total_target_calls=result.total_target_calls,
                    linf=result.linf,
                    l2=result.l2,
                    policy_digest=result.policy_digest_after,
                    action_trace=result.actions,
                )
                all_rows.append(row)
                method_rows.append(row)
                rows_by_victim[victim_id].append(row)
                initialization_flags.append(
                    bool(
                        result.query_trace
                        and result.query_trace[0]["purpose"] == "initialization"
                    )
                )
                if (
                    trace_samples_per_method == -1
                    or captured_traces < trace_samples_per_method
                ):
                    traces.append({"method": method, **result.as_dict()})
                    captured_traces += 1
        method_summary = _method_summary(
            method_rows,
            outcomes,
            initialization_flags,
            attack,
            attack_policy,
            digest_before,
            deterministic,
            len(targets),
        )
        method_summary["by_victim"] = {
            victim_id: _victim_performance(victim_rows, attack)
            for victim_id, victim_rows in rows_by_victim.items()
        }
        summary[method] = method_summary
    return all_rows, traces, summary
