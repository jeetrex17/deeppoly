"""Validated tabular derivations from Phase 2 source-screen evidence."""

from __future__ import annotations

from typing import Mapping, Sequence

from .phase1_export_validation import (
    digest,
    finite_number,
    nonnegative_integer,
    require_mapping,
    require_sequence,
)
from .phase2_promotion import SCREEN_CONTROL


LEARNED_METHOD = (
    "soft_gradient_bc_action_conditioned_groupdro_ppo_stochastic"
)
METHOD_LABELS = {
    LEARNED_METHOD: "Learned stochastic",
    "soft_gradient_bc_action_conditioned_groupdro_ppo": (
        "Learned deterministic"
    ),
    "score_greedy": "Score greedy",
    "bandit_action": "Bandit",
    "random_action": "Random",
    "fixed_action": "Fixed",
}


def _final_asr(metrics: Mapping[str, object]) -> float:
    curve = require_mapping(
        metrics.get("asr_at_budgets"),
        "ASR curve",
    )
    if not curve:
        raise ValueError("ASR curve cannot be empty")
    normalized = {
        int(budget): finite_number(value, "ASR curve value")
        for budget, value in curve.items()
    }
    result = normalized[max(normalized)]
    if not 0 <= result <= 1:
        raise ValueError("final ASR must be a probability")
    return result


def condition_rows(
    verified_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in verified_runs:
        run = require_mapping(item.get("run"), "verified run")
        evaluation = require_mapping(
            run.get("source_evaluation"),
            "source evaluation",
        )
        for slice_name, families_value in evaluation.items():
            families = require_mapping(
                families_value,
                f"{slice_name} families",
            )
            for source_family, methods_value in families.items():
                methods = require_mapping(
                    methods_value,
                    f"{slice_name}/{source_family} methods",
                )
                for method, metrics_value in methods.items():
                    metrics = require_mapping(
                        metrics_value,
                        f"{slice_name}/{source_family}/{method}",
                    )
                    eligible = nonnegative_integer(
                        metrics.get("eligible"),
                        "eligible images",
                    )
                    successes = nonnegative_integer(
                        metrics.get("successes"),
                        "successful attacks",
                    )
                    if eligible <= 0 or successes > eligible:
                        raise ValueError(
                            "source success counts are invalid"
                        )
                    rows.append(
                        {
                            "fingerprint": str(run["fingerprint"]),
                            "seed": int(run["seed"]),
                            "omitted_target_family": str(
                                run["target_family"]
                            ),
                            "source_slice": str(slice_name),
                            "evaluated_source_family": str(
                                source_family
                            ),
                            "method": str(method),
                            "eligible": eligible,
                            "successes": successes,
                            "asr": _final_asr(metrics),
                            "asr_query_auc": finite_number(
                                metrics.get("asr_query_auc"),
                                "ASR-query AUC",
                            ),
                            "normalized_action_entropy": (
                                finite_number(
                                    metrics.get(
                                        "normalized_action_entropy",
                                        0.0,
                                    ),
                                    "action entropy",
                                )
                            ),
                            "query_budget": nonnegative_integer(
                                metrics.get("query_budget"),
                                "query budget",
                            ),
                        }
                    )
    rows.sort(
        key=lambda row: (
            str(row["omitted_target_family"]),
            int(row["seed"]),
            str(row["source_slice"]),
            str(row["evaluated_source_family"]),
            str(row["method"]),
        )
    )
    return rows


def mean(
    rows: Sequence[Mapping[str, object]],
    key: str,
) -> float:
    if not rows:
        raise ValueError("cannot average an empty sequence")
    return sum(float(row[key]) for row in rows) / len(rows)


def method_rows(
    conditions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    methods = sorted(
        {str(row["method"]) for row in conditions},
        key=lambda method: (
            0 if method == LEARNED_METHOD else 1,
            METHOD_LABELS.get(method, method),
        ),
    )
    rows = []
    for method in methods:
        selected = [
            row for row in conditions if row["method"] == method
        ]
        eligible = sum(int(row["eligible"]) for row in selected)
        successes = sum(int(row["successes"]) for row in selected)
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "conditions": len(selected),
                "eligible": eligible,
                "successes": successes,
                "asr": mean(selected, "asr"),
                "pooled_asr": successes / eligible,
                "asr_query_auc": mean(
                    selected,
                    "asr_query_auc",
                ),
                "normalized_action_entropy": mean(
                    selected,
                    "normalized_action_entropy",
                ),
            }
        )
    return rows


def _bc_row(run: Mapping[str, object]) -> dict[str, object]:
    policy = require_mapping(run.get("policy"), "policy")
    training = require_mapping(policy.get("training"), "training")
    cloning = require_mapping(
        training.get("behavior_cloning"),
        "behavior cloning",
    )
    validation = require_mapping(
        cloning.get("validation"),
        "behavior cloning validation",
    )
    gate = require_mapping(
        cloning.get("gate"),
        "behavior cloning gate",
    )
    target_mode = validation.get("target_mode")
    if target_mode not in {"soft", "mixed_soft_and_hard"}:
        raise ValueError("Phase 2 export requires soft BC diagnostics")
    top5 = finite_number(
        validation.get("top5_accuracy"),
        "BC top-5 accuracy",
    )
    oracle_top5 = finite_number(
        validation.get("validation_oracle_top5_accuracy"),
        "BC oracle top-5 accuracy",
    )
    soft_ce = finite_number(
        validation.get("soft_cross_entropy"),
        "BC soft cross-entropy",
    )
    oracle_soft_ce = finite_number(
        validation.get("validation_oracle_soft_cross_entropy"),
        "BC oracle soft cross-entropy",
    )
    return {
        "fingerprint": str(run["fingerprint"]),
        "seed": int(run["seed"]),
        "omitted_target_family": str(run["target_family"]),
        "accepted_steps": nonnegative_integer(
            validation.get("accepted_steps"),
            "BC accepted validation steps",
        ),
        "top1_accuracy": finite_number(
            validation.get("top1_accuracy"),
            "BC top-1 accuracy",
        ),
        "top5_accuracy": top5,
        "validation_oracle_top5_accuracy": oracle_top5,
        "top5_gain": top5 - oracle_top5,
        "soft_cross_entropy": soft_ce,
        "validation_oracle_soft_cross_entropy": oracle_soft_ce,
        "soft_ce_improvement": oracle_soft_ce - soft_ce,
        "elapsed_seconds": finite_number(
            cloning.get("elapsed_seconds"),
            "BC elapsed seconds",
        ),
        "gate_passed": gate.get("passed") is True,
    }


def _recorded_runtime(
    run: Mapping[str, object],
) -> tuple[float, float, float, float]:
    policy = require_mapping(run.get("policy"), "policy")
    training = require_mapping(policy.get("training"), "training")
    cloning = require_mapping(
        training.get("behavior_cloning"),
        "behavior cloning",
    )
    blocks = require_sequence(training.get("blocks"), "PPO blocks")
    bc_seconds = finite_number(
        cloning.get("elapsed_seconds"),
        "BC elapsed seconds",
    )
    ppo_seconds = sum(
        finite_number(
            require_mapping(block, "PPO block").get(
                "elapsed_seconds"
            ),
            "PPO block elapsed seconds",
        )
        for block in blocks
    )
    evaluation_seconds = finite_number(
        run.get("source_evaluation_elapsed_seconds"),
        "source evaluation elapsed seconds",
    )
    return (
        bc_seconds,
        ppo_seconds,
        evaluation_seconds,
        bc_seconds + ppo_seconds + evaluation_seconds,
    )


def fold_rows(
    verified_runs: Sequence[Mapping[str, object]],
    conditions: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    folds: list[dict[str, object]] = []
    bc_rows: list[dict[str, object]] = []
    for item in verified_runs:
        run = require_mapping(item.get("run"), "verified run")
        fingerprint = str(run["fingerprint"])
        selected = [
            row
            for row in conditions
            if row["fingerprint"] == fingerprint
        ]
        learned = [
            row for row in selected if row["method"] == LEARNED_METHOD
        ]
        control = [
            row for row in selected if row["method"] == SCREEN_CONTROL
        ]
        if len(learned) != 4 or len(control) != 4:
            raise ValueError(
                "each Phase 2 fold requires four matched conditions"
            )
        bc = _bc_row(run)
        bc_rows.append(bc)
        policy = require_mapping(run.get("policy"), "policy")
        training = require_mapping(policy.get("training"), "training")
        bc_seconds, ppo_seconds, evaluation_seconds, total = (
            _recorded_runtime(run)
        )
        learned_asr = mean(learned, "asr")
        learned_auc = mean(learned, "asr_query_auc")
        control_asr = mean(control, "asr")
        control_auc = mean(control, "asr_query_auc")
        source_gate = require_mapping(
            run.get("source_competence_gate"),
            "source competence gate",
        )
        folds.append(
            {
                "fingerprint": fingerprint,
                "seed": int(run["seed"]),
                "omitted_target_family": str(
                    run["target_family"]
                ),
                "learned_asr": learned_asr,
                "score_greedy_asr": control_asr,
                "asr_gain": learned_asr - control_asr,
                "learned_auc": learned_auc,
                "score_greedy_auc": control_auc,
                "auc_gain": learned_auc - control_auc,
                "trained_episodes": nonnegative_integer(
                    training.get("trained_episodes"),
                    "trained episodes",
                ),
                "source_calls": nonnegative_integer(
                    training.get("source_calls"),
                    "source calls",
                ),
                "bc_seconds": bc_seconds,
                "ppo_seconds": ppo_seconds,
                "source_evaluation_seconds": evaluation_seconds,
                "recorded_total_seconds": total,
                "bc_gate_passed": bool(bc["gate_passed"]),
                "strict_source_gate_passed": (
                    source_gate.get("passed") is True
                ),
            }
        )
    return folds, bc_rows


def training_block_rows(
    verified_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in verified_runs:
        run = require_mapping(item.get("run"), "verified run")
        training = require_mapping(
            require_mapping(run.get("policy"), "policy").get(
                "training"
            ),
            "training",
        )
        blocks = require_sequence(training.get("blocks"), "PPO blocks")
        for block_index, block_value in enumerate(blocks):
            block = require_mapping(block_value, "PPO block")
            families = require_mapping(
                block.get("family_diagnostics"),
                "PPO family diagnostics",
            )
            family_records = [
                require_mapping(value, "PPO family diagnostic")
                for value in families.values()
            ]
            ppo = require_mapping(block.get("ppo"), "PPO metrics")
            rows.append(
                {
                    "fingerprint": str(run["fingerprint"]),
                    "seed": int(run["seed"]),
                    "omitted_target_family": str(
                        run["target_family"]
                    ),
                    "block_index": block_index,
                    "episode_offset": nonnegative_integer(
                        block.get("episode_offset"),
                        "episode offset",
                    ),
                    "scheduled_episodes": nonnegative_integer(
                        block.get("episodes"),
                        "block episodes",
                    ),
                    "trained_episodes": nonnegative_integer(
                        block.get("trained_episodes"),
                        "block trained episodes",
                    ),
                    "eligible_episodes": sum(
                        nonnegative_integer(
                            record.get("eligible_episodes"),
                            "eligible episodes",
                        )
                        for record in family_records
                    ),
                    "successful_episodes": sum(
                        nonnegative_integer(
                            record.get("successful_episodes"),
                            "successful episodes",
                        )
                        for record in family_records
                    ),
                    "mean_episode_return": sum(
                        finite_number(
                            require_mapping(
                                record.get("episode_return"),
                                "episode return",
                            ).get("mean"),
                            "mean episode return",
                        )
                        for record in family_records
                    )
                    / len(family_records),
                    "mean_margin_reduction": sum(
                        finite_number(
                            require_mapping(
                                record.get("margin_reduction"),
                                "margin reduction",
                            ).get("mean"),
                            "mean margin reduction",
                        )
                        for record in family_records
                    )
                    / len(family_records),
                    "source_calls": nonnegative_integer(
                        block.get("source_calls"),
                        "block source calls",
                    ),
                    "elapsed_seconds": finite_number(
                        block.get("elapsed_seconds"),
                        "block elapsed seconds",
                    ),
                    "ppo_loss": finite_number(
                        ppo.get("loss"),
                        "PPO loss",
                    ),
                    "ppo_policy_loss": finite_number(
                        ppo.get("policy_loss"),
                        "PPO policy loss",
                    ),
                    "ppo_value_loss": finite_number(
                        ppo.get("value_loss"),
                        "PPO value loss",
                    ),
                    "ppo_entropy": finite_number(
                        ppo.get("entropy"),
                        "PPO entropy",
                    ),
                }
            )
    return rows


def victim_rows(
    verified_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in verified_runs:
        run = require_mapping(item.get("run"), "verified run")
        instances = require_mapping(
            run.get("victim_instances"),
            "victim instances",
        )
        for source_family, records_value in instances.items():
            records = require_sequence(
                records_value,
                "victim instance records",
            )
            for instance_index, record_value in enumerate(records):
                record = require_mapping(
                    record_value,
                    "victim instance",
                )
                rows.append(
                    {
                        "fingerprint": str(run["fingerprint"]),
                        "omitted_target_family": str(
                            run["target_family"]
                        ),
                        "source_family": str(source_family),
                        "instance_index": nonnegative_integer(
                            record.get(
                                "instance_index",
                                instance_index,
                            ),
                            "victim instance index",
                        ),
                        "victim_id": str(record["victim_id"]),
                        "source_validation_accuracy": finite_number(
                            record.get("source_validation_accuracy"),
                            "victim validation accuracy",
                        ),
                        "checkpoint_sha256": digest(
                            record.get("checkpoint_sha256"),
                            "victim checkpoint checksum",
                        ),
                        "resumed": record.get("resumed") is True,
                    }
                )
    return rows
