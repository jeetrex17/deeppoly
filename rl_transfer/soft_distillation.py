"""Sequence behavior cloning with optional soft action targets."""

from __future__ import annotations

import math
from typing import Iterable, Protocol, Sequence

import numpy as np
import torch
from torch import nn

from .recurrent import RecurrentAttackPolicy


class BehaviorCloneRecord(Protocol):
    observation: tuple[float, ...]
    action: int
    accepted: bool
    trajectory_id: str
    step_index: int
    action_distribution: tuple[float, ...] | None


def validated_action_distribution(
    action: int,
    values: Iterable[float] | np.ndarray | None,
) -> tuple[float, ...] | None:
    """Copy and validate a complete teacher distribution."""

    if values is None:
        return None
    try:
        distribution = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "action_distribution must contain numeric probabilities"
        ) from error
    if (
        not distribution
        or action >= len(distribution)
        or any(
            not math.isfinite(value) or value < 0
            for value in distribution
        )
    ):
        raise ValueError(
            "action_distribution must cover the action with finite "
            "non-negative probabilities"
        )
    probability_sum = math.fsum(distribution)
    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("action_distribution probabilities must sum to one")
    if distribution[action] < max(distribution) - 1e-12:
        raise ValueError(
            "behavior-cloning action must maximize action_distribution"
        )
    return tuple(value / probability_sum for value in distribution)


def _group_trajectories(
    examples: Sequence[BehaviorCloneRecord],
) -> tuple[tuple[BehaviorCloneRecord, ...], ...]:
    grouped: dict[str, list[BehaviorCloneRecord]] = {}
    for step in examples:
        grouped.setdefault(step.trajectory_id, []).append(step)
    return tuple(
        tuple(sorted(trajectory, key=lambda step: step.step_index))
        for trajectory in grouped.values()
    )


def _sequence_tensors(
    trajectories: Sequence[Sequence[BehaviorCloneRecord]],
    observation_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(trajectories)
    max_steps = max(len(trajectory) for trajectory in trajectories)
    observations = torch.zeros(
        (batch_size, max_steps, observation_dim),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.zeros(
        (batch_size, max_steps),
        dtype=torch.long,
        device=device,
    )
    valid = torch.zeros(
        (batch_size, max_steps),
        dtype=torch.bool,
        device=device,
    )
    accepted = torch.zeros_like(valid)
    for trajectory_index, trajectory in enumerate(trajectories):
        for step_index, step in enumerate(trajectory):
            observations[trajectory_index, step_index] = torch.tensor(
                step.observation,
                dtype=torch.float32,
                device=device,
            )
            actions[trajectory_index, step_index] = step.action
            valid[trajectory_index, step_index] = True
            accepted[trajectory_index, step_index] = step.accepted
    return observations, actions, valid, accepted


def _sequence_soft_targets(
    trajectories: Sequence[Sequence[BehaviorCloneRecord]],
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Materialize one minibatch, with one-hot hard-label fallbacks."""

    targets = torch.zeros(
        (
            len(trajectories),
            max(len(trajectory) for trajectory in trajectories),
            action_dim,
        ),
        dtype=torch.float32,
        device=device,
    )
    for trajectory_index, trajectory in enumerate(trajectories):
        for step_index, step in enumerate(trajectory):
            if not step.accepted:
                continue
            if step.action_distribution is None:
                targets[trajectory_index, step_index, step.action] = 1.0
            else:
                targets[trajectory_index, step_index] = torch.as_tensor(
                    step.action_distribution,
                    dtype=torch.float32,
                    device=device,
                )
    return targets


def _sequence_logits(
    policy: RecurrentAttackPolicy,
    observations: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    hidden = policy.initial_state(batch_size=observations.shape[0])
    logits_by_step: list[torch.Tensor] = []
    for step_index in range(observations.shape[1]):
        logits, _, proposed_hidden = policy(
            observations[:, step_index],
            hidden,
        )
        step_valid = valid[:, step_index].unsqueeze(1)
        hidden = torch.where(step_valid, proposed_hidden, hidden)
        logits_by_step.append(logits)
    return torch.stack(logits_by_step, dim=1)


def _empirical_label_baselines(
    actions: Sequence[int],
    action_dim: int,
) -> dict[str, float]:
    numeric_actions = np.asarray(actions, dtype=int)
    counts = np.bincount(numeric_actions, minlength=action_dim)
    total = int(counts.sum())
    probabilities = counts / total
    return {
        "empirical_top1_accuracy": float(counts.max() / total),
        "empirical_nll": float(
            -np.mean(np.log(probabilities[numeric_actions]))
        ),
    }


def _soft_target_baselines(
    accepted: Sequence[BehaviorCloneRecord],
    action_dim: int,
) -> dict[str, float]:
    """Compute the best constant soft baseline with O(action_dim) memory."""

    target_entropy_sum = 0.0
    target_mass = np.zeros(action_dim, dtype=np.float64)
    for step in accepted:
        if step.action_distribution is None:
            target_mass[step.action] += 1.0
            continue
        target = np.asarray(step.action_distribution, dtype=np.float64)
        positive = target > 0
        target_entropy_sum += float(
            -np.sum(target[positive] * np.log(target[positive]))
        )
        target_mass += target
    target_entropy = target_entropy_sum / len(accepted)
    empirical_probabilities = target_mass / len(accepted)
    positive_mass = target_mass > 0
    empirical_soft_ce = float(
        -np.dot(
            target_mass[positive_mass],
            np.log(empirical_probabilities[positive_mass]),
        )
        / len(accepted)
    )
    uniform_soft_ce = math.log(action_dim)
    return {
        "uniform_soft_cross_entropy": uniform_soft_ce,
        "uniform_soft_kl": max(0.0, uniform_soft_ce - target_entropy),
        "empirical_soft_cross_entropy": empirical_soft_ce,
        "empirical_soft_kl": max(0.0, empirical_soft_ce - target_entropy),
    }


def _soft_classification_baselines(
    hard_actions: Sequence[int],
    action_dim: int,
) -> dict[str, float]:
    counts = np.bincount(
        np.asarray(hard_actions, dtype=int),
        minlength=action_dim,
    )
    total = int(counts.sum())
    top5_count = int(np.sort(counts)[-min(5, action_dim):].sum())
    return {
        "uniform_top1_accuracy": 1.0 / action_dim,
        "uniform_top5_accuracy": min(5, action_dim) / action_dim,
        "empirical_top1_accuracy": float(counts.max() / total),
        "empirical_top5_accuracy": float(top5_count / total),
    }


def _validate_examples(
    policy: RecurrentAttackPolicy,
    examples: Sequence[BehaviorCloneRecord],
    *,
    evaluation: bool,
) -> tuple[BehaviorCloneRecord, ...]:
    accepted = tuple(step for step in examples if step.accepted)
    if not accepted:
        message = (
            "behavior-cloning evaluation requires accepted actions"
            if evaluation
            else "behavior cloning requires at least one accepted source action"
        )
        raise ValueError(message)
    prefix = "evaluation" if evaluation else "demonstration"
    if any(len(step.observation) != policy.observation_dim for step in accepted):
        raise ValueError(
            f"{prefix} observation dimension does not match the policy"
        )
    if any(step.action >= policy.action_dim for step in accepted):
        raise ValueError(
            f"{prefix} action is outside the policy "
            f"{'catalog' if evaluation else 'action catalog'}"
        )
    if any(
        step.action_distribution is not None
        and len(step.action_distribution) != policy.action_dim
        for step in accepted
    ):
        raise ValueError(
            f"{prefix} action distribution does not match the policy catalog"
        )
    return accepted


def fit_behavior_clone_policy(
    policy: RecurrentAttackPolicy,
    examples: Sequence[BehaviorCloneRecord],
    *,
    epochs: int,
    seed: int,
    batch_size: int,
) -> dict[str, object]:
    """Warm-start the actor from accepted source-teacher decisions."""

    accepted = _validate_examples(policy, examples, evaluation=False)
    trajectories = _group_trajectories(examples)
    uses_soft_targets = any(
        step.action_distribution is not None for step in accepted
    )
    device = next(policy.parameters()).device
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    policy.train()
    for epoch in range(epochs):
        order = rng.permutation(len(trajectories))
        loss_sum = 0.0
        correct = 0
        top5_correct = 0
        count = 0
        soft_kl_sum = 0.0
        teacher_entropy_sum = 0.0
        teacher_probability_regret_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch = tuple(
                trajectories[int(index)]
                for index in order[start:start + batch_size]
            )
            observations, actions, valid, accepted_mask = _sequence_tensors(
                batch,
                policy.observation_dim,
                device,
            )
            selected_logits = _sequence_logits(
                policy,
                observations,
                valid,
            )[accepted_mask]
            selected_actions = actions[accepted_mask]
            if not len(selected_actions):
                continue
            if uses_soft_targets:
                targets = _sequence_soft_targets(
                    batch,
                    policy.action_dim,
                    device,
                )[accepted_mask]
                log_probabilities = nn.functional.log_softmax(
                    selected_logits,
                    dim=1,
                )
                loss_values = -(targets * log_probabilities).sum(dim=1)
                loss = loss_values.mean()
                target_logs = targets.clamp_min(1e-30).log()
                teacher_entropies = -(targets * target_logs).sum(dim=1)
                soft_kls = (
                    targets * (target_logs - log_probabilities)
                ).sum(dim=1).clamp_min(0.0)
                model_actions = selected_logits.argmax(1)
                teacher_probability_regrets = (
                    targets.max(dim=1).values
                    - targets.gather(
                        1,
                        model_actions.unsqueeze(1),
                    ).squeeze(1)
                ).clamp_min(0.0)
            else:
                loss = nn.functional.cross_entropy(
                    selected_logits,
                    selected_actions,
                )
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                policy.parameters(),
                policy.config.gradient_clip_norm,
            )
            policy.optimizer.step()
            selected_count = len(selected_actions)
            loss_sum += float(loss.detach()) * selected_count
            correct += int(
                (selected_logits.argmax(1) == selected_actions).sum()
            )
            if uses_soft_targets:
                top5_correct += int(
                    selected_logits.topk(
                        min(5, policy.action_dim),
                        dim=1,
                    ).indices.eq(
                        selected_actions.unsqueeze(1)
                    ).any(dim=1).sum()
                )
                soft_kl_sum += float(soft_kls.detach().sum())
                teacher_entropy_sum += float(teacher_entropies.detach().sum())
                teacher_probability_regret_sum += float(
                    teacher_probability_regrets.detach().sum()
                )
            count += selected_count
        epoch_metrics = {
            "epoch": float(epoch + 1),
            "loss": loss_sum / count,
            "accuracy": correct / count,
        }
        if uses_soft_targets:
            epoch_metrics.update(
                {
                    "top1_accuracy": correct / count,
                    "top5_accuracy": top5_correct / count,
                    "soft_cross_entropy": loss_sum / count,
                    "soft_kl": soft_kl_sum / count,
                    "teacher_entropy": teacher_entropy_sum / count,
                    "teacher_probability_regret": (
                        teacher_probability_regret_sum / count
                    ),
                }
            )
        history.append(epoch_metrics)
    policy.eval()
    label_baselines = _empirical_label_baselines(
        tuple(step.action for step in accepted),
        policy.action_dim,
    )
    result: dict[str, object] = {
        "training_mode": "sequence_filtered_hindsight_imitation",
        "trajectories": len(trajectories),
        "accepted_steps": len(accepted),
        "rejected_steps": len(examples) - len(accepted),
        "epochs": epochs,
        "uniform_accuracy": 1.0 / policy.action_dim,
        "uniform_nll": math.log(policy.action_dim),
        "final_loss": history[-1]["loss"],
        "final_accuracy": history[-1]["accuracy"],
        "history": history,
        "training_empirical_top1_accuracy": label_baselines[
            "empirical_top1_accuracy"
        ],
        "training_empirical_nll": label_baselines["empirical_nll"],
        "baseline_provenance": "training_labels_empirical_constant",
        "baseline_estimator": "empirical_best_constant_no_smoothing",
        "deprecated_frequency_aliases_present": True,
        "deprecated_frequency_alias_semantics": (
            "same_training_split_empirical_constant"
        ),
        "majority_accuracy": label_baselines[
            "empirical_top1_accuracy"
        ],
        "frequency_nll": label_baselines["empirical_nll"],
    }
    if uses_soft_targets:
        soft_baselines = _soft_target_baselines(
            accepted,
            policy.action_dim,
        )
        classification_baselines = _soft_classification_baselines(
            tuple(step.action for step in accepted),
            policy.action_dim,
        )
        result.update(
            {
                "target_mode": (
                    "soft"
                    if all(
                        step.action_distribution is not None
                        for step in accepted
                    )
                    else "mixed_soft_and_hard"
                ),
                "final_top1_accuracy": history[-1]["top1_accuracy"],
                "final_top5_accuracy": history[-1]["top5_accuracy"],
                "final_soft_cross_entropy": history[-1][
                    "soft_cross_entropy"
                ],
                "final_soft_kl": history[-1]["soft_kl"],
                "final_teacher_entropy": history[-1]["teacher_entropy"],
                "final_teacher_probability_regret": history[-1][
                    "teacher_probability_regret"
                ],
                "training_empirical_soft_cross_entropy": soft_baselines[
                    "empirical_soft_cross_entropy"
                ],
                "training_empirical_soft_kl": soft_baselines[
                    "empirical_soft_kl"
                ],
                "training_empirical_top1_accuracy": (
                    classification_baselines[
                        "empirical_top1_accuracy"
                    ]
                ),
                "training_empirical_top5_accuracy": (
                    classification_baselines[
                        "empirical_top5_accuracy"
                    ]
                ),
                "uniform_soft_cross_entropy": soft_baselines[
                    "uniform_soft_cross_entropy"
                ],
                "uniform_soft_kl": soft_baselines["uniform_soft_kl"],
                "uniform_top1_accuracy": classification_baselines[
                    "uniform_top1_accuracy"
                ],
                "uniform_top5_accuracy": classification_baselines[
                    "uniform_top5_accuracy"
                ],
            }
        )
    return result


def evaluate_behavior_clone_policy_impl(
    policy: RecurrentAttackPolicy,
    examples: Sequence[BehaviorCloneRecord],
) -> dict[str, float | int | str]:
    accepted = _validate_examples(policy, examples, evaluation=True)
    trajectories = _group_trajectories(examples)
    uses_soft_targets = any(
        step.action_distribution is not None for step in accepted
    )
    device = next(policy.parameters()).device
    observations, actions, valid, accepted_mask = _sequence_tensors(
        trajectories,
        policy.observation_dim,
        device,
    )
    with torch.inference_mode():
        selected_logits = _sequence_logits(
            policy,
            observations,
            valid,
        )[accepted_mask]
        selected_actions = actions[accepted_mask]
        nll = nn.functional.cross_entropy(
            selected_logits,
            selected_actions,
        )
        accuracy = (
            selected_logits.argmax(1) == selected_actions
        ).float().mean()
        if uses_soft_targets:
            top5_accuracy = selected_logits.topk(
                min(5, policy.action_dim),
                dim=1,
            ).indices.eq(
                selected_actions.unsqueeze(1)
            ).any(dim=1).float().mean()
            targets = _sequence_soft_targets(
                trajectories,
                policy.action_dim,
                device,
            )[accepted_mask]
            log_probabilities = nn.functional.log_softmax(
                selected_logits,
                dim=1,
            )
            target_logs = targets.clamp_min(1e-30).log()
            soft_cross_entropy = -(
                targets * log_probabilities
            ).sum(dim=1).mean()
            teacher_entropy = -(
                targets * target_logs
            ).sum(dim=1).mean()
            soft_kl = (
                targets * (target_logs - log_probabilities)
            ).sum(dim=1).clamp_min(0.0).mean()
            model_actions = selected_logits.argmax(1)
            teacher_probability_regret = (
                targets.max(dim=1).values
                - targets.gather(
                    1,
                    model_actions.unsqueeze(1),
                ).squeeze(1)
            ).clamp_min(0.0).mean()
    label_baselines = _empirical_label_baselines(
        tuple(step.action for step in accepted),
        policy.action_dim,
    )
    result: dict[str, float | int | str] = {
        "training_mode": "sequence_filtered_hindsight_imitation",
        "trajectories": len(trajectories),
        "accepted_steps": len(accepted),
        "nll": float(nll),
        "accuracy": float(accuracy),
        "uniform_nll": math.log(policy.action_dim),
        "uniform_accuracy": 1.0 / policy.action_dim,
        "validation_oracle_top1_accuracy": label_baselines[
            "empirical_top1_accuracy"
        ],
        "validation_oracle_nll": label_baselines["empirical_nll"],
        "baseline_provenance": "evaluated_labels_validation_oracle",
        "baseline_estimator": "empirical_best_constant_no_smoothing",
        "deprecated_frequency_aliases_present": True,
        "deprecated_frequency_alias_semantics": (
            "validation_oracle_estimated_from_evaluated_labels"
        ),
        "majority_accuracy": label_baselines[
            "empirical_top1_accuracy"
        ],
        "frequency_nll": label_baselines["empirical_nll"],
    }
    if uses_soft_targets:
        soft_baselines = _soft_target_baselines(
            accepted,
            policy.action_dim,
        )
        classification_baselines = _soft_classification_baselines(
            tuple(step.action for step in accepted),
            policy.action_dim,
        )
        result.update(
            {
                "target_mode": (
                    "soft"
                    if all(
                        step.action_distribution is not None
                        for step in accepted
                    )
                    else "mixed_soft_and_hard"
                ),
                "top1_accuracy": float(accuracy),
                "top5_accuracy": float(top5_accuracy),
                "soft_cross_entropy": float(soft_cross_entropy),
                "soft_kl": float(soft_kl),
                "teacher_entropy": float(teacher_entropy),
                "teacher_probability_regret": float(
                    teacher_probability_regret
                ),
                "validation_oracle_soft_cross_entropy": soft_baselines[
                    "empirical_soft_cross_entropy"
                ],
                "validation_oracle_soft_kl": soft_baselines[
                    "empirical_soft_kl"
                ],
                "validation_oracle_top1_accuracy": (
                    classification_baselines[
                        "empirical_top1_accuracy"
                    ]
                ),
                "validation_oracle_top5_accuracy": (
                    classification_baselines[
                        "empirical_top5_accuracy"
                    ]
                ),
                "uniform_soft_cross_entropy": soft_baselines[
                    "uniform_soft_cross_entropy"
                ],
                "uniform_soft_kl": soft_baselines["uniform_soft_kl"],
                "uniform_top1_accuracy": classification_baselines[
                    "uniform_top1_accuracy"
                ],
                "uniform_top5_accuracy": classification_baselines[
                    "uniform_top5_accuracy"
                ],
            }
        )
    return result
