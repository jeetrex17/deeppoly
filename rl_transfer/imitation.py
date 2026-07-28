"""Source-only behavior-cloning utilities for recurrent attack policies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .actions import apply_action, patch_catalog
from .audit import AuditedVictim
from .config import AttackConfig
from .features import configured_patch_image_features
from .operator import AttackOperatorContract, choose_attack_transition
from .population import balanced_family_schedule
from .recurrent import RecurrentAttackPolicy
from .research_protocol import calibration_resistant_observation
from .rewards import recurrent_attack_reward, score_margin
from .soft_distillation import (
    evaluate_behavior_clone_policy_impl,
    fit_behavior_clone_policy,
    validated_action_distribution,
)


@dataclass(frozen=True)
class BehaviorCloneStep:
    observation: tuple[float, ...]
    action: int
    accepted: bool
    trajectory_id: str
    step_index: int
    action_distribution: tuple[float, ...] | None

    def __init__(
        self,
        observation: Iterable[float] | np.ndarray,
        action: int,
        accepted: bool,
        trajectory_id: str = "trajectory-0",
        step_index: int = 0,
        action_distribution: Iterable[float] | np.ndarray | None = None,
    ) -> None:
        numeric = tuple(float(value) for value in observation)
        if not numeric or any(not math.isfinite(value) for value in numeric):
            raise ValueError("behavior-cloning observations must be finite and non-empty")
        if not isinstance(action, int) or isinstance(action, bool) or action < 0:
            raise ValueError("behavior-cloning action must be a non-negative integer")
        if not isinstance(accepted, bool):
            raise ValueError("accepted must be boolean")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("trajectory_id must be a non-empty string")
        if (
            not isinstance(step_index, int)
            or isinstance(step_index, bool)
            or step_index < 0
        ):
            raise ValueError("step_index must be a non-negative integer")
        distribution = validated_action_distribution(
            action,
            action_distribution,
        )
        object.__setattr__(self, "observation", numeric)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "action_distribution", distribution)


def behavior_clone_policy(
    policy: RecurrentAttackPolicy,
    steps: Iterable[BehaviorCloneStep],
    *,
    epochs: int,
    seed: int,
    batch_size: int = 256,
) -> dict[str, object]:
    """Warm-start the actor from accepted source-teacher decisions only."""

    examples = tuple(steps)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("behavior-cloning epochs must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("behavior-cloning batch size must be a positive integer")
    if any(not isinstance(step, BehaviorCloneStep) for step in examples):
        raise ValueError("steps must contain BehaviorCloneStep values")
    return fit_behavior_clone_policy(
        policy,
        examples,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
    )


def evaluate_behavior_clone_policy(
    policy: RecurrentAttackPolicy,
    steps: Iterable[BehaviorCloneStep],
) -> dict[str, float | int | str]:
    return evaluate_behavior_clone_policy_impl(policy, tuple(steps))


def _victim_device(victim: nn.Module, image: torch.Tensor) -> torch.device:
    parameter = next(victim.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(victim.buffers(), None)
    return buffer.device if buffer is not None else image.device


def _normalize_victims(
    victims: Mapping[
        str,
        tuple[str, nn.Module] | Sequence[tuple[str, nn.Module]],
    ],
) -> dict[str, tuple[tuple[str, nn.Module], ...]]:
    normalized: dict[str, tuple[tuple[str, nn.Module], ...]] = {}
    for family, value in victims.items():
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], nn.Module)
        ):
            instances = (value,)
        else:
            instances = tuple(value)
        if not instances or any(
            not isinstance(instance, tuple)
            or len(instance) != 2
            or not isinstance(instance[0], str)
            or not isinstance(instance[1], nn.Module)
            for instance in instances
        ):
            raise ValueError("each demonstration family requires named victim modules")
        normalized[str(family)] = instances
    if not normalized:
        raise ValueError("at least one source family is required")
    return normalized


def collect_best_of_k_demonstrations(
    victims: Mapping[
        str,
        tuple[str, nn.Module] | Sequence[tuple[str, nn.Module]],
    ],
    samples: Sequence[tuple[torch.Tensor, int]],
    config: AttackConfig,
    *,
    episodes: int,
    candidates: int,
    decisions: int,
    seed: int,
) -> tuple[tuple[BehaviorCloneStep, ...], dict[str, object]]:
    """Collect source-only teacher decisions using privileged best-of-K queries."""

    if not samples:
        raise ValueError("demonstration samples are required")
    integer_controls = (episodes, candidates, decisions, seed)
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in integer_controls
    ):
        raise ValueError("demonstration controls must be integers")
    if episodes < 1 or candidates < 2 or decisions < 1:
        raise ValueError("episodes, candidates, and decisions must be positive")
    if candidates > config.action_dim:
        raise ValueError("teacher candidate count exceeds the action catalog")
    if not config.rollback_on_non_improvement:
        raise ValueError("best-of-K demonstrations require the matched rollback operator")

    normalized_victims = _normalize_victims(victims)
    schedule = balanced_family_schedule(
        tuple(normalized_victims),
        episodes,
        seed,
    )
    family_offsets = {family: 0 for family in normalized_victims}
    source_calls = {family: 0 for family in normalized_victims}
    eligible = {family: 0 for family in normalized_victims}
    successes = {family: 0 for family in normalized_victims}
    collected: list[BehaviorCloneStep] = []
    for episode, family in enumerate(schedule):
        instances = normalized_victims[family]
        victim_id, victim = instances[family_offsets[family] % len(instances)]
        family_offsets[family] += 1
        image, label = samples[episode % len(samples)]
        device = _victim_device(victim, image)
        original = image.detach().clone().float().clamp(0, 1).to(device)
        accepted_image = original.clone()
        budget = 1 + candidates * decisions
        oracle = AuditedVictim(victim, budget, "scores", victim_id)
        sample_id = f"bc-source:{family}:{episode}"
        accepted_response = oracle.query(
            accepted_image,
            sample_id,
            "teacher-initialization",
            0,
        )
        clean_scores = accepted_response.scores.clone()
        if accepted_response.predicted_label != label:
            source_calls[family] += oracle.calls
            continue
        eligible[family] += 1
        catalog = patch_catalog(config.grid_size, image.shape[0])
        action_counts = np.zeros(config.action_dim, dtype=np.float32)
        action_values = np.zeros(config.action_dim, dtype=np.float32)
        previous_action: int | None = None
        previous_reward = 0.0
        episode_success = False
        for decision in range(decisions):
            observation = calibration_resistant_observation(
                accepted_response.scores,
                label,
                clean_scores,
                (decisions - decision) / decisions,
                previous_action,
                config.action_dim,
                previous_reward,
                decision / max(1, decisions),
                action_counts if config.action_history_features else None,
                action_values if config.action_history_features else None,
                configured_patch_image_features(
                    original,
                    accepted_image,
                    config,
                ),
            )
            episode_key = hashlib.sha256(
                f"{seed}:{victim_id}:{episode}:{decision}".encode("utf-8")
            ).digest()
            candidate_rng = random.Random(int.from_bytes(episode_key[:8], "big"))
            candidate_indices = candidate_rng.sample(
                range(config.action_dim),
                candidates,
            )
            proposals: list[
                tuple[float, int, torch.Tensor, object, bool]
            ] = []
            for candidate_index in candidate_indices:
                proposal = apply_action(
                    accepted_image,
                    original,
                    catalog[candidate_index],
                    config.epsilon,
                    config.step_size,
                    config.grid_size,
                )
                response = oracle.query(
                    proposal,
                    sample_id,
                    "teacher-candidate",
                    decision,
                )
                success = response.predicted_label != label
                proposals.append(
                    (
                        score_margin(response.scores, label),
                        candidate_index,
                        proposal,
                        response,
                        success,
                    )
                )
            (
                proposal_margin,
                selected_action,
                selected_proposal,
                selected_response,
                episode_success,
            ) = min(proposals, key=lambda value: value[0])
            current_margin = score_margin(accepted_response.scores, label)
            transition = choose_attack_transition(
                accepted_image,
                selected_proposal,
                current_margin=current_margin,
                proposal_margin=proposal_margin,
                success=episode_success,
                rollback_on_non_improvement=True,
            )
            reward = recurrent_attack_reward(
                accepted_response.scores,
                selected_response.scores,
                label,
                episode_success,
                config,
            )
            collected.append(
                BehaviorCloneStep(
                    observation,
                    selected_action,
                    transition.accepted,
                    trajectory_id=sample_id,
                    step_index=decision,
                )
            )
            action_counts[selected_action] += 1.0
            action_values[selected_action] += (
                reward - action_values[selected_action]
            ) / action_counts[selected_action]
            previous_action, previous_reward = selected_action, reward
            if transition.accepted:
                accepted_image = transition.image
                accepted_response = selected_response
            if episode_success:
                successes[family] += 1
                break
        source_calls[family] += oracle.calls
    accepted_steps = sum(step.accepted for step in collected)
    return tuple(collected), {
        "teacher": "source_best_of_k",
        "episodes": episodes,
        "candidates_per_decision": candidates,
        "decisions_per_episode": decisions,
        "steps": len(collected),
        "accepted_steps": accepted_steps,
        "rejected_steps": len(collected) - accepted_steps,
        "eligible_episodes_by_family": eligible,
        "successful_episodes_by_family": successes,
        "source_calls_by_family": source_calls,
        "source_calls": sum(source_calls.values()),
        "operator": AttackOperatorContract.from_config(config).as_dict(),
        "operator_digest": AttackOperatorContract.from_config(config).digest(),
    }


def collect_gradient_demonstrations(
    victims: Mapping[
        str,
        tuple[str, nn.Module] | Sequence[tuple[str, nn.Module]],
    ],
    samples: Sequence[tuple[torch.Tensor, int]],
    config: AttackConfig,
    *,
    episodes: int,
    decisions: int,
    seed: int,
    soft_temperature: float | None = None,
) -> tuple[tuple[BehaviorCloneStep, ...], dict[str, object]]:
    """Label source actions with a privileged gradient teacher.

    Gradients are used only on owned source victims during representation
    pretraining. Deployment remains a score-based black-box attack.
    """

    if not samples:
        raise ValueError("demonstration samples are required")
    controls = (episodes, decisions, seed)
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in controls
    ):
        raise ValueError("gradient-teacher controls must be integers")
    if episodes < 1 or decisions < 1:
        raise ValueError("gradient-teacher episodes and decisions must be positive")
    if soft_temperature is not None and (
        isinstance(soft_temperature, bool)
        or not isinstance(soft_temperature, (int, float))
        or not math.isfinite(float(soft_temperature))
        or float(soft_temperature) <= 0
    ):
        raise ValueError("soft_temperature must be finite and positive")
    if not config.rollback_on_non_improvement:
        raise ValueError("gradient demonstrations require the matched rollback operator")

    normalized_victims = _normalize_victims(victims)
    schedule = balanced_family_schedule(
        tuple(normalized_victims),
        episodes,
        seed,
    )
    family_offsets = {family: 0 for family in normalized_victims}
    source_calls = {family: 0 for family in normalized_victims}
    gradient_evaluations = {family: 0 for family in normalized_victims}
    eligible = {family: 0 for family in normalized_victims}
    successes = {family: 0 for family in normalized_victims}
    soft_target_entropy_sum = 0.0
    soft_target_expected_regret_sum = 0.0
    soft_target_expected_normalized_regret_sum = 0.0
    soft_target_top1_mass_sum = 0.0
    soft_target_cost_scale_sum = 0.0
    soft_target_min_cost_scale = math.inf
    soft_target_max_cost_scale = 0.0
    soft_target_count = 0
    soft_target_generated_count = 0
    collected: list[BehaviorCloneStep] = []
    for episode, family in enumerate(schedule):
        instances = normalized_victims[family]
        victim_id, victim = instances[family_offsets[family] % len(instances)]
        family_offsets[family] += 1
        victim.eval()
        image, label = samples[episode % len(samples)]
        device = _victim_device(victim, image)
        original = image.detach().clone().float().clamp(0, 1).to(device)
        accepted_image = original.clone()
        oracle = AuditedVictim(victim, 1 + decisions, "scores", victim_id)
        sample_id = f"bc-gradient-source:{family}:{episode}"
        accepted_response = oracle.query(
            accepted_image,
            sample_id,
            "teacher-initialization",
            0,
        )
        clean_scores = accepted_response.scores.clone()
        if accepted_response.predicted_label != label:
            source_calls[family] += oracle.calls
            continue
        eligible[family] += 1
        catalog = patch_catalog(config.grid_size, image.shape[0])
        action_counts = np.zeros(config.action_dim, dtype=np.float32)
        action_values = np.zeros(config.action_dim, dtype=np.float32)
        previous_action: int | None = None
        previous_reward = 0.0
        for decision in range(decisions):
            observation = calibration_resistant_observation(
                accepted_response.scores,
                label,
                clean_scores,
                (decisions - decision) / decisions,
                previous_action,
                config.action_dim,
                previous_reward,
                decision / max(1, decisions),
                action_counts if config.action_history_features else None,
                action_values if config.action_history_features else None,
                configured_patch_image_features(
                    original,
                    accepted_image,
                    config,
                ),
            )
            differentiable = accepted_image.detach().clone().requires_grad_(True)
            logits = victim(differentiable.unsqueeze(0))[0]
            if logits.ndim != 1 or not torch.isfinite(logits).all():
                raise ValueError("gradient teacher requires finite class logits")
            rival = torch.cat(
                (logits[:label], logits[label + 1 :])
            ).max()
            margin = logits[label] - rival
            gradient = torch.autograd.grad(margin, differentiable)[0].detach()
            gradient_evaluations[family] += 1
            candidate_indices = list(range(config.action_dim))
            proposals: list[tuple[float, int, torch.Tensor]] = []
            for action_index in candidate_indices:
                proposal = apply_action(
                    accepted_image,
                    original,
                    catalog[action_index],
                    config.epsilon,
                    config.step_size,
                    config.grid_size,
                )
                linearized_change = float(
                    (gradient * (proposal - accepted_image)).sum()
                )
                proposals.append(
                    (linearized_change, action_index, proposal)
                )
            _, selected_action, selected_proposal = min(
                proposals,
                key=lambda value: value[0],
            )
            action_distribution: tuple[float, ...] | None = None
            soft_diagnostics: tuple[
                float,
                float,
                float,
                float,
                float,
            ] | None = None
            if soft_temperature is not None:
                linearized_costs = np.asarray(
                    tuple(value[0] for value in proposals),
                    dtype=np.float64,
                )
                if not np.isfinite(linearized_costs).all():
                    raise ValueError(
                        "soft gradient teacher requires finite linearized costs"
                    )
                relative_costs = linearized_costs - linearized_costs.min()
                cost_scale = max(
                    float(np.std(linearized_costs)),
                    float(np.finfo(np.float64).eps),
                )
                normalized_relative_costs = relative_costs / cost_scale
                unnormalized = np.exp(
                    -normalized_relative_costs / float(soft_temperature)
                )
                normalization = float(unnormalized.sum())
                if not math.isfinite(normalization) or normalization <= 0:
                    raise ValueError(
                        "soft gradient teacher produced an invalid normalization"
                    )
                probabilities = unnormalized / normalization
                action_distribution = tuple(
                    float(value) for value in probabilities
                )
                positive = probabilities > 0
                target_entropy = float(
                    -np.sum(
                        probabilities[positive]
                        * np.log(probabilities[positive])
                    )
                )
                expected_regret = float(
                    np.dot(probabilities, relative_costs),
                )
                expected_normalized_regret = float(
                    np.dot(probabilities, normalized_relative_costs),
                )
                soft_diagnostics = (
                    target_entropy,
                    expected_regret,
                    expected_normalized_regret,
                    float(probabilities.max()),
                    cost_scale,
                )
                soft_target_generated_count += 1
            selected_response = oracle.query(
                selected_proposal,
                sample_id,
                "gradient-teacher-action",
                decision,
            )
            success = selected_response.predicted_label != label
            current_margin = score_margin(accepted_response.scores, label)
            proposal_margin = score_margin(selected_response.scores, label)
            transition = choose_attack_transition(
                accepted_image,
                selected_proposal,
                current_margin=current_margin,
                proposal_margin=proposal_margin,
                success=success,
                rollback_on_non_improvement=True,
            )
            if transition.accepted and soft_diagnostics is not None:
                (
                    target_entropy,
                    expected_regret,
                    expected_normalized_regret,
                    target_top1_mass,
                    cost_scale,
                ) = soft_diagnostics
                soft_target_entropy_sum += target_entropy
                soft_target_expected_regret_sum += expected_regret
                soft_target_expected_normalized_regret_sum += (
                    expected_normalized_regret
                )
                soft_target_top1_mass_sum += target_top1_mass
                soft_target_cost_scale_sum += cost_scale
                soft_target_min_cost_scale = min(
                    soft_target_min_cost_scale,
                    cost_scale,
                )
                soft_target_max_cost_scale = max(
                    soft_target_max_cost_scale,
                    cost_scale,
                )
                soft_target_count += 1
            reward = recurrent_attack_reward(
                accepted_response.scores,
                selected_response.scores,
                label,
                success,
                config,
            )
            collected.append(
                BehaviorCloneStep(
                    observation,
                    selected_action,
                    transition.accepted,
                    trajectory_id=sample_id,
                    step_index=decision,
                    action_distribution=action_distribution,
                )
            )
            action_counts[selected_action] += 1.0
            action_values[selected_action] += (
                reward - action_values[selected_action]
            ) / action_counts[selected_action]
            previous_action, previous_reward = selected_action, reward
            if transition.accepted:
                accepted_image = transition.image
                accepted_response = selected_response
            if success:
                successes[family] += 1
                break
        source_calls[family] += oracle.calls
    accepted_steps = sum(step.accepted for step in collected)
    metrics: dict[str, object] = {
        "teacher": "source_privileged_cw_logit_gradient",
        "episodes": episodes,
        "decisions_per_episode": decisions,
        "steps": len(collected),
        "accepted_steps": accepted_steps,
        "rejected_steps": len(collected) - accepted_steps,
        "eligible_episodes_by_family": eligible,
        "successful_episodes_by_family": successes,
        "source_calls_by_family": source_calls,
        "source_calls": sum(source_calls.values()),
        "gradient_evaluations_by_family": gradient_evaluations,
        "gradient_evaluations": sum(gradient_evaluations.values()),
        "operator": AttackOperatorContract.from_config(config).as_dict(),
        "operator_digest": AttackOperatorContract.from_config(config).digest(),
    }
    if soft_temperature is not None:
        denominator = max(1, soft_target_count)
        metrics.update(
            {
                "soft_temperature": float(soft_temperature),
                "soft_target_cost_normalization": (
                    "per_state_standard_deviation"
                ),
                "soft_target_metric_scope": (
                    "accepted_behavior_clone_steps"
                ),
                "soft_target_generated_count": soft_target_generated_count,
                "soft_target_count": soft_target_count,
                "soft_target_mean_entropy": (
                    soft_target_entropy_sum / denominator
                ),
                "soft_target_mean_expected_linearized_regret": (
                    soft_target_expected_regret_sum / denominator
                ),
                "soft_target_mean_expected_normalized_regret": (
                    soft_target_expected_normalized_regret_sum
                    / denominator
                ),
                "soft_target_mean_top1_mass": (
                    soft_target_top1_mass_sum / denominator
                ),
                "soft_target_mean_cost_scale": (
                    soft_target_cost_scale_sum / denominator
                ),
                "soft_target_min_cost_scale": (
                    soft_target_min_cost_scale
                    if soft_target_count
                    else 0.0
                ),
                "soft_target_max_cost_scale": (
                    soft_target_max_cost_scale
                    if soft_target_count
                    else 0.0
                ),
            }
        )
    return tuple(collected), metrics
