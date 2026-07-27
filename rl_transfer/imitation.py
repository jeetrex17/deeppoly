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
from .features import patch_image_features
from .operator import AttackOperatorContract, choose_attack_transition
from .population import balanced_family_schedule
from .recurrent import RecurrentAttackPolicy
from .research_protocol import calibration_resistant_observation
from .rewards import recurrent_attack_reward, score_margin


@dataclass(frozen=True)
class BehaviorCloneStep:
    observation: tuple[float, ...]
    action: int
    accepted: bool
    trajectory_id: str
    step_index: int

    def __init__(
        self,
        observation: Iterable[float] | np.ndarray,
        action: int,
        accepted: bool,
        trajectory_id: str = "trajectory-0",
        step_index: int = 0,
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
        object.__setattr__(self, "observation", numeric)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "step_index", step_index)


def _group_trajectories(
    examples: Sequence[BehaviorCloneStep],
) -> tuple[tuple[BehaviorCloneStep, ...], ...]:
    grouped: dict[str, list[BehaviorCloneStep]] = {}
    for step in examples:
        grouped.setdefault(step.trajectory_id, []).append(step)
    return tuple(
        tuple(
            sorted(
                trajectory,
                key=lambda step: step.step_index,
            )
        )
        for trajectory in grouped.values()
    )


def _sequence_tensors(
    trajectories: Sequence[Sequence[BehaviorCloneStep]],
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


def _frequency_baselines(
    actions: Sequence[int],
    action_dim: int,
) -> dict[str, float]:
    counts = np.bincount(
        np.asarray(actions, dtype=int),
        minlength=action_dim,
    )
    total = int(counts.sum())
    probabilities = (counts + 1) / (total + action_dim)
    return {
        "majority_accuracy": float(counts.max() / total),
        "frequency_nll": float(
            -np.mean(
                np.log(
                    probabilities[
                        np.asarray(actions, dtype=int)
                    ]
                )
            )
        ),
    }


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
    accepted = tuple(step for step in examples if step.accepted)
    if not accepted:
        raise ValueError("behavior cloning requires at least one accepted source action")
    if any(len(step.observation) != policy.observation_dim for step in accepted):
        raise ValueError("demonstration observation dimension does not match the policy")
    if any(step.action >= policy.action_dim for step in accepted):
        raise ValueError("demonstration action is outside the policy action catalog")

    trajectories = _group_trajectories(examples)
    device = next(policy.parameters()).device
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    policy.train()
    for epoch in range(epochs):
        order = rng.permutation(len(trajectories))
        loss_sum = 0.0
        correct = 0
        count = 0
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
            logits = _sequence_logits(policy, observations, valid)
            selected_logits = logits[accepted_mask]
            selected_actions = actions[accepted_mask]
            if not len(selected_actions):
                continue
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
                (
                    selected_logits.argmax(1)
                    == selected_actions
                ).sum()
            )
            count += selected_count
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / count,
                "accuracy": correct / count,
            }
        )
    policy.eval()
    baselines = _frequency_baselines(
        tuple(step.action for step in accepted),
        policy.action_dim,
    )
    return {
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
        **baselines,
    }


def evaluate_behavior_clone_policy(
    policy: RecurrentAttackPolicy,
    steps: Iterable[BehaviorCloneStep],
) -> dict[str, float | int]:
    examples = tuple(steps)
    accepted = tuple(step for step in examples if step.accepted)
    if not accepted:
        raise ValueError("behavior-cloning evaluation requires accepted actions")
    if any(len(step.observation) != policy.observation_dim for step in accepted):
        raise ValueError("evaluation observation dimension does not match the policy")
    if any(step.action >= policy.action_dim for step in accepted):
        raise ValueError("evaluation action is outside the policy catalog")
    trajectories = _group_trajectories(examples)
    device = next(policy.parameters()).device
    observations, actions, valid, accepted_mask = _sequence_tensors(
        trajectories,
        policy.observation_dim,
        device,
    )
    with torch.inference_mode():
        logits = _sequence_logits(policy, observations, valid)
        selected_logits = logits[accepted_mask]
        selected_actions = actions[accepted_mask]
        nll = nn.functional.cross_entropy(
            selected_logits,
            selected_actions,
        )
        accuracy = (
            selected_logits.argmax(1) == selected_actions
        ).float().mean()
    baselines = _frequency_baselines(
        tuple(step.action for step in accepted),
        policy.action_dim,
    )
    return {
        "training_mode": "sequence_filtered_hindsight_imitation",
        "trajectories": len(trajectories),
        "accepted_steps": len(accepted),
        "nll": float(nll),
        "accuracy": float(accuracy),
        "uniform_nll": math.log(policy.action_dim),
        "uniform_accuracy": 1.0 / policy.action_dim,
        **baselines,
    }


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
                (
                    patch_image_features(
                        original,
                        accepted_image,
                        grid_size=config.grid_size,
                    )
                    if config.image_patch_features
                    else None
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
                (
                    patch_image_features(
                        original,
                        accepted_image,
                        grid_size=config.grid_size,
                    )
                    if config.image_patch_features
                    else None
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
    return tuple(collected), {
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
