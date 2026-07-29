"""Bounded source-only PPO refinement for the D1 residual ranker."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch
from torch import nn

from .actions import apply_action, patch_catalog
from .audit import AuditedVictim
from .config import AttackConfig
from .features import configured_patch_image_features
from .operator import choose_attack_transition
from .phase2_residual_d1 import D1_MAX_PPO_EPISODES, D1_SOURCE_FAMILIES
from .population import FamilyRobustWeights, balanced_family_schedule
from .research_protocol import calibration_resistant_observation
from .residual_ranker import ResidualRankerPolicy, score_greedy_action_order
from .rewards import recurrent_attack_reward, score_margin


RESIDUAL_PPO_BLOCK_EPISODES = 50
RESIDUAL_PPO_QUERY_BUDGET = 50
_RETURN_DISCOUNT = 0.98


@dataclass(frozen=True)
class _ResidualPPOSequence:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    prior_order: tuple[int, ...]


def _distribution_summary(values: Sequence[float]) -> dict[str, object]:
    numeric = tuple(float(value) for value in values)
    std = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric) if numeric else None,
        "std": std if numeric else None,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
    }


def _validate_controls(
    policy: ResidualRankerPolicy,
    config: AttackConfig,
    *,
    episodes: int,
    seed: int,
    prior_seed: int,
    episode_offset: int,
    deadline_check: Callable[[], None] | None,
) -> None:
    if not isinstance(policy, ResidualRankerPolicy):
        raise TypeError("residual PPO requires a ResidualRankerPolicy")
    if not isinstance(config, AttackConfig):
        raise TypeError("residual PPO requires an AttackConfig")
    if (
        config.max_queries != RESIDUAL_PPO_QUERY_BUDGET
        or not config.rollback_on_non_improvement
    ):
        raise ValueError("residual PPO requires the matched 50-query rollback operator")
    if (
        isinstance(episodes, bool)
        or not isinstance(episodes, int)
        or not 1 <= episodes <= RESIDUAL_PPO_BLOCK_EPISODES
    ):
        raise ValueError("residual PPO blocks require between 1 and 50 episodes")
    if (
        isinstance(episode_offset, bool)
        or not isinstance(episode_offset, int)
        or episode_offset < 0
        or episode_offset + episodes > D1_MAX_PPO_EPISODES
    ):
        raise ValueError("residual PPO episode bounds cannot exceed 200")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (seed, prior_seed)
    ):
        raise ValueError("residual PPO seeds must be integers")
    if deadline_check is not None and not callable(deadline_check):
        raise TypeError("deadline_check must be callable")
    if policy.action_dim != config.action_dim:
        raise ValueError("residual PPO policy and action catalog do not match")
    if policy.backbone.observation_dim != config.recurrent_observation_dim:
        raise ValueError("residual PPO observation dimensions do not match")


def _source_instances(
    source_victims: Mapping[
        str,
        tuple[str, nn.Module] | Sequence[tuple[str, nn.Module]],
    ],
) -> dict[str, tuple[tuple[str, nn.Module], ...]]:
    if not isinstance(source_victims, Mapping):
        raise TypeError("source victims must be a family mapping")
    if set(source_victims) != set(D1_SOURCE_FAMILIES):
        raise ValueError(
            "residual PPO accepts only the locked source families; "
            "held-out families are forbidden"
        )
    normalized: dict[str, tuple[tuple[str, nn.Module], ...]] = {}
    victim_ids: set[str] = set()
    for family in D1_SOURCE_FAMILIES:
        value = source_victims[family]
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], nn.Module)
        ):
            instances = (value,)
        else:
            try:
                instances = tuple(value)
            except TypeError as error:
                raise ValueError(
                    "each source family requires named victim modules"
                ) from error
        if not instances:
            raise ValueError("each source family requires a victim instance")
        for instance in instances:
            if (
                not isinstance(instance, tuple)
                or len(instance) != 2
                or not isinstance(instance[0], str)
                or not instance[0]
                or not isinstance(instance[1], nn.Module)
            ):
                raise ValueError("each source family requires named victim modules")
            if instance[0] in victim_ids:
                raise ValueError("source victim IDs must be globally unique")
            victim_ids.add(instance[0])
        normalized[family] = instances
    return normalized


def _validated_samples(
    samples: Sequence[tuple[torch.Tensor, int]],
    config: AttackConfig,
) -> tuple[tuple[torch.Tensor, int], ...]:
    try:
        source_samples = tuple(samples)
    except TypeError as error:
        raise ValueError("residual PPO requires source samples") from error
    if not source_samples:
        raise ValueError("residual PPO requires source samples")
    for sample in source_samples:
        if (
            not isinstance(sample, tuple)
            or len(sample) != 2
            or not isinstance(sample[0], torch.Tensor)
            or sample[0].ndim != 3
            or sample[0].shape[0] != 3
            or not sample[0].is_floating_point()
            or not bool(torch.isfinite(sample[0]).all())
            or sample[0].shape[1] < config.grid_size
            or sample[0].shape[2] < config.grid_size
            or sample[0].shape[1] % config.grid_size
            or sample[0].shape[2] % config.grid_size
            or isinstance(sample[1], bool)
            or not isinstance(sample[1], int)
            or sample[1] < 0
        ):
            raise ValueError(
                "source samples must contain finite RGB tensors and labels"
            )
    return source_samples


def _initial_weights(
    initial_family_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if initial_family_weights is None:
        uniform = 1.0 / len(D1_SOURCE_FAMILIES)
        return {family: uniform for family in D1_SOURCE_FAMILIES}
    if set(initial_family_weights) != set(D1_SOURCE_FAMILIES):
        raise ValueError("initial family weights must match source families")
    numeric: dict[str, float] = {}
    for family in D1_SOURCE_FAMILIES:
        value = initial_family_weights[family]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("initial family weights must be finite and non-negative")
        numeric[family] = float(value)
    total = sum(numeric.values())
    if total <= 0:
        raise ValueError("initial family weights must have positive mass")
    return {family: numeric[family] / total for family in D1_SOURCE_FAMILIES}


def _starting_offsets(
    initial_instance_offsets: Mapping[str, int] | None,
) -> dict[str, int]:
    if initial_instance_offsets is None:
        return {family: 0 for family in D1_SOURCE_FAMILIES}
    if set(initial_instance_offsets) != set(D1_SOURCE_FAMILIES):
        raise ValueError("instance offsets must match source families")
    offsets = {
        family: initial_instance_offsets[family] for family in D1_SOURCE_FAMILIES
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in offsets.values()
    ):
        raise ValueError("instance offsets must be non-negative integers")
    return offsets


def _episode_generator(seed: int, global_episode: int) -> torch.Generator:
    payload = f"residual-source-ppo-v1:{seed}:{global_episode}".encode("utf-8")
    numeric_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return torch.Generator().manual_seed(numeric_seed)


def _discounted_returns(
    rewards: Sequence[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + _RETURN_DISCOUNT * running
        values.append(running)
    return torch.tensor(
        tuple(reversed(values)),
        dtype=dtype,
        device=device,
    )


def _updated_weights(
    weights_before: Mapping[str, float],
    family_returns: Mapping[str, Sequence[float]],
) -> tuple[dict[str, float], dict[str, float]]:
    losses = {
        family: -statistics.fmean(returns)
        for family, returns in family_returns.items()
        if returns
    }
    observed = tuple(family for family in D1_SOURCE_FAMILIES if family in losses)
    if not observed:
        return dict(weights_before), losses
    observed_mass = sum(weights_before[family] for family in observed)
    if observed_mass <= 0:
        equal = 1.0 / len(observed)
        observed_values = tuple(equal for _ in observed)
        observed_mass = 1.0
    else:
        observed_values = tuple(
            weights_before[family] / observed_mass for family in observed
        )
    observed_weights = FamilyRobustWeights(
        observed,
        values=observed_values,
    ).update({family: losses[family] for family in observed})
    weights_after = {
        family: (
            observed_weights[family] * observed_mass
            if family in observed_weights
            else weights_before[family]
        )
        for family in D1_SOURCE_FAMILIES
    }
    total = sum(weights_after.values())
    return (
        {family: weights_after[family] / total for family in D1_SOURCE_FAMILIES},
        losses,
    )


def _ppo_update_combined(
    policy: ResidualRankerPolicy,
    weighted_sequences: Sequence[tuple[_ResidualPPOSequence, float]],
    *,
    deadline_check: Callable[[], None] | None,
) -> dict[str, object]:
    if not weighted_sequences:
        raise ValueError("residual PPO requires at least one sequence")
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    weights = torch.tensor(
        tuple(weight for _, weight in weighted_sequences),
        dtype=dtype,
        device=device,
    )
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or float(weights.sum()) <= 0
    ):
        raise ValueError("residual PPO sequence weights are invalid")
    advantages = torch.cat(
        tuple(
            sequence.advantages.detach().flatten() for sequence, _ in weighted_sequences
        )
    )
    advantage_mean = advantages.mean()
    advantage_std = advantages.std(unbiased=False).clamp_min(1e-8)
    normalized_advantages = tuple(
        (sequence.advantages.detach() - advantage_mean) / advantage_std
        for sequence, _ in weighted_sequences
    )
    config = policy.backbone.config
    final_metrics: dict[str, object] = {}
    for _ in range(config.update_epochs):
        if deadline_check is not None:
            deadline_check()
        policy_terms: list[torch.Tensor] = []
        value_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        for (sequence, weight), sequence_advantages in zip(
            weighted_sequences,
            normalized_advantages,
        ):
            hidden = policy.initial_state()
            logits_steps: list[torch.Tensor] = []
            value_steps: list[torch.Tensor] = []
            for proposal_index, observation in enumerate(sequence.observations):
                combined, hidden = policy.combined_logits(
                    observation,
                    hidden,
                    prior_order=sequence.prior_order,
                    proposal_index=proposal_index,
                )
                logits_steps.append(combined)
                value_steps.append(policy.backbone.critic(hidden).squeeze(-1))
            logits = torch.stack(logits_steps)
            values = torch.stack(value_steps)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probabilities = distribution.log_prob(sequence.actions)
            ratios = (log_probabilities - sequence.old_log_probabilities.detach()).exp()
            unclipped = ratios * sequence_advantages
            clipped = (
                ratios.clamp(
                    1.0 - config.clip_ratio,
                    1.0 + config.clip_ratio,
                )
                * sequence_advantages
            )
            policy_terms.append(-torch.minimum(unclipped, clipped).mean() * weight)
            value_terms.append(
                nn.functional.mse_loss(values, sequence.returns) * weight
            )
            entropy_terms.append(distribution.entropy().mean() * weight)
        policy_loss = torch.stack(policy_terms).sum()
        value_loss = torch.stack(value_terms).sum()
        entropy = torch.stack(entropy_terms).sum()
        loss = (
            policy_loss
            + config.value_weight * value_loss
            - config.entropy_weight * entropy
        )
        policy.backbone.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            policy.backbone.parameters(),
            config.gradient_clip_norm,
        )
        if deadline_check is not None:
            deadline_check()
        policy.backbone.optimizer.step()
        if deadline_check is not None:
            deadline_check()
        final_metrics = {
            "objective": "clipped_prior_plus_residual_actor_critic",
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "update_epochs": config.update_epochs,
            "clip_ratio": config.clip_ratio,
            "value_weight": config.value_weight,
            "entropy_weight": config.entropy_weight,
            "advantage_mean": float(advantage_mean.detach()),
            "advantage_std": float(advantage_std.detach()),
        }
    return final_metrics


def train_residual_ranker_ppo(
    policy: ResidualRankerPolicy,
    source_victims: Mapping[
        str,
        tuple[str, nn.Module] | Sequence[tuple[str, nn.Module]],
    ],
    source_samples: Sequence[tuple[torch.Tensor, int]],
    config: AttackConfig,
    *,
    episodes: int,
    seed: int,
    prior_seed: int,
    initial_family_weights: Mapping[str, float] | None = None,
    episode_offset: int = 0,
    initial_instance_offsets: Mapping[str, int] | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Run one resumable source-only block and update prior-plus-residual PPO.

    The budget includes initialization. Rollouts and PPO replay use the matched
    rollback operator and the same rotating combined-logit prior cursor.
    """
    _validate_controls(
        policy,
        config,
        episodes=episodes,
        seed=seed,
        prior_seed=prior_seed,
        episode_offset=episode_offset,
        deadline_check=deadline_check,
    )
    victims = _source_instances(source_victims)
    samples = _validated_samples(source_samples, config)
    weights_before = _initial_weights(initial_family_weights)
    starting_offsets = _starting_offsets(initial_instance_offsets)
    full_schedule = balanced_family_schedule(
        D1_SOURCE_FAMILIES,
        episode_offset + episodes,
        seed,
    )
    schedule = full_schedule[episode_offset:]
    parameter = next(policy.parameters())
    device = parameter.device
    dtype = parameter.dtype
    catalog = patch_catalog(config.grid_size, channels=3)
    family_episode_counts = {family: 0 for family in D1_SOURCE_FAMILIES}
    family_eligible = {family: 0 for family in D1_SOURCE_FAMILIES}
    family_successes = {family: 0 for family in D1_SOURCE_FAMILIES}
    family_returns: dict[str, list[float]] = {
        family: [] for family in D1_SOURCE_FAMILIES
    }
    family_margins: dict[str, list[float]] = {
        family: [] for family in D1_SOURCE_FAMILIES
    }
    source_calls_by_family = {family: 0 for family in D1_SOURCE_FAMILIES}
    victim_ids = tuple(
        victim_id for family in D1_SOURCE_FAMILIES for victim_id, _ in victims[family]
    )
    source_calls_by_victim = {victim_id: 0 for victim_id in victim_ids}
    instance_episode_counts = {victim_id: 0 for victim_id in victim_ids}
    instance_eligible = {victim_id: 0 for victim_id in victim_ids}
    instance_successes = {victim_id: 0 for victim_id in victim_ids}
    instance_returns: dict[str, list[float]] = {
        victim_id: [] for victim_id in victim_ids
    }
    instance_margins: dict[str, list[float]] = {
        victim_id: [] for victim_id in victim_ids
    }
    sample_indices: list[int] = []
    sequences: list[tuple[str, _ResidualPPOSequence]] = []
    success_values: list[float] = []
    margin_values: list[float] = []
    return_values: list[float] = []
    for local_episode, family in enumerate(schedule):
        if deadline_check is not None:
            deadline_check()
        instances = victims[family]
        instance_index = (
            starting_offsets[family] + family_episode_counts[family]
        ) % len(instances)
        family_episode_counts[family] += 1
        victim_id, victim = instances[instance_index]
        instance_episode_counts[victim_id] += 1
        global_episode = episode_offset + local_episode
        sample_index = global_episode % len(samples)
        sample_indices.append(sample_index)
        image, label = samples[sample_index]
        sample_id = (
            f"d1-source-ppo:{global_episode}:{family}:{victim_id}:{sample_index}"
        )
        prior_order = score_greedy_action_order(
            action_dim=policy.action_dim,
            seed=prior_seed,
            sample_id=sample_id,
        )
        oracle = AuditedVictim(
            victim,
            RESIDUAL_PPO_QUERY_BUDGET,
            "scores",
            victim_id,
        )
        original = (
            image.detach()
            .clone()
            .to(
                device=device,
                dtype=dtype,
            )
            .clamp(0.0, 1.0)
        )
        accepted = original.clone()
        current = oracle.query(
            accepted,
            sample_id,
            "source-ppo-initialization",
            0,
        )
        if current.predicted_label != label:
            source_calls_by_family[family] += oracle.calls
            source_calls_by_victim[victim_id] += oracle.calls
            continue
        clean_scores = current.scores.clone()
        clean_margin = score_margin(clean_scores, label)
        current_margin = clean_margin
        family_eligible[family] += 1
        instance_eligible[victim_id] += 1
        hidden = policy.initial_state()
        previous_action: int | None = None
        previous_reward = 0.0
        action_counts = np.zeros(config.action_dim, dtype=np.float32)
        action_values = np.zeros(config.action_dim, dtype=np.float32)
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        old_logs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        success = False
        generator = _episode_generator(seed, global_episode)
        while oracle.calls < RESIDUAL_PPO_QUERY_BUDGET and not success:
            if deadline_check is not None:
                deadline_check()
            observation = calibration_resistant_observation(
                current.scores,
                label,
                clean_scores,
                (RESIDUAL_PPO_QUERY_BUDGET - oracle.calls) / RESIDUAL_PPO_QUERY_BUDGET,
                previous_action,
                config.action_dim,
                previous_reward,
                oracle.calls / RESIDUAL_PPO_QUERY_BUDGET,
                action_counts if config.action_history_features else None,
                action_values if config.action_history_features else None,
                configured_patch_image_features(original, accepted, config),
            )
            observation_tensor = torch.as_tensor(
                observation,
                dtype=dtype,
                device=device,
            )
            proposal_index = len(actions)
            with torch.no_grad():
                combined, next_hidden = policy.combined_logits(
                    observation_tensor,
                    hidden,
                    prior_order=prior_order,
                    proposal_index=proposal_index,
                )
                value = policy.backbone.critic(next_hidden).squeeze(-1)
                distribution = torch.distributions.Categorical(logits=combined)
                sampled_action = torch.multinomial(
                    distribution.probs.detach().cpu(),
                    1,
                    generator=generator,
                ).squeeze(0)
                action = sampled_action.to(device=device)
                old_log = distribution.log_prob(action)
            action_index = int(action)
            proposal = apply_action(
                accepted,
                original,
                catalog[action_index],
                config.epsilon,
                config.step_size,
                config.grid_size,
            )
            candidate = oracle.query(
                proposal,
                sample_id,
                "source-residual-ppo",
                proposal_index + 1,
            )
            success = candidate.predicted_label != label
            candidate_margin = score_margin(candidate.scores, label)
            reward = recurrent_attack_reward(
                current.scores,
                candidate.scores,
                label,
                success,
                config,
            )
            transition = choose_attack_transition(
                accepted,
                proposal,
                current_margin=current_margin,
                proposal_margin=candidate_margin,
                success=success,
                rollback_on_non_improvement=True,
            )
            action_counts[action_index] += 1.0
            action_values[action_index] += (
                reward - action_values[action_index]
            ) / action_counts[action_index]
            observations.append(observation_tensor)
            actions.append(action.detach())
            old_logs.append(old_log.detach())
            values.append(value.detach())
            rewards.append(reward)
            hidden = next_hidden.detach()
            previous_action = action_index
            previous_reward = reward
            if transition.accepted:
                accepted = transition.image
                current = candidate
                current_margin = candidate_margin
        source_calls_by_family[family] += oracle.calls
        source_calls_by_victim[victim_id] += oracle.calls
        margin_reduction = clean_margin - current_margin
        episode_return = sum(rewards)
        family_returns[family].append(episode_return)
        family_margins[family].append(margin_reduction)
        instance_returns[victim_id].append(episode_return)
        instance_margins[victim_id].append(margin_reduction)
        success_values.append(float(success))
        margin_values.append(margin_reduction)
        return_values.append(episode_return)
        if success:
            family_successes[family] += 1
            instance_successes[victim_id] += 1
        if rewards:
            returns = _discounted_returns(
                rewards,
                device=device,
                dtype=dtype,
            )
            value_tensor = torch.stack(values)
            sequences.append(
                (
                    family,
                    _ResidualPPOSequence(
                        observations=torch.stack(observations),
                        actions=torch.stack(actions),
                        old_log_probabilities=torch.stack(old_logs),
                        advantages=returns - value_tensor,
                        returns=returns,
                        prior_order=prior_order,
                    ),
                )
            )
    weights_after, family_losses = _updated_weights(
        weights_before,
        family_returns,
    )
    source_calls = sum(source_calls_by_family.values())
    eligible_episodes = sum(family_eligible.values())
    successful_episodes = sum(family_successes.values())
    family_diagnostics = {
        family: {
            "scheduled_episodes": family_episode_counts[family],
            "eligible_episodes": family_eligible[family],
            "successful_episodes": family_successes[family],
            "success_rate": family_successes[family] / family_eligible[family]
            if family_eligible[family]
            else 0.0,
            "source_calls": source_calls_by_family[family],
            "episode_return": _distribution_summary(family_returns[family]),
            "margin_reduction": _distribution_summary(family_margins[family]),
            "groupdro_loss": family_losses.get(family),
            "weight_before": weights_before[family],
            "weight_after": weights_after[family],
        }
        for family in D1_SOURCE_FAMILIES
    }
    instance_diagnostics = {
        victim_id: {
            "scheduled_episodes": instance_episode_counts[victim_id],
            "eligible_episodes": instance_eligible[victim_id],
            "successful_episodes": instance_successes[victim_id],
            "success_rate": instance_successes[victim_id] / instance_eligible[victim_id]
            if instance_eligible[victim_id]
            else 0.0,
            "source_calls": source_calls_by_victim[victim_id],
            "episode_return": _distribution_summary(instance_returns[victim_id]),
            "margin_reduction": _distribution_summary(instance_margins[victim_id]),
        }
        for victim_id in victim_ids
    }
    shared_metrics: dict[str, object] = {
        "episodes": episodes,
        "trained_episodes": len(sequences),
        "eligible_episodes": eligible_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": (
            successful_episodes / eligible_episodes if eligible_episodes else 0.0
        ),
        "success": _distribution_summary(success_values),
        "margin_reduction": _distribution_summary(margin_values),
        "episode_return": _distribution_summary(return_values),
        "source_calls": source_calls,
        "source_calls_by_family": source_calls_by_family,
        "source_calls_by_victim": source_calls_by_victim,
        "hidden_target_calls": 0,
        "sample_indices": sample_indices,
        "unique_sample_count": len(set(sample_indices)),
        "episode_offset": episode_offset,
        "next_episode_offset": episode_offset + episodes,
        "instance_offsets": {
            family: (starting_offsets[family] + family_episode_counts[family])
            for family in D1_SOURCE_FAMILIES
        },
        "schedule": schedule,
        "family_weights": weights_after,
        "family_diagnostics": family_diagnostics,
        "instance_diagnostics": instance_diagnostics,
    }
    if not sequences:
        return shared_metrics
    sequence_counts = {
        family: sum(sequence_family == family for sequence_family, _ in sequences)
        for family in D1_SOURCE_FAMILIES
    }
    observed = tuple(family for family in D1_SOURCE_FAMILIES if sequence_counts[family])
    training_mass = sum(weights_after[family] for family in observed)
    if training_mass <= 0:
        training_family_weights = {family: 1.0 / len(observed) for family in observed}
    else:
        training_family_weights = {
            family: weights_after[family] / training_mass for family in observed
        }
    ppo = _ppo_update_combined(
        policy,
        tuple(
            (
                sequence,
                (training_family_weights[family] / sequence_counts[family]),
            )
            for family, sequence in sequences
        ),
        deadline_check=deadline_check,
    )
    return {**shared_metrics, "ppo": ppo}


refine_residual_ranker_ppo = train_residual_ranker_ppo


__all__ = (
    "RESIDUAL_PPO_BLOCK_EPISODES",
    "RESIDUAL_PPO_QUERY_BUDGET",
    "refine_residual_ranker_ppo",
    "train_residual_ranker_ppo",
)
