from dataclasses import asdict, dataclass
import hashlib
import math
import random
import statistics
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .audit import AuditedVictim
from .actions import apply_action, patch_catalog
from .config import AttackConfig
from .features import configured_patch_image_features
from .population import FamilyRobustWeights, balanced_family_schedule
from .operator import choose_attack_transition
from .recurrent import PPOSequence, RecurrentAttackPolicy
from .rewards import recurrent_attack_reward, score_margin


@dataclass(frozen=True)
class FrozenEpisodeResult:
    sample_id: str
    victim_id: str
    family: str
    clean_correct: bool
    success: bool
    query_to_success: int | None
    total_target_calls: int
    linf: float
    l2: float
    actions: tuple[int, ...]
    policy_digest_before: str
    policy_digest_after: str
    query_trace: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _distribution_summary(values: Sequence[float]) -> dict[str, object]:
    numeric = tuple(float(value) for value in values)
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric) if numeric else None,
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0 if numeric else None,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
    }


def calibration_resistant_observation(
    scores: torch.Tensor,
    label: int,
    initial_scores: torch.Tensor,
    remaining: float,
    previous_action: int | None,
    action_dim: int,
    previous_reward: float,
    step_fraction: float,
    action_counts: np.ndarray | None = None,
    action_values: np.ndarray | None = None,
    image_features: np.ndarray | None = None,
) -> np.ndarray:
    rank = int((scores > scores[label]).sum())
    normalized_rank = rank / max(1, scores.numel() - 1)
    entropy = float(-(scores.clamp_min(1e-12) * scores.clamp_min(1e-12).log()).sum() / math.log(scores.numel()))
    delta = float((scores[label] - initial_scores[label]) / initial_scores[label].abs().clamp_min(1e-6))
    rival = float(torch.cat((scores[:label], scores[label + 1:])).max())
    action_feature = -1.0 if previous_action is None else previous_action / max(1, action_dim - 1)
    base = np.asarray(
        (
            normalized_rank,
            entropy,
            delta,
            float(scores[label]) - rival,
            remaining,
            action_feature,
            math.tanh(previous_reward),
            step_fraction,
        ),
        dtype=np.float32,
    )
    blocks = [base]
    if action_counts is not None or action_values is not None:
        if action_counts is None or action_values is None:
            raise ValueError("action counts and values must be supplied together")
        counts = np.asarray(action_counts, dtype=np.float32)
        values = np.asarray(action_values, dtype=np.float32)
        if (
            counts.shape != (action_dim,)
            or values.shape != (action_dim,)
            or not np.isfinite(counts).all()
            or not np.isfinite(values).all()
            or (counts < 0).any()
        ):
            raise ValueError("action history features do not match the action catalog")
        normalized_counts = counts / max(1.0, float(counts.max()))
        blocks.extend((normalized_counts, np.tanh(values)))
    if image_features is not None:
        numeric_image_features = np.asarray(image_features, dtype=np.float32)
        if (
            numeric_image_features.ndim != 1
            or not numeric_image_features.size
            or not np.isfinite(numeric_image_features).all()
        ):
            raise ValueError("image features must be a finite one-dimensional vector")
        blocks.append(numeric_image_features)
    return np.concatenate(blocks).astype(np.float32, copy=False)


def _validated_catalog(policy: RecurrentAttackPolicy, config: AttackConfig, channels: int):
    catalog = patch_catalog(config.grid_size, channels)
    if len(catalog) != config.action_dim or policy.action_dim != len(catalog):
        raise ValueError("policy, config, and action catalog dimensions must match")
    return catalog


def run_frozen_episode(
    policy,
    victim: nn.Module,
    image: torch.Tensor,
    label: int,
    sample_id: str,
    victim_id: str,
    family: str,
    config: AttackConfig,
    deterministic: bool = True,
    episode_seed: int | None = None,
) -> FrozenEpisodeResult:
    if config.max_queries < 1:
        raise ValueError("at least one query is required for T1")
    device = (
        next(policy.parameters()).device
        if isinstance(policy, nn.Module)
        else next(victim.parameters()).device
    )
    before = policy.persistent_digest()
    oracle = AuditedVictim(victim, config.max_queries, "scores", victim_id)
    original = image.detach().clone().float().clamp(0, 1).to(device)
    adversarial = original.clone()
    initial = oracle.query(adversarial, sample_id, "initialization", 0)
    clean_scores = initial.scores.clone()
    clean_correct = initial.predicted_label == label
    if not clean_correct:
        return FrozenEpisodeResult(sample_id, victim_id, family, False, False, None, oracle.calls, 0.0, 0.0, (), before, policy.persistent_digest(), tuple(oracle.trace_dicts()))
    hidden = policy.initial_state()
    stochastic_seed = (
        episode_seed
        if episode_seed is not None
        else int.from_bytes(
            hashlib.sha256(
                f"frozen-policy-v1:{victim_id}:{sample_id}".encode()
            ).digest()[:8],
            "big",
        )
    )
    action_rng = random.Random(stochastic_seed)
    previous_action, previous_reward = None, 0.0
    action_counts = np.zeros(config.action_dim, dtype=np.float32)
    action_values = np.zeros(config.action_dim, dtype=np.float32)
    actions: list[int] = []
    catalog = _validated_catalog(policy, config, image.shape[0])
    success, success_query = False, None
    while oracle.calls < config.max_queries and not success:
        observation = calibration_resistant_observation(
            initial.scores,
            label,
            clean_scores,
            (config.max_queries - oracle.calls) / config.max_queries,
            previous_action,
            config.action_dim,
            previous_reward,
            oracle.calls / config.max_queries,
            action_counts if config.action_history_features else None,
            action_values if config.action_history_features else None,
            configured_patch_image_features(
                original,
                adversarial,
                config,
            ),
        )
        if isinstance(policy, RecurrentAttackPolicy):
            action, hidden = policy.act(
                observation,
                hidden,
                deterministic=deterministic,
                random_draw=(
                    None if deterministic else action_rng.random()
                ),
            )
        else:
            action, hidden = policy.act(
                observation,
                hidden,
                deterministic=deterministic,
            )
        proposal = apply_action(
            adversarial,
            original,
            catalog[action],
            config.epsilon,
            config.step_size,
            config.grid_size,
        )
        response = oracle.query(proposal, sample_id, "attack", len(actions) + 1)
        actions.append(action)
        success = response.predicted_label != label
        previous_reward = recurrent_attack_reward(
            initial.scores,
            response.scores,
            label,
            success,
            config,
        )
        current_margin = score_margin(initial.scores, label)
        proposal_margin = score_margin(response.scores, label)
        transition = choose_attack_transition(
            adversarial,
            proposal,
            current_margin=current_margin,
            proposal_margin=proposal_margin,
            success=success,
            rollback_on_non_improvement=config.rollback_on_non_improvement,
        )
        adversarial = transition.image
        action_counts[action] += 1.0
        action_values[action] += (
            previous_reward - action_values[action]
        ) / action_counts[action]
        previous_action = action
        if transition.accepted:
            initial = response
        if success:
            success_query = oracle.calls
    delta = adversarial - original
    after = policy.persistent_digest()
    if before != after:
        raise RuntimeError("frozen deployment mutated persistent policy state")
    return FrozenEpisodeResult(sample_id, victim_id, family, True, success, success_query, oracle.calls, float(delta.abs().max()), float(delta.flatten().norm()), tuple(actions), before, after, tuple(oracle.trace_dicts()))


def run_score_greedy_episode(
    victim: nn.Module,
    image: torch.Tensor,
    label: int,
    sample_id: str,
    victim_id: str,
    family: str,
    config: AttackConfig,
    seed: int,
) -> FrozenEpisodeResult:
    """Query-matched SimBA-style score attack with accept/reject proposals.

    Each proposal changes one channel/patch directly to the L-infinity boundary,
    then is retained only when it reduces the true-label confidence margin. The
    initialization query is included in the same total target-query budget.
    """

    parameter = next(victim.parameters(), None)
    buffer = next(victim.buffers(), None)
    device = (
        parameter.device
        if parameter is not None
        else buffer.device
        if buffer is not None
        else image.device
    )
    digest_payload = (
        f"score-greedy-v2:{seed}:{config.grid_size}:{config.max_queries}:"
        f"{config.epsilon:.12g}:{config.step_size:.12g}:"
        f"{config.rollback_on_non_improvement}"
    )
    digest = hashlib.sha256(digest_payload.encode()).hexdigest()
    oracle = AuditedVictim(victim, config.max_queries, "scores", victim_id)
    original = image.detach().clone().float().clamp(0, 1).to(device)
    accepted = original.clone()
    response = oracle.query(accepted, sample_id, "initialization", 0)
    if response.predicted_label != label:
        return FrozenEpisodeResult(
            sample_id,
            victim_id,
            family,
            False,
            False,
            None,
            oracle.calls,
            0.0,
            0.0,
            (),
            digest,
            digest,
            tuple(oracle.trace_dicts()),
        )
    accepted_margin = score_margin(response.scores, label)
    catalog = list(enumerate(patch_catalog(config.grid_size, image.shape[0])))
    episode_seed = seed ^ int.from_bytes(
        hashlib.sha256(sample_id.encode()).digest()[:8], "big"
    )
    random.Random(episode_seed).shuffle(catalog)
    actions: list[int] = []
    success = False
    success_query: int | None = None
    proposal_index = 0
    while oracle.calls < config.max_queries and not success:
        action_index, action = catalog[proposal_index % len(catalog)]
        proposal = apply_action(
            accepted,
            original,
            action,
            config.epsilon,
            (
                config.step_size
                if config.rollback_on_non_improvement
                else config.epsilon
            ),
            config.grid_size,
        )
        candidate = oracle.query(
            proposal,
            sample_id,
            "score-greedy-proposal",
            proposal_index + 1,
        )
        actions.append(action_index)
        success = candidate.predicted_label != label
        candidate_margin = score_margin(candidate.scores, label)
        transition = choose_attack_transition(
            accepted,
            proposal,
            current_margin=accepted_margin,
            proposal_margin=candidate_margin,
            success=success,
            rollback_on_non_improvement=True,
        )
        if transition.accepted:
            accepted = transition.image
            accepted_margin = candidate_margin
        if success:
            success_query = oracle.calls
        proposal_index += 1
    delta = accepted - original
    return FrozenEpisodeResult(
        sample_id,
        victim_id,
        family,
        True,
        success,
        success_query,
        oracle.calls,
        float(delta.abs().max()),
        float(delta.flatten().norm()),
        tuple(actions),
        digest,
        digest,
        tuple(oracle.trace_dicts()),
    )


def train_population_policy(
    policy: RecurrentAttackPolicy,
    victims: Mapping[
        str,
        tuple[str, nn.Module] | Sequence[tuple[str, nn.Module]],
    ],
    samples: Sequence[tuple[torch.Tensor, int]],
    config: AttackConfig,
    episodes: int,
    seed: int,
    initial_family_weights: Mapping[str, float] | None = None,
    episode_offset: int = 0,
    initial_instance_offsets: Mapping[str, int] | None = None,
) -> dict[str, object]:
    if not victims or not samples:
        raise ValueError("victims and samples are required")
    if initial_family_weights is not None and set(initial_family_weights) != set(victims):
        raise ValueError("initial family weights must match the victim families")
    if episode_offset < 0:
        raise ValueError("episode_offset cannot be negative")
    if initial_instance_offsets is not None and set(initial_instance_offsets) != set(victims):
        raise ValueError("initial instance offsets must match the victim families")
    victim_instances: dict[str, tuple[tuple[str, nn.Module], ...]] = {}
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
            raise ValueError("each source family requires one or more named victim modules")
        victim_instances[family] = instances
    device = next(policy.parameters()).device
    schedule = balanced_family_schedule(tuple(victims), episodes, seed)
    family_rewards = {family: [] for family in victims}
    family_margin_reductions = {family: [] for family in victims}
    family_eligible = {family: 0 for family in victims}
    family_successes = {family: 0 for family in victims}
    instance_rewards = {
        victim_id: []
        for instances in victim_instances.values()
        for victim_id, _ in instances
    }
    instance_margin_reductions = {
        victim_id: []
        for instances in victim_instances.values()
        for victim_id, _ in instances
    }
    instance_eligible = {victim_id: 0 for victim_id in instance_rewards}
    instance_successes = {victim_id: 0 for victim_id in instance_rewards}
    instance_episode_counts = {victim_id: 0 for victim_id in instance_rewards}
    sequences: list[tuple[str, PPOSequence]] = []
    source_calls = 0
    source_calls_by_family = {family: 0 for family in victims}
    source_calls_by_victim = {
        victim_id: 0
        for instances in victim_instances.values()
        for victim_id, _ in instances
    }
    family_episode_counts = {family: 0 for family in victims}
    starting_instance_offsets = {
        family: (
            int(initial_instance_offsets[family])
            if initial_instance_offsets is not None
            else 0
        )
        for family in victims
    }
    if any(offset < 0 for offset in starting_instance_offsets.values()):
        raise ValueError("instance offsets cannot be negative")
    sample_indices: list[int] = []
    generator = torch.Generator().manual_seed(seed)
    for episode, family in enumerate(schedule):
        instances = victim_instances[family]
        instance_index = (
            starting_instance_offsets[family] + family_episode_counts[family]
        ) % len(instances)
        family_episode_counts[family] += 1
        victim_id, victim = instances[instance_index]
        instance_episode_counts[victim_id] += 1
        global_episode = episode_offset + episode
        sample_index = global_episode % len(samples)
        sample_indices.append(sample_index)
        image, label = samples[sample_index]
        oracle = AuditedVictim(victim, config.max_queries, "scores", victim_id)
        original = image.detach().clone().float().clamp(0, 1).to(device)
        adversarial = original.clone()
        sample_id = f"train-{global_episode}"
        initial = oracle.query(adversarial, sample_id, "initialization", 0)
        clean_scores = initial.scores.clone()
        if initial.predicted_label != label:
            source_calls += oracle.calls
            source_calls_by_family[family] += oracle.calls
            source_calls_by_victim[victim_id] += oracle.calls
            continue
        family_eligible[family] += 1
        instance_eligible[victim_id] += 1
        clean_margin = score_margin(clean_scores, label)
        hidden = policy.initial_state()
        catalog = _validated_catalog(policy, config, image.shape[0])
        observations, actions, old_logs, rewards, values = [], [], [], [], []
        previous_action, previous_reward = None, 0.0
        action_counts = np.zeros(config.action_dim, dtype=np.float32)
        action_values = np.zeros(config.action_dim, dtype=np.float32)
        success = False
        while oracle.calls < config.max_queries:
            observation = calibration_resistant_observation(
                initial.scores,
                label,
                clean_scores,
                (config.max_queries - oracle.calls) / config.max_queries,
                previous_action,
                config.action_dim,
                previous_reward,
                oracle.calls / config.max_queries,
                action_counts if config.action_history_features else None,
                action_values if config.action_history_features else None,
                configured_patch_image_features(
                    original,
                    adversarial,
                    config,
                ),
            )
            observation_tensor = torch.as_tensor(observation, device=device)
            with torch.no_grad():
                logits, value, next_hidden = policy(observation_tensor, hidden)
                distribution = torch.distributions.Categorical(logits=logits)
                sampled_action = torch.multinomial(
                    distribution.probs.detach().cpu(),
                    1,
                    generator=generator,
                ).squeeze(0)
                action = sampled_action.to(device)
            action_index = int(action)
            proposal = apply_action(
                adversarial,
                original,
                catalog[action_index],
                config.epsilon,
                config.step_size,
                config.grid_size,
            )
            response = oracle.query(proposal, sample_id, "attack", oracle.calls)
            success = response.predicted_label != label
            reward = recurrent_attack_reward(
                initial.scores,
                response.scores,
                label,
                success,
                config,
            )
            transition = choose_attack_transition(
                adversarial,
                proposal,
                current_margin=score_margin(initial.scores, label),
                proposal_margin=score_margin(response.scores, label),
                success=success,
                rollback_on_non_improvement=config.rollback_on_non_improvement,
            )
            adversarial = transition.image
            action_counts[action_index] += 1.0
            action_values[action_index] += (
                reward - action_values[action_index]
            ) / action_counts[action_index]
            observations.append(observation_tensor)
            actions.append(action)
            old_logs.append(distribution.log_prob(action))
            values.append(value)
            rewards.append(reward)
            hidden = next_hidden.detach()
            if transition.accepted:
                initial = response
            previous_action, previous_reward = action_index, reward
            if success:
                break
        source_calls += oracle.calls
        source_calls_by_family[family] += oracle.calls
        source_calls_by_victim[victim_id] += oracle.calls
        margin_reduction = clean_margin - score_margin(initial.scores, label)
        family_margin_reductions[family].append(margin_reduction)
        instance_margin_reductions[victim_id].append(margin_reduction)
        if success:
            family_successes[family] += 1
            instance_successes[victim_id] += 1
        if not rewards:
            continue
        returns = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + 0.98 * running
            returns.append(running)
        returns_tensor = torch.tensor(tuple(reversed(returns)), dtype=torch.float32, device=device)
        values_tensor = torch.stack(values)
        advantages = returns_tensor - values_tensor
        sequences.append((family, PPOSequence(torch.stack(observations), torch.stack(actions), torch.stack(old_logs), advantages, returns_tensor)))
        episode_return = sum(rewards)
        family_rewards[family].append(episode_return)
        instance_rewards[victim_id].append(episode_return)
    weights_before = (
        {family: float(initial_family_weights[family]) for family in victims}
        if initial_family_weights is not None
        else {family: 1 / len(victims) for family in victims}
    )
    losses = {
        family: -statistics.fmean(values)
        for family, values in family_rewards.items()
        if values
    }
    observed_families = tuple(family for family in victims if family in losses)
    if observed_families:
        observed_mass = sum(weights_before[family] for family in observed_families)
        observed_weights = FamilyRobustWeights(
            observed_families,
            values=tuple(
                weights_before[family] / observed_mass
                for family in observed_families
            ),
        ).update(losses)
        weights = {
            family: (
                observed_weights[family] * observed_mass
                if family in observed_weights
                else weights_before[family]
            )
            for family in victims
        }
    else:
        weights = dict(weights_before)
    family_diagnostics = {
        family: {
            "scheduled_episodes": family_episode_counts[family],
            "eligible_episodes": family_eligible[family],
            "successful_episodes": family_successes[family],
            "source_calls": source_calls_by_family[family],
            "episode_return": _distribution_summary(family_rewards[family]),
            "margin_reduction": _distribution_summary(
                family_margin_reductions[family]
            ),
            "groupdro_loss": losses.get(family),
            "weight_before": weights_before[family],
            "weight_after": weights[family],
        }
        for family in victims
    }
    instance_diagnostics = {
        victim_id: {
            "scheduled_episodes": instance_episode_counts[victim_id],
            "eligible_episodes": instance_eligible[victim_id],
            "successful_episodes": instance_successes[victim_id],
            "source_calls": source_calls_by_victim[victim_id],
            "episode_return": _distribution_summary(instance_rewards[victim_id]),
            "margin_reduction": _distribution_summary(
                instance_margin_reductions[victim_id]
            ),
        }
        for victim_id in instance_rewards
    }
    shared_metrics = {
        "episodes": episodes,
        "source_calls": source_calls,
        "source_calls_by_family": source_calls_by_family,
        "source_calls_by_victim": source_calls_by_victim,
        "sample_indices": sample_indices,
        "unique_sample_count": len(set(sample_indices)),
        "episode_offset": episode_offset,
        "instance_offsets": {
            family: starting_instance_offsets[family] + family_episode_counts[family]
            for family in victims
        },
        "schedule": schedule,
        "family_weights": weights,
        "family_diagnostics": family_diagnostics,
        "instance_diagnostics": instance_diagnostics,
    }
    if not sequences:
        return {
            **shared_metrics,
            "trained_episodes": 0,
        }
    sequence_counts = {family: sum(item_family == family for item_family, _ in sequences) for family in victims}
    training_mass = sum(weights[family] for family in observed_families)
    metrics = policy.ppo_update_sequences(
        [
            (
                sequence,
                weights[family] / training_mass / sequence_counts[family],
            )
            for family, sequence in sequences
        ]
    )
    return {
        **shared_metrics,
        "trained_episodes": len(sequences),
        "ppo": metrics,
    }
