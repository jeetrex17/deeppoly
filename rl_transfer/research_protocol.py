from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .audit import AuditedVictim
from .actions import apply_action, patch_catalog
from .config import AttackConfig
from .population import FamilyRobustWeights, balanced_family_schedule
from .recurrent import PPOSequence, RecurrentAttackPolicy


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


def calibration_resistant_observation(scores: torch.Tensor, label: int, initial_scores: torch.Tensor, remaining: float, previous_action: int | None, action_dim: int, previous_reward: float, step_fraction: float) -> np.ndarray:
    rank = int((scores > scores[label]).sum())
    normalized_rank = rank / max(1, scores.numel() - 1)
    entropy = float(-(scores.clamp_min(1e-12) * scores.clamp_min(1e-12).log()).sum() / math.log(scores.numel()))
    delta = float((scores[label] - initial_scores[label]) / initial_scores[label].abs().clamp_min(1e-6))
    rival = float(torch.cat((scores[:label], scores[label + 1:])).max())
    action_feature = -1.0 if previous_action is None else previous_action / max(1, action_dim - 1)
    return np.asarray((normalized_rank, entropy, delta, float(scores[label]) - rival, remaining, action_feature, math.tanh(previous_reward), step_fraction), dtype=np.float32)


def _validated_catalog(policy: RecurrentAttackPolicy, config: AttackConfig, channels: int):
    catalog = patch_catalog(config.grid_size, channels)
    if len(catalog) != config.action_dim or policy.action_dim != len(catalog):
        raise ValueError("policy, config, and action catalog dimensions must match")
    return catalog


def run_frozen_episode(policy: RecurrentAttackPolicy, victim: nn.Module, image: torch.Tensor, label: int, sample_id: str, victim_id: str, family: str, config: AttackConfig, deterministic: bool = True) -> FrozenEpisodeResult:
    if config.max_queries < 1:
        raise ValueError("at least one query is required for T1")
    before = policy.persistent_digest()
    oracle = AuditedVictim(victim, config.max_queries, "scores", victim_id)
    original = image.detach().clone().float().clamp(0, 1)
    adversarial = original.clone()
    initial = oracle.query(adversarial, sample_id, "initialization", 0)
    clean_scores = initial.scores.clone()
    clean_correct = initial.predicted_label == label
    if not clean_correct:
        return FrozenEpisodeResult(sample_id, victim_id, family, False, False, None, oracle.calls, 0.0, 0.0, (), before, policy.persistent_digest(), tuple(oracle.trace_dicts()))
    hidden = policy.initial_state()
    previous_action, previous_reward = None, 0.0
    actions: list[int] = []
    catalog = _validated_catalog(policy, config, image.shape[0])
    success, success_query = False, None
    while oracle.calls < config.max_queries and not success:
        observation = calibration_resistant_observation(initial.scores, label, clean_scores, (config.max_queries - oracle.calls) / config.max_queries, previous_action, config.action_dim, previous_reward, oracle.calls / config.max_queries)
        action, hidden = policy.act(observation, hidden, deterministic=deterministic)
        adversarial = apply_action(adversarial, original, catalog[action], config.epsilon, config.step_size, config.grid_size)
        response = oracle.query(adversarial, sample_id, "attack", len(actions) + 1)
        actions.append(action)
        success = response.predicted_label != label
        previous_reward = 10.0 if success else float(initial.scores[label] - response.scores[label]) - 0.05
        previous_action = action
        initial = response
        if success:
            success_query = oracle.calls
    delta = adversarial - original
    after = policy.persistent_digest()
    if before != after:
        raise RuntimeError("frozen deployment mutated persistent policy state")
    return FrozenEpisodeResult(sample_id, victim_id, family, True, success, success_query, oracle.calls, float(delta.abs().max()), float(delta.flatten().norm()), tuple(actions), before, after, tuple(oracle.trace_dicts()))


def train_population_policy(policy: RecurrentAttackPolicy, victims: Mapping[str, tuple[str, nn.Module]], samples: Sequence[tuple[torch.Tensor, int]], config: AttackConfig, episodes: int, seed: int) -> dict[str, object]:
    if not victims or not samples:
        raise ValueError("victims and samples are required")
    schedule = balanced_family_schedule(tuple(victims), episodes, seed)
    family_rewards = {family: [] for family in victims}
    sequences: list[tuple[str, PPOSequence]] = []
    generator = torch.Generator().manual_seed(seed)
    for episode, family in enumerate(schedule):
        victim_id, victim = victims[family]
        image, label = samples[episode % len(samples)]
        oracle = AuditedVictim(victim, config.max_queries, "scores", victim_id)
        original, adversarial = image.clone(), image.clone()
        initial = oracle.query(adversarial, f"train-{episode}", "initialization", 0)
        clean_scores = initial.scores.clone()
        if initial.predicted_label != label:
            continue
        hidden = policy.initial_state()
        catalog = _validated_catalog(policy, config, image.shape[0])
        observations, actions, old_logs, rewards, values = [], [], [], [], []
        previous_action, previous_reward = None, 0.0
        while oracle.calls < config.max_queries:
            observation = calibration_resistant_observation(initial.scores, label, clean_scores, (config.max_queries - oracle.calls) / config.max_queries, previous_action, config.action_dim, previous_reward, oracle.calls / config.max_queries)
            observation_tensor = torch.as_tensor(observation)
            with torch.no_grad():
                logits, value, next_hidden = policy(observation_tensor, hidden)
                distribution = torch.distributions.Categorical(logits=logits)
                action = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(0)
            adversarial = apply_action(adversarial, original, catalog[int(action)], config.epsilon, config.step_size, config.grid_size)
            response = oracle.query(adversarial, f"train-{episode}", "attack", oracle.calls)
            success = response.predicted_label != label
            reward = 10.0 if success else float(initial.scores[label] - response.scores[label]) - 0.05
            observations.append(observation_tensor); actions.append(action); old_logs.append(distribution.log_prob(action)); values.append(value)
            rewards.append(reward)
            hidden, initial, previous_action, previous_reward = next_hidden.detach(), response, int(action), reward
            if success:
                break
        if not rewards:
            continue
        returns = []; running = 0.0
        for reward in reversed(rewards):
            running = reward + 0.98 * running
            returns.append(running)
        returns_tensor = torch.tensor(tuple(reversed(returns)), dtype=torch.float32)
        values_tensor = torch.stack(values)
        advantages = returns_tensor - values_tensor
        sequences.append((family, PPOSequence(torch.stack(observations), torch.stack(actions), torch.stack(old_logs), advantages, returns_tensor)))
        family_rewards[family].append(sum(rewards))
    if not sequences:
        return {"episodes": episodes, "trained_episodes": 0, "schedule": schedule, "family_weights": {family: 1 / len(victims) for family in victims}}
    losses = {family: -sum(values) / len(values) if values else 0.0 for family, values in family_rewards.items()}
    weights = FamilyRobustWeights(tuple(victims)).update(losses)
    sequence_counts = {family: sum(item_family == family for item_family, _ in sequences) for family in victims}
    metrics = policy.ppo_update_sequences([(sequence, weights[family] / sequence_counts[family]) for family, sequence in sequences])
    return {"episodes": episodes, "trained_episodes": len(sequences), "schedule": schedule, "family_weights": weights, "ppo": metrics}
