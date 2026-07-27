from typing import Iterable
from dataclasses import dataclass

import torch

from .config import AttackConfig
from .dqn import DQNAgent
from .environment import PatchAttackEnv
from .reproducibility import state_digest
from .metrics import EpisodeResult, aggregate_results


@dataclass(frozen=True)
class AttackSample:
    sample_id: str
    image: torch.Tensor
    label: int


@dataclass(frozen=True)
class TransferResult:
    metrics: object
    policy_digest_before: str
    policy_digest_after: str


def train_policy(agent: DQNAgent, victim: torch.nn.Module, samples: Iterable[tuple[torch.Tensor, int]], config: AttackConfig, episodes: int, seed: int = 0) -> dict[str, float]:
    rewards = []
    materialized = tuple(samples)
    if not materialized:
        raise ValueError("training samples cannot be empty")
    for episode in range(episodes):
        image, label = materialized[episode % len(materialized)]
        try:
            env = PatchAttackEnv(victim, image, label, config, seed=seed + episode)
        except Exception as error:
            from .environment import IneligibleSampleError
            if isinstance(error, IneligibleSampleError):
                continue
            raise
        state = env.reset()
        total = 0.0
        while not env.done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, done)
            agent.learn()
            state, total = next_state, total + reward
        rewards.append(total)
    return {"episodes": float(episodes), "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0}


def evaluate_policy(agent: DQNAgent, victim: torch.nn.Module, samples: Iterable[tuple[torch.Tensor, int]], config: AttackConfig, seed: int = 0) -> dict[str, float]:
    successes, eligible, queries, linf, l2 = 0, 0, [], [], []
    for index, (image, label) in enumerate(samples):
        try:
            env = PatchAttackEnv(victim, image, label, config, seed=seed + index)
        except Exception as error:
            from .environment import IneligibleSampleError
            if isinstance(error, IneligibleSampleError):
                continue
            raise
        state = env._state()
        if env.clean_prediction != label:
            continue
        eligible += 1
        while not env.done:
            state, _, _, info = env.step(agent.act(state, greedy=True))
        success = bool(info["success"])
        successes += int(success)
        if success:
            delta = env.adv - env.original
            queries.append(env.queries)
            linf.append(float(delta.abs().max()))
            l2.append(float(delta.flatten().norm()))
    return {
        "eligible": float(eligible),
        "attack_success_rate": float(successes / eligible) if eligible else float("nan"),
        "mean_success_queries": float(sum(queries) / len(queries)) if queries else float("nan"),
        "mean_linf": float(sum(linf) / len(linf)) if linf else float("nan"),
        "mean_l2": float(sum(l2) / len(l2)) if l2 else float("nan"),
    }


def _as_tuples(samples: Iterable[AttackSample]) -> tuple[tuple[torch.Tensor, int], ...]:
    return tuple((sample.image, sample.label) for sample in samples)


def run_frozen_transfer(agent: DQNAgent, victim: torch.nn.Module, samples: Iterable[AttackSample], config: AttackConfig) -> TransferResult:
    before = agent.policy_digest()
    rows = []
    for index, sample in enumerate(samples):
        try:
            env = PatchAttackEnv(victim, config)
            env.reset(sample.image, sample.label, sample.sample_id)
        except Exception as error:
            from .environment import IneligibleSampleError
            if isinstance(error, IneligibleSampleError):
                rows.append(EpisodeResult(sample.sample_id, False, False, 0, 0.0, 0.0, 0.0, 0))
                continue
            raise
        state = env._state()
        info = {"success": False}
        while not env.done:
            state, _, _, info = env.step(agent.act(state, evaluate=True))
        delta = env.adversarial_image - sample.image
        rows.append(EpisodeResult(sample.sample_id, True, info["success"], info["queries"] - 1, float(delta.abs().max()), float(delta.flatten().norm()), float(info.get("confidence_drop", 0.0)), int(env.touched.sum())))
    return TransferResult(aggregate_results(rows), before, agent.policy_digest())


def run_continual_transfer(agent: DQNAgent, victim: torch.nn.Module, adaptation_samples: Iterable[AttackSample], evaluation_samples: Iterable[AttackSample], config: AttackConfig, adaptation_epochs: int = 1) -> tuple[TransferResult, DQNAgent]:
    adapted = agent.clone()
    before = adapted.policy_digest()
    train_policy(adapted, victim, _as_tuples(adaptation_samples), config, adaptation_epochs)
    result = run_frozen_transfer(adapted, victim, evaluation_samples, config)
    return TransferResult(result.metrics, before, result.policy_digest_after), adapted


def run_transfer_protocols(source_policy: DQNAgent, source_victim: torch.nn.Module, target_victim: torch.nn.Module, source_samples: Iterable[tuple[torch.Tensor, int]], target_adaptation_samples: Iterable[tuple[torch.Tensor, int]], target_eval_samples: Iterable[tuple[torch.Tensor, int]], config: AttackConfig, adaptation_episodes: int = 25, seed: int = 0) -> dict[str, dict[str, float | str]]:
    source_digest_before = state_digest(source_policy.online)
    source_victim_digest = state_digest(source_victim)
    target_victim_digest = state_digest(target_victim)
    frozen = evaluate_policy(source_policy, target_victim, target_eval_samples, config, seed)
    frozen["policy_digest_before"] = source_digest_before
    frozen["policy_digest_after"] = state_digest(source_policy.online)
    continual = source_policy.clone()
    continual_before = state_digest(continual.online)
    train_policy(continual, target_victim, target_adaptation_samples, config, adaptation_episodes, seed + 1000)
    continual_result = evaluate_policy(continual, target_victim, target_eval_samples, config, seed)
    continual_result["policy_digest_before"] = continual_before
    continual_result["policy_digest_after"] = state_digest(continual.online)
    continual_result["updates"] = float(continual.updates)
    return {"frozen_transfer": frozen, "continual_transfer": continual_result, "invariants": {"source_policy_unchanged": source_digest_before == state_digest(source_policy.online), "source_victim_unchanged": source_victim_digest == state_digest(source_victim), "target_victim_unchanged": target_victim_digest == state_digest(target_victim), "continual_started_from_source": continual_before == source_digest_before}}
