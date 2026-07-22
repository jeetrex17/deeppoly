"""Reproducible RL adversarial-transfer experiments."""

from .config import AttackConfig, ExperimentConfig
from .dqn import DQNAgent
from .environment import PatchAttackEnv
from .protocols import evaluate_policy, run_transfer_protocols
from .recurrent import RecurrentAttackPolicy
from .registry import VictimRegistry, VictimSpec
from .research_protocol import run_frozen_episode, train_population_policy

__all__ = [
    "AttackConfig",
    "ExperimentConfig",
    "DQNAgent",
    "PatchAttackEnv",
    "evaluate_policy",
    "run_transfer_protocols",
    "RecurrentAttackPolicy",
    "VictimRegistry",
    "VictimSpec",
    "run_frozen_episode",
    "train_population_policy",
]
