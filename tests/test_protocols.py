import unittest

import numpy as np
import torch
from torch import nn

from rl_transfer.config import AttackConfig, DQNConfig
from rl_transfer.dqn import DQNAgent
from rl_transfer.protocols import AttackSample, run_continual_transfer, run_frozen_transfer
from rl_transfer.reproducibility import module_digest


class MeanThresholdVictim(nn.Module):
    def __init__(self, threshold: float = 0.45) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.tensor(threshold))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(1, 2, 3))
        threshold = self.threshold.expand_as(means)
        return torch.stack((means, threshold), dim=1)


def fixed_negative_agent() -> DQNAgent:
    config = DQNConfig(
        hidden_dims=(8,),
        batch_size=1,
        min_replay_size=1,
        replay_capacity=16,
        target_sync_interval=2,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        learning_rate=1e-2,
    )
    agent = DQNAgent(state_dim=7, action_dim=2, config=config, seed=5)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        agent.online.layers[-1].bias.copy_(torch.tensor([1.0, 0.0]))
        agent.target.load_state_dict(agent.online.state_dict())
    return agent


class TransferProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.attack = AttackConfig(grid_size=1, epsilon=0.1, step_size=0.1, max_queries=1)
        self.samples = tuple(
            AttackSample(f"sample-{index}", torch.full((1, 4, 4), 0.5), 0)
            for index in range(3)
        )

    def test_frozen_transfer_is_observationally_pure(self) -> None:
        agent = fixed_negative_agent()
        victim = MeanThresholdVictim()
        agent_before = agent.training_digest()
        victim_before = module_digest(victim)

        result = run_frozen_transfer(agent, victim, self.samples, self.attack)

        self.assertEqual(result.policy_digest_before, result.policy_digest_after)
        self.assertEqual(agent.training_digest(), agent_before)
        self.assertEqual(module_digest(victim), victim_before)
        self.assertEqual(result.metrics.eligible_samples, 3)
        self.assertEqual(result.metrics.attack_success_rate, 1.0)
        self.assertLessEqual(result.metrics.max_linf, self.attack.epsilon + 1e-6)

    def test_continual_transfer_updates_only_a_policy_clone(self) -> None:
        source_agent = fixed_negative_agent()
        victim = MeanThresholdVictim(threshold=-1.0)  # never fooled, provides full episodes
        source_before = source_agent.training_digest()
        victim_before = module_digest(victim)

        result, adapted = run_continual_transfer(
            source_agent,
            victim,
            self.samples,
            self.samples,
            self.attack,
            adaptation_epochs=2,
        )

        self.assertEqual(source_agent.training_digest(), source_before)
        self.assertEqual(module_digest(victim), victim_before)
        self.assertEqual(result.policy_digest_before, source_agent.policy_digest())
        self.assertNotEqual(result.policy_digest_after, result.policy_digest_before)
        self.assertGreater(adapted.update_count, source_agent.update_count)
        self.assertTrue(np.isfinite(result.metrics.mean_queries))


if __name__ == "__main__":
    unittest.main()
