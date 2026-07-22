import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from rl_transfer.config import DQNConfig
from rl_transfer.dqn import DQNAgent, Transition


def transition(index: int, state_dim: int = 5) -> Transition:
    state = np.full(state_dim, index / 10.0, dtype=np.float32)
    next_state = np.full(state_dim, (index + 1) / 10.0, dtype=np.float32)
    return Transition(state, index % 2, float(index), next_state, index % 3 == 0)


class DQNTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = DQNConfig(
            hidden_dims=(16,),
            batch_size=2,
            replay_capacity=8,
            min_replay_size=2,
            target_sync_interval=10,
            epsilon_start=0.2,
            epsilon_end=0.05,
            epsilon_decay=0.9,
            learning_rate=1e-2,
        )

    def test_learning_changes_online_policy_but_not_early_target(self) -> None:
        agent = DQNAgent(state_dim=5, action_dim=2, config=self.config, seed=7)
        target_before = agent.target_digest()
        policy_before = agent.policy_digest()
        agent.observe(transition(0))
        agent.observe(transition(1))

        loss = agent.learn()

        self.assertIsNotNone(loss)
        self.assertNotEqual(agent.policy_digest(), policy_before)
        self.assertEqual(agent.target_digest(), target_before)
        self.assertEqual(agent.update_count, 1)

    def test_clone_and_checkpoint_restore_complete_training_state(self) -> None:
        agent = DQNAgent(state_dim=5, action_dim=2, config=self.config, seed=13)
        for index in range(4):
            agent.observe(transition(index))
            agent.learn()
        agent.decay_epsilon()

        clone = agent.clone()
        self.assertEqual(clone.training_digest(), agent.training_digest())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.pt"
            agent.save(path)
            with mock.patch("rl_transfer.dqn.torch.load", wraps=torch.load) as loader:
                restored = DQNAgent.load(path, device="cpu")
            loader.assert_called_once_with(path, map_location="cpu", weights_only=True)

        self.assertEqual(restored.training_digest(), agent.training_digest())
        self.assertEqual(restored.epsilon, agent.epsilon)
        self.assertEqual(len(restored.replay), len(agent.replay))
        self.assertEqual(restored.update_count, agent.update_count)

    def test_greedy_evaluation_does_not_mutate_training_state(self) -> None:
        agent = DQNAgent(state_dim=5, action_dim=2, config=self.config, seed=3)
        before = agent.training_digest()

        first = agent.act(np.zeros(5, dtype=np.float32), evaluate=True)
        second = agent.act(np.zeros(5, dtype=np.float32), evaluate=True)

        self.assertEqual(first, second)
        self.assertEqual(agent.training_digest(), before)


if __name__ == "__main__":
    unittest.main()
