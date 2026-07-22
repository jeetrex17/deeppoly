import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch

from rl_transfer.config import AttackConfig
from rl_transfer.dqn import DQNAgent
from rl_transfer.environment import PatchAttackEnv
from rl_transfer.models import SmallCNN, TargetCNN, freeze_model
from rl_transfer.protocols import evaluate_policy, run_transfer_protocols
from rl_transfer.reproducibility import state_digest


class TestRLTransfer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.config = AttackConfig(epsilon=.2, step_size=.1, grid_size=2, max_queries=3)
        self.image = torch.full((3, 8, 8), .5)

    def test_environment_post_action_and_bound(self):
        victim = freeze_model(SmallCNN())
        env = PatchAttackEnv(victim, self.image, 0, self.config, seed=3)
        env.reset()
        state, _, _, info = env.step(0)
        self.assertEqual(state.shape[0], self.config.state_dim)
        self.assertLessEqual(float((env.adv - env.original).abs().max()), self.config.epsilon + 1e-6)
        self.assertEqual(info["queries"], 2)

    def test_dqn_checkpoint_round_trip(self):
        agent = DQNAgent(self.config.state_dim, self.config.action_dim, seed=2, batch_size=1)
        state = torch.zeros(self.config.state_dim).numpy()
        agent.push(state, 0, 1, state, True)
        agent.learn()
        clone = DQNAgent(self.config.state_dim, self.config.action_dim, seed=9, batch_size=1)
        clone.load_checkpoint(copy.deepcopy(agent.checkpoint()))
        self.assertEqual(state_digest(agent.online), state_digest(clone.online))
        self.assertEqual(agent.updates, clone.updates)

    def test_frozen_and_continual_invariants(self):
        source = freeze_model(SmallCNN())
        target = freeze_model(TargetCNN())
        samples = [(self.image, 0)] * 4
        agent = DQNAgent(self.config.state_dim, self.config.action_dim, seed=1, batch_size=1)
        before = state_digest(agent.online)
        result = run_transfer_protocols(agent, source, target, samples, samples[:2], samples[2:], self.config, adaptation_episodes=3)
        self.assertTrue(result["invariants"]["source_policy_unchanged"])
        self.assertTrue(result["invariants"]["source_victim_unchanged"])
        self.assertTrue(result["invariants"]["target_victim_unchanged"])
        self.assertEqual(before, result["frozen_transfer"]["policy_digest_after"])
        self.assertGreater(result["continual_transfer"]["updates"], 0)

    def test_empty_eligible_is_nan(self):
        victim = freeze_model(SmallCNN())
        # Label 9 is almost certainly not the prediction for this untrained fixture.
        agent = DQNAgent(self.config.state_dim, self.config.action_dim, seed=1)
        result = evaluate_policy(agent, victim, [(self.image, 9)], self.config)
        self.assertEqual(result["eligible"], 0)
        self.assertTrue(math.isnan(result["attack_success_rate"]))


if __name__ == "__main__":
    unittest.main()
