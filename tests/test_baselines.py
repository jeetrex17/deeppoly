import unittest

import numpy as np

from rl_transfer.baselines import BanditActionPolicy


class BanditActionPolicyTests(unittest.TestCase):
    def test_explores_actions_then_reuses_rewarded_action_without_persistent_updates(self) -> None:
        policy = BanditActionPolicy(
            action_dim=3,
            seed=7,
            exploration=0.0,
            warmup_actions=2,
        )
        digest = policy.persistent_digest()
        observation = np.zeros(8, dtype=np.float32)
        state = policy.initial_state()
        actions = []
        for reward in (0.0, 0.8, -0.5):
            observation[6] = np.tanh(reward)
            action, state = policy.act(observation, state)
            actions.append(action)
        self.assertEqual(len(set(actions[:2])), 2)
        self.assertEqual(actions[2], actions[0])
        self.assertEqual(policy.persistent_digest(), digest)

    def test_rejects_invalid_configuration_and_observation(self) -> None:
        with self.assertRaises(ValueError):
            BanditActionPolicy(0, seed=1)
        with self.assertRaises(ValueError):
            BanditActionPolicy(2, seed=1, exploration=-1)
        with self.assertRaises(ValueError):
            BanditActionPolicy(2, seed=1, warmup_actions=0)
        with self.assertRaises(ValueError):
            BanditActionPolicy(2, seed=1).act(np.zeros(2, dtype=np.float32))

    def test_candidate_subset_varies_with_the_initial_observation(self) -> None:
        policy = BanditActionPolicy(action_dim=24, seed=7, warmup_actions=6)
        first = np.zeros(8, dtype=np.float32)
        second = np.arange(8, dtype=np.float32)
        _, first_state = policy.act(first)
        _, second_state = policy.act(second)
        self.assertNotEqual(first_state["candidates"], second_state["candidates"])


if __name__ == "__main__":
    unittest.main()
