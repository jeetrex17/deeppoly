import unittest

import torch
from torch import nn

from rl_transfer.config import AttackConfig
from rl_transfer.environment import EpisodeFinishedError, IneligibleSampleError, PatchAttackEnv


class MeanThresholdVictim(nn.Module):
    def __init__(self, threshold: float = 0.45) -> None:
        super().__init__()
        self.threshold = threshold
        self.queries = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.queries += images.shape[0]
        means = images.mean(dim=(1, 2, 3))
        threshold = torch.full_like(means, self.threshold)
        return torch.stack((means, threshold), dim=1)


class PatchAttackEnvironmentTests(unittest.TestCase):
    def test_detects_success_after_latest_action_and_counts_queries(self) -> None:
        victim = MeanThresholdVictim()
        config = AttackConfig(grid_size=1, epsilon=0.1, step_size=0.1, max_queries=1)
        env = PatchAttackEnv(victim, config)

        state = env.reset(torch.full((1, 4, 4), 0.5), label=0, sample_id="sample-0")
        next_state, reward, terminated, info = env.step(0)  # negative, channel 0

        self.assertEqual(state.shape, (7,))
        self.assertEqual(next_state.shape, (7,))
        self.assertTrue(terminated)
        self.assertTrue(info.success)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info.attack_queries, 1)
        self.assertEqual(victim.queries, 2)  # one eligibility query + one action query

    def test_projects_repeated_actions_to_pixel_and_linf_bounds(self) -> None:
        victim = MeanThresholdVictim(threshold=-1.0)
        config = AttackConfig(grid_size=2, epsilon=0.12, step_size=0.08, max_queries=3)
        env = PatchAttackEnv(victim, config)
        original = torch.full((3, 4, 4), 0.95)

        env.reset(original, label=0, sample_id="bounded")
        for _ in range(3):
            _, _, terminated, _ = env.step(1)  # positive, first patch/channel
            if terminated:
                break

        adversarial = env.adversarial_image
        self.assertTrue(torch.all(adversarial >= 0.0))
        self.assertTrue(torch.all(adversarial <= 1.0))
        self.assertLessEqual(float((adversarial - original).abs().max()), 0.120001)
        self.assertTrue(torch.equal(original, torch.full((3, 4, 4), 0.95)))

    def test_rejects_clean_misclassified_samples(self) -> None:
        env = PatchAttackEnv(
            MeanThresholdVictim(threshold=0.9),
            AttackConfig(grid_size=1, epsilon=0.1, step_size=0.05, max_queries=2),
        )

        with self.assertRaises(IneligibleSampleError):
            env.reset(torch.full((1, 4, 4), 0.5), label=0, sample_id="wrong")

    def test_rejects_actions_after_episode_finishes(self) -> None:
        env = PatchAttackEnv(
            MeanThresholdVictim(),
            AttackConfig(grid_size=1, epsilon=0.1, step_size=0.1, max_queries=1),
        )
        env.reset(torch.full((1, 4, 4), 0.5), label=0, sample_id="done")
        env.step(0)

        with self.assertRaises(EpisodeFinishedError):
            env.step(0)


class AttackConfigTests(unittest.TestCase):
    def test_validates_threat_model(self) -> None:
        invalid = (
            {"grid_size": 0},
            {"epsilon": 0.0},
            {"epsilon": 1.1},
            {"step_size": 0.0},
            {"step_size": 0.2, "epsilon": 0.1},
            {"max_queries": 0},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                AttackConfig(**overrides)


if __name__ == "__main__":
    unittest.main()
