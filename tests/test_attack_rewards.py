import math
import unittest

import torch

from rl_transfer.config import AttackConfig
from rl_transfer.environment import PatchAttackEnv
from rl_transfer.rewards import (
    dense_margin_reward,
    patch_environment_reward,
    recurrent_attack_reward,
    score_margin,
)


class MeanLogitVictim(torch.nn.Module):
    def __init__(self, rival_logit: float = 0.45) -> None:
        super().__init__()
        self.rival_logit = rival_logit

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(1, 2, 3))
        return torch.stack((means, torch.full_like(means, self.rival_logit)), dim=1)


class DenseAttackRewardTests(unittest.TestCase):
    def test_rewards_margin_reduction_including_rival_progress(self) -> None:
        config = AttackConfig(
            reward_mode="margin_delta",
            margin_reward_scale=2.0,
            terminal_success_bonus=3.0,
            query_penalty=0.1,
        )

        nonterminal = dense_margin_reward(0.5, 0.35, False, config)
        terminal = dense_margin_reward(0.5, -0.1, True, config)

        self.assertAlmostEqual(nonterminal, 0.2)
        self.assertAlmostEqual(terminal, 4.1)

    def test_recurrent_reward_uses_full_true_label_margin(self) -> None:
        config = AttackConfig(reward_mode="margin_delta", query_penalty=0.0)
        previous = torch.tensor((0.6, 0.3, 0.1))
        # The true score stays fixed, but the strongest rival rises by 0.1.
        current = torch.tensor((0.6, 0.4, 0.0))

        reward = recurrent_attack_reward(previous, current, 0, False, config)

        self.assertAlmostEqual(reward, 0.1, places=6)
        self.assertAlmostEqual(score_margin(previous, 0), 0.3, places=6)
        self.assertAlmostEqual(score_margin(current, 0), 0.2, places=6)

    def test_patch_environment_reports_incremental_margin_reduction(self) -> None:
        config = AttackConfig(
            grid_size=1,
            epsilon=0.1,
            step_size=0.1,
            max_queries=2,
            reward_mode="margin_delta",
            margin_reward_scale=1.5,
            query_penalty=0.02,
        )
        env = PatchAttackEnv(MeanLogitVictim(rival_logit=0.3), config)
        before = env.reset(torch.full((1, 4, 4), 0.5), label=0)

        after, reward, _, info = env.step(0)

        # With grid_size=1, index 3 is the p(true)-p(rival) state feature.
        expected_reduction = float(before[3] - after[3])
        self.assertAlmostEqual(info["margin_reduction"], expected_reduction, places=6)
        self.assertAlmostEqual(reward, 1.5 * expected_reduction - 0.02, places=6)
        self.assertEqual(info["reward_mode"], "margin_delta")

    def test_legacy_recurrent_reward_is_available(self) -> None:
        config = AttackConfig(
            reward_mode="legacy",
            terminal_success_bonus=10.0,
            query_penalty=0.05,
        )
        previous = torch.tensor((0.7, 0.2, 0.1))
        current = torch.tensor((0.6, 0.3, 0.1))

        self.assertAlmostEqual(
            recurrent_attack_reward(previous, current, 0, False, config),
            0.05,
            places=6,
        )
        self.assertEqual(recurrent_attack_reward(previous, current, 0, True, config), 10.0)

    def test_default_legacy_mode_reproduces_original_patch_rewards(self) -> None:
        config = AttackConfig()

        failed = patch_environment_reward(0.7, 0.2, 0.6, 0.3, False, 2, config)
        succeeded = patch_environment_reward(0.7, 0.2, 0.4, 0.5, True, 2, config)

        self.assertEqual(config.reward_mode, "legacy")
        self.assertAlmostEqual(failed, -0.65)
        self.assertAlmostEqual(succeeded, 9.6)

    def test_reward_configuration_and_score_inputs_are_validated(self) -> None:
        invalid = (
            {"reward_mode": "unknown"},
            {"margin_reward_scale": 0.0},
            {"margin_reward_scale": math.inf},
            {"terminal_success_bonus": -1.0},
            {"query_penalty": -0.01},
            {"query_penalty": math.nan},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                AttackConfig(**overrides)
        with self.assertRaises(ValueError):
            score_margin(torch.tensor((0.5,)), 0)
        with self.assertRaises(ValueError):
            score_margin(torch.tensor((0.5, 0.5)), 2)
        with self.assertRaises(ValueError):
            score_margin(torch.tensor((math.nan, 0.5)), 0)


if __name__ == "__main__":
    unittest.main()
