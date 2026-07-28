import math
import unittest

import numpy as np
import torch
from torch import nn

from rl_transfer.config import AttackConfig
from rl_transfer.phase2_policy import FrozenTemperaturePolicy
from rl_transfer.recurrent import RecurrentAttackPolicy
from rl_transfer.research_protocol import run_frozen_episode


class FixedLogitPolicy(RecurrentAttackPolicy):
    """Small checkpoint stand-in with stable logits for sampling tests."""

    def __init__(self) -> None:
        super().__init__(observation_dim=3, action_dim=2, hidden_dim=4, seed=19)

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del observation
        logits = torch.tensor((2.0, 0.0), dtype=hidden.dtype, device=hidden.device)
        value = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
        return logits, value, hidden + 0.25


class TwoClassVictim(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(1, 2, 3))
        return torch.stack((means, 1 - means), dim=1)


class FrozenTemperaturePolicyTests(unittest.TestCase):
    def test_temperature_changes_probabilities_and_seeded_sampling(self) -> None:
        checkpoint = FixedLogitPolicy()
        cold = FrozenTemperaturePolicy(checkpoint, temperature=0.5)
        hot = FrozenTemperaturePolicy(checkpoint, temperature=2.0)
        observation = np.zeros(3, dtype=np.float32)
        hidden = checkpoint.initial_state()

        cold_probabilities = cold.action_probabilities(observation, hidden)
        hot_probabilities = hot.action_probabilities(observation, hidden)

        self.assertGreater(
            float(cold_probabilities[0]),
            float(hot_probabilities[0]),
        )
        self.assertEqual(
            cold.act(
                observation,
                hidden,
                deterministic=False,
                random_draw=0.8,
            )[0],
            0,
        )
        self.assertEqual(
            hot.act(
                observation,
                hidden,
                deterministic=False,
                random_draw=0.8,
            )[0],
            1,
        )

    def test_deterministic_action_matches_checkpoint_at_any_temperature(self) -> None:
        checkpoint = FixedLogitPolicy()
        observation = np.ones(3, dtype=np.float32)
        hidden = checkpoint.initial_state()
        expected_action, expected_hidden = checkpoint.act(
            observation,
            hidden,
            deterministic=True,
        )

        for temperature in (0.1, 1.0, 10.0):
            controlled = FrozenTemperaturePolicy(checkpoint, temperature)
            action, next_hidden = controlled.act(
                observation,
                hidden,
                deterministic=True,
            )
            self.assertEqual(action, expected_action)
            self.assertTrue(torch.equal(next_hidden, expected_hidden))

    def test_seeded_draws_are_reproducible_and_checkpoint_is_not_mutated(self) -> None:
        checkpoint = FixedLogitPolicy()
        digest_before = checkpoint.persistent_digest()
        state_before = {
            name: value.detach().clone()
            for name, value in checkpoint.state_dict().items()
        }
        controlled = FrozenTemperaturePolicy(checkpoint, temperature=1.5)
        observation = np.asarray((0.1, 0.2, 0.3), dtype=np.float32)
        observation_before = observation.copy()
        hidden = checkpoint.initial_state()
        hidden_before = hidden.clone()
        draws = np.random.default_rng(41).random(20)

        first = [
            controlled.act(
                observation,
                hidden,
                deterministic=False,
                random_draw=float(draw),
            )[0]
            for draw in draws
        ]
        second = [
            controlled.act(
                observation,
                hidden,
                deterministic=False,
                random_draw=float(draw),
            )[0]
            for draw in draws
        ]

        self.assertEqual(first, second)
        self.assertEqual(controlled.persistent_digest(), digest_before)
        self.assertEqual(checkpoint.persistent_digest(), digest_before)
        self.assertTrue(np.array_equal(observation, observation_before))
        self.assertTrue(torch.equal(hidden, hidden_before))
        for name, value in checkpoint.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[name]))

    def test_existing_frozen_episode_uses_reproducible_seeded_sampling(self) -> None:
        attack = AttackConfig(grid_size=1, max_queries=5)
        checkpoint = RecurrentAttackPolicy(
            observation_dim=attack.recurrent_observation_dim,
            action_dim=attack.action_dim,
            hidden_dim=8,
            seed=13,
        )
        controlled = FrozenTemperaturePolicy(checkpoint, temperature=1.7)
        image = torch.full((3, 4, 4), 0.7)

        first = run_frozen_episode(
            controlled,
            TwoClassVictim(),
            image,
            label=0,
            sample_id="sample",
            victim_id="victim",
            family="toy",
            config=attack,
            deterministic=False,
            episode_seed=71,
        )
        second = run_frozen_episode(
            controlled,
            TwoClassVictim(),
            image,
            label=0,
            sample_id="sample",
            victim_id="victim",
            family="toy",
            config=attack,
            deterministic=False,
            episode_seed=71,
        )

        self.assertEqual(first.actions, second.actions)
        self.assertEqual(
            first.policy_digest_before,
            first.policy_digest_after,
        )
        self.assertEqual(
            first.policy_digest_before,
            checkpoint.persistent_digest(),
        )

    def test_initial_state_and_model_metadata_are_delegated(self) -> None:
        checkpoint = FixedLogitPolicy()
        controlled = FrozenTemperaturePolicy(checkpoint, temperature=1.25)

        self.assertEqual(controlled.observation_dim, checkpoint.observation_dim)
        self.assertEqual(controlled.action_dim, checkpoint.action_dim)
        self.assertEqual(controlled.hidden_dim, checkpoint.hidden_dim)
        self.assertEqual(controlled.actor_mode, checkpoint.actor_mode)
        self.assertEqual(
            controlled.action_grid_size,
            checkpoint.action_grid_size,
        )
        self.assertIs(controlled.config, checkpoint.config)
        self.assertTrue(
            torch.equal(controlled.initial_state(), checkpoint.initial_state())
        )
        self.assertEqual(
            tuple(controlled.parameters()),
            tuple(checkpoint.parameters()),
        )

    def test_invalid_constructor_and_action_inputs_fail_closed(self) -> None:
        checkpoint = FixedLogitPolicy()
        for temperature in (0.0, -1.0, math.inf, -math.inf, math.nan, True):
            with self.subTest(temperature=temperature), self.assertRaises(
                (TypeError, ValueError)
            ):
                FrozenTemperaturePolicy(checkpoint, temperature)
        with self.assertRaises(TypeError):
            FrozenTemperaturePolicy(object(), 1.0)  # type: ignore[arg-type]

        controlled = FrozenTemperaturePolicy(checkpoint, 1.0)
        hidden = checkpoint.initial_state()
        valid_observation = np.zeros(3, dtype=np.float32)
        invalid_calls = (
            lambda: controlled.act(
                np.zeros((1, 3), dtype=np.float32),
                hidden,
            ),
            lambda: controlled.act(
                np.asarray((0.0, math.nan, 0.0), dtype=np.float32),
                hidden,
            ),
            lambda: controlled.act(
                valid_observation,
                torch.zeros(3),
            ),
            lambda: controlled.act(
                valid_observation,
                hidden,
                deterministic=1,  # type: ignore[arg-type]
            ),
            lambda: controlled.act(
                valid_observation,
                hidden,
                deterministic=False,
                random_draw=1.0,
            ),
            lambda: controlled.action_probabilities(
                valid_observation,
                torch.full((4,), math.nan),
            ),
            lambda: controlled.act(  # type: ignore[arg-type]
                valid_observation,
                None,
            ),
            lambda: controlled.act(
                valid_observation.astype(object),
                hidden,
            ),
            lambda: controlled.act(
                valid_observation,
                hidden.double(),
            ),
            lambda: controlled.act(
                valid_observation,
                hidden,
                deterministic=False,
                random_draw=math.nan,
            ),
            lambda: controlled.act(
                valid_observation,
                hidden,
                deterministic=False,
                random_draw=True,
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call), self.assertRaises(
                (TypeError, ValueError)
            ):
                invalid_call()

    def test_training_entry_points_are_disabled(self) -> None:
        controlled = FrozenTemperaturePolicy(FixedLogitPolicy(), 1.0)
        with self.assertRaises(RuntimeError):
            controlled.ppo_update(None)  # type: ignore[arg-type]
        with self.assertRaises(RuntimeError):
            controlled.ppo_update_sequences([])


if __name__ == "__main__":
    unittest.main()
