from pathlib import Path
from dataclasses import asdict
import json
import tempfile
import unittest

import numpy as np
import torch

from rl_transfer.artifacts import (
    load_recurrent_checkpoint,
    save_recurrent_checkpoint,
    sha256_file,
)
from rl_transfer.cifar_config import MacPilotConfig
from rl_transfer.cifar_policy_training import _new_policy
from rl_transfer.recurrent import (
    PPOConfig,
    PPOSequence,
    RecurrentAttackPolicy,
)


class ActionConditionedPolicyTests(unittest.TestCase):
    def test_cifar_config_selects_action_conditioned_actor(self) -> None:
        payload = json.loads(
            Path("configs/rl_transfer/cifar10_rtx_bc_ppo.json").read_text()
        )
        config = MacPilotConfig(
            **{
                **payload,
                "policy_actor_mode": "action_conditioned",
                "image_patch_feature_mode": "statistics",
                "behavior_cloning_soft_temperature": 0.5,
                "policy_evaluation_temperature": 0.5,
                "train_ablation_policies": False,
            }
        )
        attack = config.attack_config()
        policy = _new_policy(config, attack, torch.device("cpu"))
        self.assertEqual(policy.actor_mode, "action_conditioned")
        self.assertEqual(policy.action_grid_size, config.grid_size)
        self.assertEqual(attack.image_patch_feature_mode, "statistics")
        self.assertEqual(config.behavior_cloning_soft_temperature, 0.5)
        self.assertEqual(config.policy_evaluation_temperature, 0.5)
        with self.assertRaises(ValueError):
            MacPilotConfig(**{**payload, "policy_actor_mode": "unknown"})
        with self.assertRaises(ValueError):
            MacPilotConfig(
                **{
                    **payload,
                    "image_patch_feature_mode": "unknown",
                }
            )
        with self.assertRaises(ValueError):
            MacPilotConfig(
                **{
                    **payload,
                    "behavior_cloning_soft_temperature": 0.0,
                }
            )
        with self.assertRaises(ValueError):
            MacPilotConfig(
                **{
                    **payload,
                    "policy_evaluation_temperature": float("inf"),
                }
            )

    def test_action_conditioned_actor_requires_patch_catalog_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            RecurrentAttackPolicy(
                observation_dim=8,
                action_dim=95,
                hidden_dim=16,
                actor_mode="action_conditioned",
                action_grid_size=4,
            )
        with self.assertRaises(ValueError):
            RecurrentAttackPolicy(
                observation_dim=8,
                action_dim=96,
                hidden_dim=16,
                actor_mode="action_conditioned",
            )

    def test_action_conditioned_actor_has_shared_geometric_features(self) -> None:
        policy = RecurrentAttackPolicy(
            observation_dim=8,
            action_dim=24,
            hidden_dim=16,
            seed=4,
            actor_mode="action_conditioned",
            action_grid_size=2,
        )
        observation = torch.zeros(8)
        logits, value, hidden = policy(observation, policy.initial_state())
        self.assertEqual(logits.shape, (24,))
        self.assertEqual(value.shape, ())
        self.assertEqual(hidden.shape, (16,))
        self.assertEqual(policy.actor.action_features.shape, (24, 6))
        self.assertFalse(policy.actor.action_features.requires_grad)
        self.assertTrue(torch.isfinite(logits).all())

    def test_action_conditioned_policy_trains_and_remains_finite(self) -> None:
        policy = RecurrentAttackPolicy(
            observation_dim=3,
            action_dim=6,
            hidden_dim=8,
            seed=5,
            actor_mode="action_conditioned",
            action_grid_size=1,
            config=PPOConfig(entropy_weight=0.0, update_epochs=2),
        )
        observations = torch.eye(3)
        hidden = policy.initial_state()
        old_logs: list[torch.Tensor] = []
        with torch.no_grad():
            for observation in observations:
                logits, _, hidden = policy(observation, hidden)
                old_logs.append(torch.log_softmax(logits, dim=-1)[0])
        sequence = PPOSequence(
            observations=observations,
            actions=torch.zeros(3, dtype=torch.long),
            old_log_probabilities=torch.stack(old_logs),
            advantages=torch.ones(3),
            returns=torch.zeros(3),
        )
        before = policy.persistent_digest()
        metrics = policy.ppo_update_sequences([(sequence, 1.0)])
        self.assertNotEqual(before, policy.persistent_digest())
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_checkpoint_round_trip_preserves_actor_architecture(self) -> None:
        policy = RecurrentAttackPolicy(
            observation_dim=8,
            action_dim=24,
            hidden_dim=16,
            seed=9,
            actor_mode="action_conditioned",
            action_grid_size=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            digest = save_recurrent_checkpoint(
                path,
                policy,
                {"fingerprint": "test"},
            )
            restored, metadata = load_recurrent_checkpoint(
                path,
                expected_observation_dim=8,
                expected_action_dim=24,
                expected_hidden_dim=16,
                expected_actor_mode="action_conditioned",
            )
            self.assertEqual(metadata, {"fingerprint": "test"})
            self.assertEqual(restored.actor_mode, "action_conditioned")
            self.assertEqual(restored.action_grid_size, 2)
            self.assertEqual(restored.persistent_digest(), policy.persistent_digest())
            self.assertEqual(len(digest), 64)

    def test_legacy_flat_checkpoint_schema_remains_loadable(self) -> None:
        policy = RecurrentAttackPolicy(
            observation_dim=8,
            action_dim=6,
            hidden_dim=8,
            seed=11,
        )
        payload = {
            "schema_version": 1,
            "observation_dim": policy.observation_dim,
            "action_dim": policy.action_dim,
            "hidden_dim": policy.hidden_dim,
            "ppo_config": asdict(policy.config),
            "model": policy.state_dict(),
            "optimizer": policy.optimizer.state_dict(),
            "metadata": {"fingerprint": "phase-1"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(payload, path)
            path.with_suffix(".pt.sha256").write_text(
                sha256_file(path) + "\n"
            )
            restored, metadata = load_recurrent_checkpoint(
                path,
                expected_actor_mode="flat",
            )
        self.assertEqual(restored.actor_mode, "flat")
        self.assertIsNone(restored.action_grid_size)
        self.assertEqual(metadata, {"fingerprint": "phase-1"})


if __name__ == "__main__":
    unittest.main()
