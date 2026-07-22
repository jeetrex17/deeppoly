import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.utils.data import Dataset

from rl_transfer.artifacts import load_recurrent_checkpoint, save_recurrent_checkpoint
from rl_transfer.cifar_models import build_cifar_victims
from rl_transfer.cifar_pilot import (
    MacPilotConfig,
    build_cifar_split,
    run_cifar_pilot_from_datasets,
)
from rl_transfer.config import AttackConfig
from rl_transfer.models import freeze_model
from rl_transfer.recurrent import RecurrentAttackPolicy
from rl_transfer.research_protocol import run_frozen_episode, train_population_policy
from rl_transfer.runtime import resolve_device


class TwoClassVictim(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(1, 2, 3))
        return torch.stack((means, 1 - means), dim=1)


class BalancedTensorDataset(Dataset):
    def __init__(self, per_class: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.targets = [label for label in range(10) for _ in range(per_class)]
        self.images = torch.rand((len(self.targets), 3, 32, 32), generator=generator)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


class RuntimeTests(unittest.TestCase):
    @mock.patch("rl_transfer.runtime.torch.cuda.is_available", return_value=False)
    @mock.patch("rl_transfer.runtime.torch.backends.mps.is_available", return_value=True)
    def test_auto_prefers_available_mps(self, _mps, _cuda) -> None:
        selection = resolve_device("auto")
        self.assertEqual(selection.requested, "auto")
        self.assertEqual(selection.device, torch.device("mps"))

    @mock.patch("rl_transfer.runtime.torch.cuda.is_available", return_value=False)
    @mock.patch("rl_transfer.runtime.torch.backends.mps.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _mps, _cuda) -> None:
        self.assertEqual(resolve_device("auto").device, torch.device("cpu"))

    @mock.patch("rl_transfer.runtime.torch.backends.mps.is_available", return_value=False)
    def test_explicit_unavailable_mps_fails(self, _mps) -> None:
        with self.assertRaises(ValueError):
            resolve_device("mps")
        with self.assertRaises(ValueError):
            resolve_device("gpu")


class CIFARContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_labels = tuple(label for label in range(10) for _ in range(20))
        self.test_labels = tuple(label for label in range(10) for _ in range(5))

    def test_split_is_balanced_disjoint_and_reproducible(self) -> None:
        first = build_cifar_split(self.train_labels, self.test_labels, 50, 50, 50, 20, seed=7)
        second = build_cifar_split(self.train_labels, self.test_labels, 50, 50, 50, 20, seed=7)
        roles = (set(first.victim_fit), set(first.policy_train), set(first.source_validation))
        self.assertFalse(roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
        self.assertEqual(first, second)
        self.assertEqual(len(first.outer_test), 20)
        self.assertEqual({self.test_labels[index] for index in first.outer_test}, set(range(10)))

    def test_split_rejects_invalid_or_insufficient_counts(self) -> None:
        with self.assertRaises(ValueError):
            build_cifar_split(self.train_labels, self.test_labels, 51, 50, 50, 20, seed=7)
        with self.assertRaises(ValueError):
            build_cifar_split(self.train_labels, self.test_labels, 100, 100, 100, 20, seed=7)

    def test_committed_mac_config_is_bounded_and_not_research_valid(self) -> None:
        config = MacPilotConfig.from_json(Path("configs/rl_transfer/cifar10_m4_pilot.json"))
        self.assertEqual(config.dataset, "CIFAR-10")
        self.assertFalse(config.research_valid)
        self.assertEqual(config.device, "auto")
        self.assertLessEqual(config.query_budget, 100)
        self.assertLessEqual(config.policy_episodes, 1000)

    def test_iteration_config_enables_dense_rewards_and_source_population(self) -> None:
        config = MacPilotConfig.from_json(
            Path("configs/rl_transfer/cifar10_m4_iteration.json")
        )
        self.assertEqual(config.reward_mode, "margin_delta")
        self.assertEqual(config.victim_profile, "research")
        self.assertEqual(config.source_instances_per_family, 2)
        self.assertEqual(config.grid_size, 4)

    def test_victim_registry_has_three_distinct_model_families(self) -> None:
        victims = build_cifar_victims(seed=7)
        self.assertEqual(set(victims), {"classical_cnn", "modern_cnn", "transformer"})
        for _, model in victims.values():
            self.assertEqual(model(torch.rand((2, 3, 32, 32))).shape, (2, 10))

    def test_in_memory_pilot_writes_resumable_artifacts(self) -> None:
        train = BalancedTensorDataset(per_class=20, seed=1)
        test = BalancedTensorDataset(per_class=5, seed=2)
        with tempfile.TemporaryDirectory() as directory:
            config = MacPilotConfig(
                schema_version=1,
                name="fixture",
                research_valid=False,
                dataset="CIFAR-10",
                device="cpu",
                download=False,
                data_root=str(Path(directory) / "data"),
                output_dir=str(Path(directory) / "output"),
                seed=7,
                victim_train_images=50,
                policy_train_images=50,
                source_validation_images=50,
                outer_test_images=20,
                victim_epochs=1,
                policy_episodes=2,
                policy_update_block=1,
                policy_learning_rate=1e-3,
                policy_entropy_weight=1e-3,
                policy_update_epochs=2,
                query_budget=2,
                grid_size=1,
                epsilon=8 / 255,
                step_size=2 / 255,
                batch_size=25,
                num_workers=0,
                hidden_dim=8,
                victim_learning_rate=1e-3,
            )
            first = run_cifar_pilot_from_datasets(config, train, test, resume=True)
            second = run_cifar_pilot_from_datasets(config, train, test, resume=True)
            run_dir = Path(first["run_dir"])
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertFalse(first["research_valid"])
            self.assertTrue((run_dir / "manifest.json").is_file())
            self.assertTrue((run_dir / "policy.pt").is_file())
            self.assertTrue((run_dir / "results.jsonl").is_file())
            self.assertIn("groupdro_recurrent_ppo_stochastic", first["evaluation"])
            first_seeds = {
                instance["victim_id"]: instance["training_seed"]
                for instances in first["victim_instances"].values()
                for instance in instances
            }
            second_seeds = {
                instance["victim_id"]: instance["training_seed"]
                for instances in second["victim_instances"].values()
                for instance in instances
            }
            self.assertEqual(first_seeds, second_seeds)
            self.assertTrue(
                all(
                    instance["resumed"]
                    for instances in second["victim_instances"].values()
                    for instance in instances
                )
            )


class RecurrentArtifactTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_weights_only_and_verified(self) -> None:
        policy = RecurrentAttackPolicy(8, 6, hidden_dim=8, seed=3)
        metadata = {"dataset": "fixture", "seed": 3, "completed_episodes": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_recurrent_checkpoint(path, policy, metadata)
            restored, restored_metadata = load_recurrent_checkpoint(path, device="cpu")
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(policy.persistent_digest(), restored.persistent_digest())
        self.assertEqual(restored_metadata, metadata)


class PopulationVictimTests(unittest.TestCase):
    def test_family_schedule_round_robins_source_instances_and_audits_calls(self) -> None:
        attack = AttackConfig(grid_size=1, max_queries=2)
        policy = RecurrentAttackPolicy(8, attack.action_dim, hidden_dim=8, seed=3)
        instances = (
            ("source-a", freeze_model(TwoClassVictim())),
            ("source-b", freeze_model(TwoClassVictim())),
        )
        metrics = train_population_policy(
            policy,
            {"classical_cnn": instances},
            ((torch.full((3, 4, 4), 0.7), 0),),
            attack,
            episodes=4,
            seed=3,
        )
        self.assertEqual(metrics["source_calls_by_family"]["classical_cnn"], 8)
        self.assertEqual(metrics["source_calls_by_victim"], {"source-a": 4, "source-b": 4})

    def test_rollout_blocks_continue_global_sample_and_instance_offsets(self) -> None:
        attack = AttackConfig(grid_size=1, max_queries=2)
        policy = RecurrentAttackPolicy(8, attack.action_dim, hidden_dim=8, seed=5)
        instances = (
            ("source-a", freeze_model(TwoClassVictim())),
            ("source-b", freeze_model(TwoClassVictim())),
        )
        samples = tuple((torch.full((3, 4, 4), value), 0) for value in (0.7, 0.71, 0.72, 0.73))
        first = train_population_policy(
            policy,
            {"classical_cnn": instances},
            samples,
            attack,
            episodes=2,
            seed=5,
            episode_offset=0,
        )
        second = train_population_policy(
            policy,
            {"classical_cnn": instances},
            samples,
            attack,
            episodes=2,
            seed=7,
            initial_family_weights=first["family_weights"],
            episode_offset=2,
            initial_instance_offsets=first["instance_offsets"],
        )
        self.assertEqual(first["sample_indices"], [0, 1])
        self.assertEqual(second["sample_indices"], [2, 3])
        self.assertEqual(first["source_calls_by_victim"], {"source-a": 2, "source-b": 2})
        self.assertEqual(second["source_calls_by_victim"], {"source-a": 2, "source-b": 2})
        self.assertEqual(second["unique_sample_count"], 2)
        diagnostics = second["family_diagnostics"]["classical_cnn"]
        self.assertEqual(diagnostics["scheduled_episodes"], 2)
        self.assertEqual(diagnostics["eligible_episodes"], 2)
        self.assertIn("episode_return", diagnostics)
        self.assertIn("margin_reduction", diagnostics)
        self.assertIn("groupdro_loss", diagnostics)
        self.assertIn("weight_before", diagnostics)
        self.assertIn("weight_after", diagnostics)


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
class MPSIntegrationTests(unittest.TestCase):
    def test_recurrent_training_and_frozen_evaluation_run_on_mps(self) -> None:
        device = torch.device("mps")
        config = AttackConfig(grid_size=1, max_queries=2)
        policy = RecurrentAttackPolicy(8, config.action_dim, hidden_dim=8, seed=4).to(device)
        victim = freeze_model(TwoClassVictim().to(device))
        image = torch.full((3, 4, 4), 0.7)
        metrics = train_population_policy(
            policy,
            {"fixture": ("victim", victim)},
            ((image, 0),),
            config,
            episodes=1,
            seed=4,
        )
        result = run_frozen_episode(
            policy,
            victim,
            image,
            0,
            "sample",
            "victim",
            "fixture",
            config,
        )
        self.assertEqual(metrics["trained_episodes"], 1)
        self.assertEqual(result.policy_digest_before, result.policy_digest_after)


if __name__ == "__main__":
    unittest.main()
