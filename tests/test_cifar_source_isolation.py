import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.utils.data import Dataset

from rl_transfer.cifar_config import MacPilotConfig
from rl_transfer.cifar_pilot import run_cifar_pilot_from_datasets


class BalancedDataset(Dataset):
    def __init__(self, per_class: int) -> None:
        self.targets = [
            label
            for label in range(10)
            for _ in range(per_class)
        ]
        self.images = torch.zeros((len(self.targets), 3, 32, 32))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


def _config(output_dir: Path) -> MacPilotConfig:
    return MacPilotConfig(
        schema_version=1,
        name="strict-source-fixture",
        research_valid=False,
        dataset="CIFAR-10",
        device="cpu",
        download=False,
        data_root=str(output_dir / "data"),
        output_dir=str(output_dir),
        seed=17,
        victim_train_images=50,
        policy_train_images=50,
        source_validation_images=50,
        outer_test_images=20,
        victim_epochs=1,
        policy_episodes=2,
        policy_update_block=1,
        policy_learning_rate=0.001,
        policy_entropy_weight=0.001,
        policy_update_epochs=1,
        query_budget=2,
        grid_size=1,
        epsilon=8 / 255,
        step_size=2 / 255,
        batch_size=10,
        num_workers=0,
        hidden_dim=8,
        victim_learning_rate=0.001,
        target_family="classical_cnn",
        source_instances_per_family=1,
        source_holdout_instances_per_family=0,
        target_instances_per_family=1,
    )


def _population() -> dict[str, tuple[tuple[str, nn.Module], ...]]:
    return {
        "modern_cnn": (("modern-source", nn.Linear(1, 1)),),
        "transformer": (("transformer-source", nn.Linear(1, 1)),),
    }


def _checkpoint(cache_dir: Path, victim_id: str) -> None:
    path = cache_dir / f"{victim_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"checkpoint:{victim_id}".encode())
    path.with_suffix(".pt.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
    )


class StrictSourceIsolationTests(unittest.TestCase):
    def test_incomplete_matching_cache_fails_before_any_fit_or_load(self) -> None:
        train = BalancedDataset(20)
        test = BalancedDataset(5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = _config(output)
            cache_dir = output / "victim_cache" / ("a" * 12)
            _checkpoint(cache_dir, "modern-source")
            with (
                mock.patch(
                    "rl_transfer.cifar_pilot._victim_cache_digest",
                    return_value="a" * 64,
                ),
                mock.patch(
                    "rl_transfer.cifar_pilot._victim_cache_contract",
                    return_value={},
                ),
                mock.patch(
                    "rl_transfer.cifar_pilot._victim_code_digest",
                    return_value="v" * 64,
                ),
                mock.patch(
                    "rl_transfer.cifar_pilot.build_cifar_victim_population",
                    return_value=_population(),
                ) as builder,
                mock.patch(
                    "rl_transfer.cifar_pilot._train_classifier"
                ) as trainer,
                mock.patch(
                    "rl_transfer.cifar_pilot.load_model_checkpoint"
                ) as loader,
                mock.patch(
                    "rl_transfer.cifar_pilot._classifier_accuracy"
                ) as accuracy,
                self.assertRaisesRegex(
                    ValueError,
                    "cache-only",
                ),
            ):
                run_cifar_pilot_from_datasets(
                    config,
                    train,
                    test,
                    evaluate_target=False,
                    source_victims_only=True,
                    victim_cache_only=True,
                )

        self.assertEqual(
            builder.call_args.kwargs["families"],
            ("modern_cnn", "transformer"),
        )
        self.assertEqual(
            builder.call_args.args[1],
            {"modern_cnn": 1, "transformer": 1},
        )
        trainer.assert_not_called()
        loader.assert_not_called()
        accuracy.assert_not_called()

    def test_strict_source_mode_never_builds_or_validates_heldout_family(
        self,
    ) -> None:
        train = BalancedDataset(20)
        test = BalancedDataset(5)
        population = _population()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = _config(output)
            cache_dir = output / "victim_cache" / ("a" * 12)
            for victim_id in ("modern-source", "transformer-source"):
                _checkpoint(cache_dir, victim_id)
            training_seeds = {
                victim_id: int.from_bytes(
                    hashlib.sha256(
                        f"victim-fit-v1:{victim_id}".encode()
                    ).digest()[:8],
                    "big",
                )
                % (2**63 - 1)
                for victim_id in ("modern-source", "transformer-source")
            }

            def metadata(path, _model, _device):
                victim_id = Path(path).stem
                return {
                    "fingerprint": "a" * 64,
                    "cache_contract": {},
                    "training_seed": training_seeds[victim_id],
                    "history": [],
                    "fit_elapsed_seconds": 0.0,
                }

            with (
                mock.patch(
                    "rl_transfer.cifar_pilot._victim_cache_digest",
                    return_value="a" * 64,
                ),
                mock.patch(
                    "rl_transfer.cifar_pilot._victim_cache_contract",
                    return_value={},
                ),
                mock.patch(
                    "rl_transfer.cifar_pilot._victim_code_digest",
                    return_value="v" * 64,
                ),
                mock.patch(
                    "rl_transfer.cifar_pilot.build_cifar_victim_population",
                    return_value=population,
                ) as builder,
                mock.patch(
                    "rl_transfer.cifar_pilot._train_classifier"
                ) as trainer,
                mock.patch(
                    "rl_transfer.cifar_pilot.load_model_checkpoint",
                    side_effect=metadata,
                ) as loader,
                mock.patch(
                    "rl_transfer.cifar_pilot._classifier_accuracy",
                    return_value=0.9,
                ) as accuracy,
                mock.patch(
                    "rl_transfer.cifar_pilot.train_policy_bundle",
                    side_effect=RuntimeError("stop-after-victims"),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "stop-after-victims",
                ),
            ):
                run_cifar_pilot_from_datasets(
                    config,
                    train,
                    test,
                    evaluate_target=False,
                    source_victims_only=True,
                    victim_cache_only=True,
                )

        self.assertEqual(
            builder.call_args.kwargs["families"],
            ("modern_cnn", "transformer"),
        )
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(accuracy.call_count, 2)
        trainer.assert_not_called()
        self.assertFalse(
            any(
                "classical" in str(call).lower()
                for call in (
                    *loader.call_args_list,
                    *accuracy.call_args_list,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
