from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from rl_transfer.artifacts import sha256_file
from rl_transfer.phase2_residual_d1b import D1B_BLOCK_EPISODES
from rl_transfer.phase2_residual_d1b_artifacts import (
    ResidualD1BBlockStore,
    ResidualD1BStoreBinding,
    canonical_json_digest,
    clone_residual_policy,
)
from rl_transfer.recurrent import PPOConfig, RecurrentAttackPolicy
from rl_transfer.residual_ranker import ResidualRankerPolicy
from rl_transfer.verified_artifacts import load_verified_json, write_verified_json


SOURCE_FAMILIES = ("classical_cnn", "transformer")


def _policy() -> ResidualRankerPolicy:
    backbone = RecurrentAttackPolicy(
        12,
        96,
        hidden_dim=8,
        seed=7,
        config=PPOConfig(update_epochs=1),
        actor_mode="action_conditioned",
        action_grid_size=4,
    )
    return ResidualRankerPolicy(
        backbone,
        confidence_threshold=0.2,
        prior_temperature=24.0,
        overrides_enabled=True,
    )


def _binding(root: Path) -> ResidualD1BStoreBinding:
    return ResidualD1BStoreBinding(
        root=root,
        device="cpu",
        observation_dim=12,
        action_dim=96,
        hidden_dim=8,
        d1a_manifest_digest="a" * 64,
        d1a_checkpoint_sha256="b" * 64,
        bc_policy_digest="c" * 64,
        source_roles_digest="d" * 64,
    )


def _metrics(offset: int = 0) -> dict[str, object]:
    return {
        "episodes": D1B_BLOCK_EPISODES,
        "trained_episodes": D1B_BLOCK_EPISODES,
        "episode_offset": offset,
        "next_episode_offset": offset + D1B_BLOCK_EPISODES,
        "family_weights": {
            "classical_cnn": 0.45,
            "transformer": 0.55,
        },
        "instance_offsets": {family: offset // 2 + 25 for family in SOURCE_FAMILIES},
        "source_calls": 100,
        "source_calls_by_family": {family: 50 for family in SOURCE_FAMILIES},
        "source_calls_by_victim": {
            "classical-source": 50,
            "transformer-source": 50,
        },
        "hidden_target_calls": 0,
    }


def _metadata(
    policy: ResidualRankerPolicy,
    metrics: dict[str, object],
    binding: ResidualD1BStoreBinding,
    *,
    block_index: int = 1,
    parent_checkpoint: dict[str, object] | None = None,
) -> dict[str, object]:
    offset = (block_index - 1) * D1B_BLOCK_EPISODES
    return {
        "schema_version": 1,
        "name": "phase2-d1b-residual-ranker-ppo-block",
        "block_index": block_index,
        "episode_offset": offset,
        "episodes": D1B_BLOCK_EPISODES,
        "episodes_completed": block_index * D1B_BLOCK_EPISODES,
        "d1a_manifest_digest": binding.d1a_manifest_digest,
        "d1a_checkpoint_sha256": binding.d1a_checkpoint_sha256,
        "bc_policy_digest": binding.bc_policy_digest,
        "source_roles_digest": binding.source_roles_digest,
        "ppo_policy_digest": policy.persistent_digest(),
        "ppo_metrics_digest": canonical_json_digest(metrics),
        "parent_checkpoint": parent_checkpoint,
        "family_weights": metrics["family_weights"],
        "instance_offsets": metrics["instance_offsets"],
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


class FakeCheckpointIO:
    def __init__(self) -> None:
        self.saved: dict[Path, tuple[RecurrentAttackPolicy, dict[str, object]]] = {}

    def save(
        self,
        path: Path,
        policy: RecurrentAttackPolicy,
        metadata: dict[str, object],
    ) -> str:
        path.write_bytes(b"synthetic-checkpoint:" + path.name.encode())
        digest = sha256_file(path)
        path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
        self.saved[path] = (copy.deepcopy(policy), copy.deepcopy(metadata))
        return digest

    def load(
        self,
        path: Path,
        device: object = "cpu",
        **kwargs: object,
    ) -> tuple[RecurrentAttackPolicy, dict[str, object]]:
        del device, kwargs
        policy, metadata = self.saved[path]
        return copy.deepcopy(policy), copy.deepcopy(metadata)


class ResidualD1BArtifactTests(unittest.TestCase):
    def test_clone_is_distinct_digest_exact_and_independent(self) -> None:
        original = _policy()
        clone = clone_residual_policy(original)

        self.assertIsNot(clone, original)
        self.assertIsNot(clone.backbone, original.backbone)
        self.assertEqual(clone.persistent_digest(), original.persistent_digest())
        with __import__("torch").no_grad():
            next(clone.backbone.parameters()).add_(1.0)
        self.assertNotEqual(clone.persistent_digest(), original.persistent_digest())

    def test_block_store_saves_loads_and_reconstructs_verified_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = _binding(root)
            policy = _policy()
            metrics = _metrics()
            metadata = _metadata(policy, metrics, binding)
            checkpoint_io = FakeCheckpointIO()
            store = ResidualD1BBlockStore(binding)

            with (
                patch(
                    "rl_transfer.phase2_residual_d1b_artifacts."
                    "save_recurrent_checkpoint",
                    side_effect=checkpoint_io.save,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_artifacts."
                    "load_recurrent_checkpoint",
                    side_effect=checkpoint_io.load,
                ),
            ):
                receipt = store.save_block(policy, metadata, metrics)
                loaded = store.load_block(receipt)
                resumed = store.load_resume_state()

            self.assertEqual(
                loaded.policy.persistent_digest(), policy.persistent_digest()
            )
            self.assertEqual(dict(loaded.metadata), metadata)
            self.assertEqual(resumed.completed_episodes, 50)
            self.assertEqual(len(resumed.blocks), 1)
            self.assertEqual(resumed.blocks[0].checkpoint, receipt)
            self.assertEqual(dict(resumed.blocks[0].metrics), metrics)
            receipt_record = load_verified_json(root / "ppo_block_050.receipt.json")
            self.assertEqual(
                receipt_record["checkpoint"]["sha256"],
                sha256_file(root / "ppo_block_050.pt"),
            )
            self.assertEqual(receipt_record["target_calls"], 0)

    def test_verified_orphan_checkpoint_receipt_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = _binding(root)
            policy = _policy()
            metrics = _metrics()
            checkpoint_io = FakeCheckpointIO()
            store = ResidualD1BBlockStore(binding)
            with (
                patch(
                    "rl_transfer.phase2_residual_d1b_artifacts."
                    "save_recurrent_checkpoint",
                    side_effect=checkpoint_io.save,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_artifacts."
                    "load_recurrent_checkpoint",
                    side_effect=checkpoint_io.load,
                ),
            ):
                store.save_block(
                    policy,
                    _metadata(policy, metrics, binding),
                    metrics,
                )
                (root / "ppo_block_050.receipt.json").unlink()
                (root / "ppo_block_050.receipt.json.sha256").unlink()
                resumed = store.load_resume_state()

            self.assertEqual(resumed.completed_episodes, 50)
            self.assertTrue((root / "ppo_block_050.receipt.json").is_file())
            self.assertTrue((root / "ppo_block_050.receipt.json.sha256").is_file())

    def test_two_block_restart_preserves_and_verifies_parent_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = _binding(root)
            policy = _policy()
            checkpoint_io = FakeCheckpointIO()
            store = ResidualD1BBlockStore(binding)
            first_metrics = _metrics(0)
            second_metrics = _metrics(50)
            with (
                patch(
                    "rl_transfer.phase2_residual_d1b_artifacts."
                    "save_recurrent_checkpoint",
                    side_effect=checkpoint_io.save,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_artifacts."
                    "load_recurrent_checkpoint",
                    side_effect=checkpoint_io.load,
                ),
            ):
                first = store.save_block(
                    policy,
                    _metadata(policy, first_metrics, binding),
                    first_metrics,
                )
                parent = {
                    "reference": first.reference,
                    "policy_digest": first.policy_digest,
                    "metadata_digest": first.metadata_digest,
                }
                second = store.save_block(
                    policy,
                    _metadata(
                        policy,
                        second_metrics,
                        binding,
                        block_index=2,
                        parent_checkpoint=parent,
                    ),
                    second_metrics,
                )
                resumed = ResidualD1BBlockStore(binding).load_resume_state()

                self.assertEqual(resumed.completed_episodes, 100)
                self.assertEqual(len(resumed.blocks), 2)
                self.assertEqual(resumed.blocks[0].checkpoint, first)
                self.assertEqual(resumed.blocks[1].checkpoint, second)
                self.assertEqual(
                    resumed.blocks[1].checkpoint_metadata["parent_checkpoint"],
                    parent,
                )

                receipt_path = root / "ppo_block_100.receipt.json"
                tampered = load_verified_json(receipt_path)
                tampered["core_metadata"]["parent_checkpoint"]["reference"] = (
                    "d1b-" + "e" * 64 + "-" + "f" * 64
                )
                write_verified_json(receipt_path, tampered)
                with self.assertRaisesRegex(ValueError, "parent|binding"):
                    ResidualD1BBlockStore(binding).load_resume_state()

    def test_resume_rejects_rebound_identity_tamper_and_block_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = _binding(root)
            policy = _policy()
            metrics = _metrics()
            checkpoint_io = FakeCheckpointIO()
            store = ResidualD1BBlockStore(binding)
            with patch(
                "rl_transfer.phase2_residual_d1b_artifacts.save_recurrent_checkpoint",
                side_effect=checkpoint_io.save,
            ):
                store.save_block(
                    policy,
                    _metadata(policy, metrics, binding),
                    metrics,
                )

            receipt_path = root / "ppo_block_050.receipt.json"
            receipt = load_verified_json(receipt_path)
            receipt["binding"]["source_roles_digest"] = "e" * 64
            write_verified_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "binding|identity"):
                store.load_resume_state()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_verified_json(
                root / "ppo_block_100.receipt.json",
                {
                    "schema_version": 1,
                    "name": "phase2-d1b-residual-ranker-ppo-receipt",
                },
            )
            with self.assertRaisesRegex(ValueError, "gap|contiguous|orphan"):
                ResidualD1BBlockStore(_binding(root)).load_resume_state()

    def test_save_rejects_metadata_metric_and_source_only_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binding = _binding(Path(directory))
            policy = _policy()
            metrics = _metrics()
            metadata = _metadata(policy, metrics, binding)
            store = ResidualD1BBlockStore(binding)

            corrupted = copy.deepcopy(metadata)
            corrupted["ppo_metrics_digest"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "metric|digest"):
                store.save_block(policy, corrupted, metrics)

            contaminated = copy.deepcopy(metadata)
            contaminated["hidden_target_calls"] = 1
            with self.assertRaisesRegex(ValueError, "target|source"):
                store.save_block(policy, contaminated, metrics)

            wrong_policy = copy.deepcopy(metadata)
            wrong_policy["ppo_policy_digest"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "policy|digest"):
                store.save_block(policy, wrong_policy, metrics)

            inconsistent_calls = copy.deepcopy(metrics)
            inconsistent_calls["source_calls"] = 99
            with self.assertRaisesRegex(ValueError, "call|account"):
                store.save_block(policy, metadata, inconsistent_calls)


if __name__ == "__main__":
    unittest.main()
