from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from torch import nn

from rl_transfer.artifacts import sha256_file
from rl_transfer.phase2_residual_d1 import ResidualD1Request
from rl_transfer.phase2_residual_d1_source import (
    load_d1_source_context,
    load_residual_d1_source,
)
from rl_transfer.verified_artifacts import load_verified_json, write_verified_json


SOURCE_FAMILIES = ("classical_cnn", "transformer")


class ResidualD1SourceLifecycleTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
    ) -> tuple[ResidualD1Request, Path, dict[str, object], object]:
        source = root / "source"
        run_dir = source / "runs" / "fold-17-modern"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "policy.pt"
        checkpoint.write_bytes(b"sealed-source-policy")
        data = root / "data"
        data.mkdir()
        output = root / "study" / "d1a"
        manifest_path = source / "screen_manifest.json"
        run = {
            "seed": 17,
            "target_family": "modern_cnn",
            "source_families": list(SOURCE_FAMILIES),
            "run_dir": run_dir.name,
            "config": {},
            "policy": {
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
                "persistent_digest": "a" * 64,
            },
            "split_digest": "sealed-split",
            "victim_cache_digest": "b" * 64,
        }
        manifest = {
            "schema_version": 1,
            "status": "screen_complete",
            "research_valid": False,
            "dataset_version": ("torchvision-synthetic;content-sha256=" + "d" * 64),
            "source_runs": [run],
            "target_calls": 0,
            "target_evaluation_performed": False,
        }
        write_verified_json(manifest_path, manifest)
        request = ResidualD1Request(
            source_manifest=manifest_path,
            source_root=source,
            output_dir=output,
            data_root=data,
        )
        attack = SimpleNamespace(name="locked-d1-attack")
        config = SimpleNamespace(
            seed=17,
            split_seed=17,
            victim_train_images=1_000,
            policy_train_images=1_000,
            source_validation_images=300,
            outer_test_images=100,
            victim_validation_images=50,
            behavior_cloning_validation_episodes=100,
            source_evaluation_images=100,
            attack_config=lambda: attack,
        )
        return request, checkpoint, run, config

    def test_sealed_source_builds_five_disjoint_roles_and_source_victims(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, checkpoint, run, config = self._fixture(Path(directory))
            train_indices = tuple(range(200))
            threshold_pool = tuple(range(200, 300))
            source_gate = tuple(range(300, 400))
            threshold_indices = tuple(range(200, 250))
            competence_indices = tuple(range(250, 300))
            evaluation_indices = tuple(range(300, 350))
            ppo_evaluation_indices = tuple(range(350, 400))
            split = SimpleNamespace(
                digest="sealed-split",
                policy_train=train_indices,
                source_validation=tuple(range(200, 500)),
            )
            populations = {
                "exact_source": {
                    family: (
                        (f"{family}-teacher-0", nn.Identity()),
                        (f"{family}-teacher-1", nn.Identity()),
                    )
                    for family in SOURCE_FAMILIES
                },
                "seen_family_new_instance": {
                    family: ((f"{family}-evaluation", nn.Identity()),)
                    for family in SOURCE_FAMILIES
                },
            }
            train_dataset = SimpleNamespace(targets=[0] * 1_000)
            test_dataset = SimpleNamespace(targets=[0] * 100)

            with (
                patch(
                    "rl_transfer.phase2_residual_d1_source.MacPilotConfig",
                    return_value=config,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_source"
                    ".D1_SOURCE_MANIFEST_SHA256",
                    sha256_file(request.source_manifest),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_source"
                    ".validate_source_run_artifacts"
                ) as source_validator,
                patch(
                    "rl_transfer.phase2_residual_d1_source.validate_d1_attack_contract"
                ) as attack_validator,
                patch(
                    "rl_transfer.phase2_residual_d1_source.build_cifar_split",
                    return_value=split,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_source.balanced_subset_indices",
                    return_value=train_indices,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_source.disjoint_balanced_subsets",
                    side_effect=(
                        (
                            tuple(range(400, 450)),
                            threshold_pool,
                            source_gate,
                        ),
                        (threshold_indices, competence_indices),
                        (evaluation_indices, ppo_evaluation_indices),
                    ),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_source._source_indices",
                    return_value=source_gate,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_source._load_source_victims",
                    return_value=populations,
                ) as victim_loader,
                patch(
                    "rl_transfer.phase2_residual_d1_source.dataset_samples",
                    side_effect=lambda dataset, indices: tuple(
                        (f"sample-{index}", index % 10) for index in indices
                    ),
                ),
            ):
                source = load_residual_d1_source(request)
                context = load_d1_source_context(
                    request,
                    train_dataset,  # type: ignore[arg-type]
                    test_dataset,  # type: ignore[arg-type]
                    dataset_content_sha256="d" * 64,
                )

            self.assertEqual(source["target_calls"], 0)
            self.assertEqual(source["hidden_target_calls"], 0)
            self.assertFalse(source["target_evaluation_performed"])
            self.assertFalse(source["hidden_target_evaluation_performed"])
            self.assertFalse(source["authorizes_hidden_target_evaluation"])
            self.assertEqual(source["dataset_content_sha256"], "d" * 64)
            fold = source["folds"][0]
            self.assertEqual(fold["checkpoint_path"], checkpoint.resolve())
            self.assertEqual(fold["checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(context.source_families, SOURCE_FAMILIES)
            self.assertEqual(context.train_indices, train_indices)
            self.assertEqual(context.threshold_indices, threshold_indices)
            self.assertEqual(context.competence_indices, competence_indices)
            self.assertEqual(context.evaluation_indices, evaluation_indices)
            self.assertEqual(
                context.ppo_evaluation_indices,
                ppo_evaluation_indices,
            )
            self.assertTrue(context.role_audit["pairwise_disjoint"])
            self.assertEqual(
                context.role_audit["role_sizes"],
                {
                    "train": 200,
                    "threshold": 50,
                    "competence": 50,
                    "d1a_evaluation": 50,
                    "d1b_evaluation": 50,
                },
            )
            self.assertTrue(
                all(
                    len(digest) == 64
                    for digest in context.role_audit["role_indices_sha256"].values()
                )
            )
            self.assertTrue(
                all(
                    len(context.teacher_victims[family]) == 2
                    and len(context.evaluation_victims[family]) == 1
                    for family in SOURCE_FAMILIES
                )
            )
            source_validator.assert_called()
            attack_validator.assert_called_once_with(
                config.attack_config(),
                SOURCE_FAMILIES,
            )
            victim_loader.assert_called_once()

            checkpoint.write_bytes(b"tampered-source-policy")
            with (
                patch(
                    "rl_transfer.phase2_residual_d1_source.MacPilotConfig",
                    return_value=config,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_source"
                    ".D1_SOURCE_MANIFEST_SHA256",
                    sha256_file(request.source_manifest),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_source"
                    ".validate_source_run_artifacts"
                ),
                self.assertRaisesRegex(ValueError, "policy identity"),
            ):
                load_residual_d1_source(request)

            self.assertEqual(run["target_family"], "modern_cnn")

    def test_sealed_source_rejects_hidden_target_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, _, _, _ = self._fixture(Path(directory))
            manifest = load_verified_json(request.source_manifest)
            write_verified_json(
                request.source_manifest,
                {
                    **manifest,
                    "hidden_target_calls": 1,
                    "hidden_target_evaluation_performed": False,
                },
            )

            with (
                patch(
                    "rl_transfer.phase2_residual_d1_source"
                    ".D1_SOURCE_MANIFEST_SHA256",
                    sha256_file(request.source_manifest),
                ),
                self.assertRaisesRegex(ValueError, "hidden_target_calls"),
            ):
                load_residual_d1_source(request)


if __name__ == "__main__":
    unittest.main()
