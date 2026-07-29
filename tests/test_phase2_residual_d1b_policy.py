from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from torch import nn
from rl_transfer.artifacts import sha256_file
from rl_transfer.phase2_residual_d1 import ResidualD1Request
from rl_transfer.phase2_residual_d1b_artifacts import canonical_json_digest
from rl_transfer.phase2_residual_d1b_policy import (
    ResidualD1AArtifactBundle,
    ResidualD1BEvaluationPayload,
    apply_d1b_threshold,
    build_d1b_source_roles,
    evaluate_d1b_competence,
    select_d1b_threshold,
    verify_d1a_artifacts,
)
from rl_transfer.recurrent import PPOConfig, RecurrentAttackPolicy
from rl_transfer.residual_ranker import ResidualRankerPolicy
from rl_transfer.verified_artifacts import write_verified_json


SOURCE_FAMILIES = ("classical_cnn", "transformer")


def _policy() -> ResidualRankerPolicy:
    return ResidualRankerPolicy(
        RecurrentAttackPolicy(
            12,
            96,
            hidden_dim=8,
            seed=9,
            config=PPOConfig(update_epochs=1),
            actor_mode="action_conditioned",
            action_grid_size=4,
        ),
        confidence_threshold=0.2,
        prior_temperature=24.0,
        overrides_enabled=True,
    )


def _request(root: Path) -> ResidualD1Request:
    source = root / "source"
    source.mkdir()
    source_manifest = source / "screen_manifest.json"
    source_manifest.write_text('{"sealed":true}\n')
    data = root / "data"
    data.mkdir()
    output = root / "study" / "d1a"
    output.mkdir(parents=True)
    return ResidualD1Request(
        source_manifest=source_manifest,
        source_root=source,
        output_dir=output,
        data_root=data,
    )


def _manifest(
    request: ResidualD1Request,
    policy: ResidualRankerPolicy,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "name": "phase2-d1a-residual-ranker-bc",
        "status": "complete",
        "request_sha256": request.digest(),
        "source_manifest_sha256": sha256_file(request.source_manifest),
        "dataset_content_sha256": "e" * 64,
        "heldout_family": "modern_cnn",
        "source_families": list(SOURCE_FAMILIES),
        "seed": 17,
        "bc_episodes": 200,
        "ppo_episodes_completed": 0,
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
        "threshold_selection": {
            "threshold": 0.2,
            "overrides_enabled": True,
        },
        "checkpoint": {
            "name": "residual_ranker_bc.pt",
            "sha256": "b" * 64,
            "persistent_digest": policy.persistent_digest(),
        },
        "source_evaluation": {
            family: {"target_calls": 0} for family in SOURCE_FAMILIES
        },
    }


class ResidualD1BPolicyTests(unittest.TestCase):
    def test_verified_d1a_reconstructs_exact_bc_and_recomputes_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            policy = _policy()
            manifest = _manifest(request, policy)
            write_verified_json(request.output_dir / "d1_manifest.json", manifest)
            bundle = ResidualD1AArtifactBundle(
                request=request,
                dataset_content_sha256="e" * 64,
                observation_dim=12,
                action_dim=96,
                hidden_dim=8,
                device="cpu",
            )
            checkpoint_metadata = {
                "schema_version": 3,
                "kind": "phase2_d1a_source_only_residual_ranker_bc",
                "seed": 17,
                "heldout_family": "modern_cnn",
                "source_manifest_sha256": sha256_file(request.source_manifest),
                "request_sha256": request.digest(),
                "confidence_threshold": 0.2,
                "prior_temperature": 24.0,
                "overrides_enabled": True,
                "target_calls": 0,
                "hidden_target_calls": 0,
                "target_evaluation_performed": False,
                "hidden_target_evaluation_performed": False,
                "target_evaluation_available": False,
                "authorizes_hidden_target_evaluation": False,
            }
            rows = ({"synthetic": "row"},)
            traces = ({"synthetic": "trace"},)

            with (
                patch(
                    "rl_transfer.phase2_residual_d1b_policy."
                    "_verify_complete_d1_children"
                ) as child_verifier,
                patch(
                    "rl_transfer.phase2_residual_d1b_policy.load_recurrent_checkpoint",
                    return_value=(
                        copy.deepcopy(policy.backbone),
                        checkpoint_metadata,
                    ),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_policy."
                    "load_verified_jsonl_records",
                    side_effect=(rows, traces),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_policy.verify_d1_raw_evidence",
                    return_value={"verified": True},
                ) as raw_verifier,
                patch(
                    "rl_transfer.phase2_residual_d1b_policy."
                    "verify_d1_recorded_summaries"
                ) as summary_verifier,
            ):
                verified = verify_d1a_artifacts(manifest, bundle)

            self.assertEqual(
                verified.bc_policy.persistent_digest(),
                policy.persistent_digest(),
            )
            self.assertEqual(
                verified.manifest_digest,
                canonical_json_digest(manifest),
            )
            self.assertEqual(verified.checkpoint_sha256, "b" * 64)
            child_verifier.assert_called_once()
            raw_verifier.assert_called_once_with(
                rows,
                traces,
                expected_methods=("score_greedy", "residual_ranker_bc"),
            )
            summary_verifier.assert_called_once_with(
                {"verified": True},
                manifest["source_evaluation"],
            )

    def test_d1a_verifier_fails_closed_on_binding_or_metadata_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            policy = _policy()
            manifest = _manifest(request, policy)
            write_verified_json(request.output_dir / "d1_manifest.json", manifest)
            bundle = ResidualD1AArtifactBundle(
                request=request,
                dataset_content_sha256="e" * 64,
                observation_dim=12,
                action_dim=96,
                hidden_dim=8,
                device="cpu",
            )
            metadata = {
                "schema_version": 3,
                "kind": "phase2_d1a_source_only_residual_ranker_bc",
                "seed": 17,
                "heldout_family": "modern_cnn",
                "source_manifest_sha256": sha256_file(request.source_manifest),
                "request_sha256": request.digest(),
                "confidence_threshold": 0.2,
                "prior_temperature": 24.0,
                "overrides_enabled": True,
                "target_calls": 0,
                "hidden_target_calls": 0,
                "target_evaluation_performed": False,
                "hidden_target_evaluation_performed": False,
                "target_evaluation_available": False,
                "authorizes_hidden_target_evaluation": False,
            }
            cases = {
                "request": ({**manifest, "request_sha256": "f" * 64}, metadata),
                "dataset": (
                    {**manifest, "dataset_content_sha256": "f" * 64},
                    metadata,
                ),
                "metadata": (
                    manifest,
                    {**metadata, "hidden_target_calls": 1},
                ),
            }
            for name, (candidate, checkpoint_metadata) in cases.items():
                with self.subTest(name=name):
                    write_verified_json(
                        request.output_dir / "d1_manifest.json",
                        candidate,
                    )
                    with (
                        patch(
                            "rl_transfer.phase2_residual_d1b_policy."
                            "_verify_complete_d1_children"
                        ),
                        patch(
                            "rl_transfer.phase2_residual_d1b_policy."
                            "load_recurrent_checkpoint",
                            return_value=(
                                copy.deepcopy(policy.backbone),
                                checkpoint_metadata,
                            ),
                        ),
                        patch(
                            "rl_transfer.phase2_residual_d1b_policy."
                            "load_verified_jsonl_records",
                            side_effect=(({},), ({},)),
                        ),
                        patch(
                            "rl_transfer.phase2_residual_d1b_policy."
                            "verify_d1_raw_evidence",
                            return_value={"verified": True},
                        ),
                        patch(
                            "rl_transfer.phase2_residual_d1b_policy."
                            "verify_d1_recorded_summaries"
                        ),
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            "binding|request|dataset|metadata|target|schema",
                        ):
                            verify_d1a_artifacts(candidate, bundle)

    def test_build_roles_uses_exact_disjoint_source_payloads(self) -> None:
        victims = {
            family: (
                (f"{family}-source-0", nn.Identity()),
                (f"{family}-source-1", nn.Identity()),
            )
            for family in SOURCE_FAMILIES
        }
        evaluation_victims = {
            family: ((f"{family}-eval-0", nn.Identity()),) for family in SOURCE_FAMILIES
        }
        attack = object()
        context = SimpleNamespace(
            source_families=SOURCE_FAMILIES,
            teacher_victims=victims,
            evaluation_victims=evaluation_victims,
            train_indices=tuple(range(0, 200)),
            threshold_indices=tuple(range(200, 250)),
            competence_indices=tuple(range(250, 300)),
            evaluation_indices=tuple(range(300, 350)),
            ppo_evaluation_indices=tuple(range(350, 400)),
            train_samples=tuple(f"train-{index}" for index in range(200)),
            ppo_evaluation_samples=tuple(f"evaluation-{index}" for index in range(50)),
            config=SimpleNamespace(attack_config=lambda: attack),
        )
        cache = SimpleNamespace(
            threshold_steps=("threshold-steps",),
            competence_steps=("competence-steps",),
        )

        roles = build_d1b_source_roles(context, cache)

        self.assertEqual(roles.ppo_training.sample_ids, tuple(range(200)))
        self.assertIs(
            roles.ppo_training.payload.source_victims,
            victims,
        )
        self.assertIs(
            roles.ppo_training.payload.attack_config,
            attack,
        )
        self.assertEqual(
            roles.threshold_selection.payload,
            ("threshold-steps",),
        )
        self.assertEqual(
            roles.competence_gate.payload,
            ("competence-steps",),
        )
        self.assertIsInstance(
            roles.evaluation.payload,
            ResidualD1BEvaluationPayload,
        )
        self.assertEqual(
            roles.evaluation.payload.sample_ids,
            tuple(range(350, 400)),
        )
        self.assertEqual(
            set(roles.evaluation.payload.source_victims),
            set(SOURCE_FAMILIES),
        )

        context.source_families = ("classical_cnn", "modern_cnn")
        with self.assertRaisesRegex(ValueError, "source|famil|held.?out"):
            build_d1b_source_roles(context, cache)

    def test_threshold_and_competence_adapters_preserve_deadline_and_roles(
        self,
    ) -> None:
        policy = _policy()
        deadline = Mock()
        selection = {
            "selection_role": "bc_validation_only",
            "threshold": 0.4,
            "accepted_steps": 10,
            "overrides_enabled": True,
        }
        competence = {
            "target_mode": "all_soft",
            "accepted_steps": 10,
            "gated_top1_accuracy": 0.7,
            "prior_top1_accuracy": 0.6,
            "soft_cross_entropy": 0.8,
            "prior_soft_cross_entropy": 0.9,
            "residual_use_fraction": 0.2,
            "by_source_family": {family: {} for family in SOURCE_FAMILIES},
            "equal_family_macro": {},
            "worst_family": {},
        }
        with (
            patch(
                "rl_transfer.phase2_residual_d1b_policy.select_confidence_threshold",
                return_value=selection,
            ) as selector,
            patch(
                "rl_transfer.phase2_residual_d1b_policy."
                "evaluate_residual_ranker_examples",
                return_value=competence,
            ) as evaluator,
        ):
            selected = select_d1b_threshold(
                policy,
                ("threshold",),
                deadline,
            )
            calibrated = apply_d1b_threshold(policy, selected)
            evaluated = evaluate_d1b_competence(
                calibrated,
                ("competence",),
                deadline,
            )

        self.assertEqual(
            selected["selection_role"],
            "d1b_threshold_selection_only",
        )
        self.assertEqual(selected["target_calls"], 0)
        self.assertEqual(evaluated["target_calls"], 0)
        self.assertIsNot(calibrated, policy)
        self.assertIsNot(calibrated.backbone, policy.backbone)
        self.assertEqual(calibrated.confidence_threshold, 0.4)
        selector.assert_called_once()
        evaluator.assert_called_once()
        self.assertIs(
            selector.call_args.kwargs["deadline_check"],
            deadline,
        )
        self.assertIs(
            evaluator.call_args.kwargs["deadline_check"],
            deadline,
        )


if __name__ == "__main__":
    unittest.main()
