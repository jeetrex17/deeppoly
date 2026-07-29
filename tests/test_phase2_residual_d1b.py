from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import dataclass, replace
from typing import Any
from unittest.mock import patch

from rl_transfer.phase2_residual_d1 import D1_SOURCE_FAMILIES
from rl_transfer.phase2_residual_d1b import (
    D1B_BLOCK_EPISODES,
    D1B_TOTAL_EPISODES,
    ResidualD1BCheckpointReceipt,
    ResidualD1BDependencies,
    ResidualD1BLoadedCheckpoint,
    ResidualD1BSourceRole,
    ResidualD1BSourceRoles,
    ResidualD1BTrainingPayload,
    VerifiedD1AArtifacts,
    existing_residual_ppo_block,
    run_residual_d1b,
)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class FakePolicy:
    name: str
    generation: int = 0
    threshold: float = 0.1


def _policy_digest(policy: FakePolicy) -> str:
    return _digest(
        {
            "name": policy.name,
            "generation": policy.generation,
            "threshold": policy.threshold,
        }
    )


def _passing_manifest(*, passed: bool = True) -> dict[str, object]:
    decision = {
        "passed": passed,
        "eligible_for_d1b_source_only_ppo": passed,
        "authorizes_hidden_target_evaluation": False,
    }
    manifest: dict[str, object] = {
        "schema_version": 3,
        "name": "phase2-d1a-residual-ranker-bc",
        "status": "complete",
        "heldout_family": "modern_cnn",
        "source_families": list(D1_SOURCE_FAMILIES),
        "seed": 17,
        "bc_episodes": 200,
        "training_performed": passed,
        "teacher_completed": passed,
        "ppo_episodes_completed": 0,
        "target_calls": 0,
        "target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
        "d1_decision": decision,
        "source_split_roles": {
            "pairwise_disjoint": True,
            "role_sizes": {
                "train": 200,
                "threshold": 50,
                "competence": 50,
                "d1a_evaluation": 50,
                "d1b_evaluation": 50,
            },
            "role_indices_sha256": {
                "train": _digest(tuple(range(0, 200))),
                "threshold": _digest(tuple(range(200, 250))),
                "competence": _digest(tuple(range(250, 300))),
                "d1a_evaluation": _digest(tuple(range(300, 350))),
                "d1b_evaluation": _digest(tuple(range(350, 400))),
            },
        },
    }
    if passed:
        manifest["checkpoint"] = {
            "name": "residual_ranker_bc.pt",
            "sha256": "b" * 64,
            "persistent_digest": _policy_digest(FakePolicy("bc")),
        }
    return manifest


def _source_roles() -> ResidualD1BSourceRoles:
    return ResidualD1BSourceRoles(
        ppo_training=ResidualD1BSourceRole(
            "ppo_training",
            tuple(range(0, 200)),
            "ppo-payload",
        ),
        threshold_selection=ResidualD1BSourceRole(
            "threshold_selection",
            tuple(range(200, 250)),
            "threshold-payload",
        ),
        competence_gate=ResidualD1BSourceRole(
            "competence_gate",
            tuple(range(250, 300)),
            "competence-payload",
        ),
        evaluation=ResidualD1BSourceRole(
            "d1b_evaluation",
            tuple(range(350, 400)),
            "evaluation-payload",
        ),
    )


class FakeD1BHarness:
    def __init__(self) -> None:
        self.bc_policy = FakePolicy("bc")
        self.verify_calls: list[tuple[object, object]] = []
        self.clone_calls: list[FakePolicy] = []
        self.train_calls: list[dict[str, Any]] = []
        self.save_calls: list[dict[str, object]] = []
        self.load_calls: list[ResidualD1BCheckpointReceipt] = []
        self.threshold_payloads: list[object] = []
        self.competence_payloads: list[object] = []
        self.checkpoints: dict[object, FakePolicy] = {}
        self.checkpoint_metadata: dict[object, dict[str, object]] = {}
        self.bad_manifest_digest = False
        self.bad_checkpoint_sha = False
        self.bad_bc_digest = False
        self.bad_receipt_metadata = False
        self.alias_clone = False
        self.bad_clone_digest = False
        self.hidden_target_stage: str | None = None

    def verify(
        self,
        manifest: object,
        artifacts: object,
    ) -> VerifiedD1AArtifacts:
        self.verify_calls.append((manifest, artifacts))
        return VerifiedD1AArtifacts(
            bc_policy=self.bc_policy,
            manifest_digest=(
                "a" * 64 if self.bad_manifest_digest else _digest(manifest)
            ),
            checkpoint_sha256=("a" * 64 if self.bad_checkpoint_sha else "b" * 64),
            bc_policy_digest=(
                "a" * 64 if self.bad_bc_digest else _policy_digest(self.bc_policy)
            ),
            hidden_target_calls=0,
        )

    def clone(self, policy: FakePolicy) -> FakePolicy:
        self.clone_calls.append(policy)
        if self.alias_clone:
            return policy
        clone = copy.deepcopy(policy)
        if self.bad_clone_digest:
            clone.generation += 1
        return clone

    def train(
        self,
        policy: FakePolicy,
        payload: object,
        **kwargs: Any,
    ) -> dict[str, object]:
        self.train_calls.append({"payload": payload, **kwargs})
        kwargs["deadline_check"]()
        policy.generation += 1
        family_weights = {
            "classical_cnn": 0.45,
            "transformer": 0.55,
        }
        starting = kwargs["initial_instance_offsets"] or {
            family: 0 for family in D1_SOURCE_FAMILIES
        }
        instance_offsets = {
            family: int(starting[family]) + 25 for family in D1_SOURCE_FAMILIES
        }
        offset = int(kwargs["episode_offset"])
        return {
            "episodes": D1B_BLOCK_EPISODES,
            "trained_episodes": D1B_BLOCK_EPISODES,
            "episode_offset": offset,
            "next_episode_offset": offset + D1B_BLOCK_EPISODES,
            "family_weights": family_weights,
            "instance_offsets": instance_offsets,
            "source_calls": (99 if self.hidden_target_stage == "source_calls" else 100),
            "source_calls_by_family": {family: 50 for family in D1_SOURCE_FAMILIES},
            "source_calls_by_victim": (
                {"target-victim": 100}
                if self.hidden_target_stage == "target_victim"
                else {
                    "classical-source": 50,
                    "transformer-source": 50,
                }
            ),
            "hidden_target_calls": (1 if self.hidden_target_stage == "block" else 0),
        }

    def save(
        self,
        policy: FakePolicy,
        metadata: dict[str, object],
    ) -> ResidualD1BCheckpointReceipt:
        self.save_calls.append(copy.deepcopy(metadata))
        reference = f"d1b-{_policy_digest(policy)}-{_digest(metadata)}"
        self.checkpoints[reference] = copy.deepcopy(policy)
        self.checkpoint_metadata[reference] = copy.deepcopy(metadata)
        return ResidualD1BCheckpointReceipt(
            reference=reference,
            policy_digest=_policy_digest(policy),
            metadata_digest=(
                "c" * 64 if self.bad_receipt_metadata else _digest(metadata)
            ),
            hidden_target_calls=(1 if self.hidden_target_stage == "receipt" else 0),
        )

    def load(
        self,
        receipt: ResidualD1BCheckpointReceipt,
    ) -> ResidualD1BLoadedCheckpoint:
        self.load_calls.append(receipt)
        return ResidualD1BLoadedCheckpoint(
            policy=copy.deepcopy(self.checkpoints[receipt.reference]),
            metadata=copy.deepcopy(self.checkpoint_metadata[receipt.reference]),
            hidden_target_calls=0,
        )

    def select_threshold(
        self,
        policy: FakePolicy,
        payload: object,
        deadline_check: Any,
    ) -> dict[str, object]:
        deadline_check()
        self.threshold_payloads.append(payload)
        return {
            "selection_role": "d1b_threshold_selection_only",
            "threshold": 0.25,
            "accepted_steps": 1,
            "overrides_enabled": True,
            "target_calls": (1 if self.hidden_target_stage == "threshold" else 0),
        }

    def apply_threshold(
        self,
        policy: FakePolicy,
        threshold: object,
    ) -> FakePolicy:
        calibrated = copy.deepcopy(policy)
        calibrated.threshold = float(threshold["threshold"])
        return calibrated

    def competence(
        self,
        policy: FakePolicy,
        payload: object,
        deadline_check: Any,
    ) -> dict[str, object]:
        deadline_check()
        self.competence_payloads.append(payload)
        return {
            "target_mode": "all_soft",
            "accepted_steps": 1,
            "gated_top1_accuracy": 0.7,
            "prior_top1_accuracy": 0.6,
            "soft_cross_entropy": 0.8,
            "prior_soft_cross_entropy": 0.9,
            "residual_use_fraction": 0.2,
            "target_calls": (1 if self.hidden_target_stage == "competence" else 0),
        }

    def dependencies(self) -> ResidualD1BDependencies:
        return ResidualD1BDependencies(
            verify_d1a=self.verify,
            clone_policy=self.clone,
            policy_digest=_policy_digest,
            train_ppo_block=self.train,
            save_block_checkpoint=self.save,
            load_block_checkpoint=self.load,
            select_threshold=self.select_threshold,
            apply_threshold=self.apply_threshold,
            evaluate_competence=self.competence,
        )


class ResidualD1BTests(unittest.TestCase):
    def test_passing_d1a_runs_four_blocks_and_preserves_bc_checkpoint(
        self,
    ) -> None:
        manifest = _passing_manifest()
        roles = _source_roles()
        harness = FakeD1BHarness()
        deadline_calls: list[int] = []
        bc_digest = _policy_digest(harness.bc_policy)

        result = run_residual_d1b(
            manifest,
            "verified-artifact-handles",
            roles,
            harness.dependencies(),
            deadline_check=lambda: deadline_calls.append(1),
        )

        self.assertEqual(len(harness.verify_calls), 1)
        self.assertEqual(len(harness.clone_calls), 1)
        self.assertEqual(len(harness.train_calls), 4)
        self.assertEqual(len(harness.save_calls), 4)
        self.assertEqual(
            [metadata["episodes_completed"] for metadata in harness.save_calls],
            [50, 100, 150, 200],
        )
        self.assertTrue(
            all(
                metadata["d1a_checkpoint_sha256"] == "b" * 64
                and metadata["bc_policy_digest"] == bc_digest
                and metadata["target_calls"] == 0
                and metadata["hidden_target_calls"] == 0
                and not metadata["target_evaluation_performed"]
                and not metadata["target_evaluation_available"]
                and not metadata["authorizes_hidden_target_evaluation"]
                for metadata in harness.save_calls
            )
        )
        self.assertIsNone(harness.save_calls[0]["parent_checkpoint"])
        for block, metadata in zip(
            result.resume_state.blocks,
            harness.save_calls[1:],
        ):
            self.assertEqual(
                metadata["parent_checkpoint"]["reference"],
                block.checkpoint.reference,
            )
        self.assertEqual(
            [call["episode_offset"] for call in harness.train_calls],
            [0, 50, 100, 150],
        )
        self.assertTrue(
            all(
                call["episodes"] == 50
                and call["seed"] == 80_017
                and call["prior_seed"] == 50_017
                and call["payload"] == "ppo-payload"
                for call in harness.train_calls
            )
        )
        self.assertIsNone(harness.train_calls[0]["initial_family_weights"])
        self.assertIsNone(harness.train_calls[0]["initial_instance_offsets"])
        for previous, current in zip(
            result.resume_state.blocks,
            harness.train_calls[1:],
        ):
            self.assertEqual(
                current["initial_family_weights"],
                previous.family_weights,
            )
            self.assertEqual(
                current["initial_instance_offsets"],
                previous.instance_offsets,
            )
        self.assertEqual(_policy_digest(harness.bc_policy), bc_digest)
        self.assertEqual(result.manifest["bc_policy_digest_before"], bc_digest)
        self.assertEqual(result.manifest["bc_policy_digest_after"], bc_digest)
        self.assertTrue(result.manifest["bc_checkpoint_preserved"])
        self.assertEqual(result.manifest["ppo_episodes_completed"], D1B_TOTAL_EPISODES)
        self.assertEqual(result.manifest["ppo_blocks_completed"], 4)
        self.assertIsNone(result.manifest["ppo_skipped_reason"])
        self.assertEqual(result.manifest["hidden_target_calls"], 0)
        self.assertFalse(result.manifest["hidden_target_evaluation_performed"])
        self.assertEqual(result.manifest["target_calls"], 0)
        self.assertFalse(result.manifest["target_evaluation_performed"])
        self.assertFalse(result.manifest["target_evaluation_available"])
        self.assertFalse(result.manifest["authorizes_hidden_target_evaluation"])
        self.assertEqual(
            result.evaluation_inputs.methods,
            (
                "score_greedy",
                "residual_ranker_bc",
                "residual_ranker_bc_ppo",
            ),
        )
        self.assertEqual(
            result.evaluation_inputs.cohort,
            "evaluation-payload",
        )
        self.assertIs(result.evaluation_inputs.bc_policy, harness.bc_policy)
        self.assertEqual(
            harness.threshold_payloads,
            ["threshold-payload"],
        )
        self.assertEqual(
            harness.competence_payloads,
            ["competence-payload"],
        )
        self.assertGreaterEqual(len(deadline_calls), 10)

    def test_nonpassing_d1a_skips_before_artifact_or_training_calls(
        self,
    ) -> None:
        harness = FakeD1BHarness()

        result = run_residual_d1b(
            _passing_manifest(passed=False),
            None,
            _source_roles(),
            harness.dependencies(),
        )

        self.assertEqual(result.manifest["status"], "skipped")
        self.assertEqual(
            result.manifest["ppo_skipped_reason"],
            "d1a_source_gate_failed",
        )
        self.assertEqual(result.manifest["ppo_episodes_completed"], 0)
        self.assertIsNone(result.resume_state)
        self.assertIsNone(result.evaluation_inputs)
        self.assertFalse(harness.verify_calls)
        self.assertFalse(harness.clone_calls)
        self.assertFalse(harness.train_calls)
        self.assertFalse(harness.threshold_payloads)
        self.assertEqual(result.manifest["hidden_target_calls"], 0)
        with self.assertRaises(TypeError):
            result.manifest["status"] = "complete"

    def test_passing_artifacts_are_digest_bound_and_clone_must_be_distinct(
        self,
    ) -> None:
        for flag in (
            "bad_manifest_digest",
            "bad_checkpoint_sha",
            "bad_bc_digest",
        ):
            with self.subTest(flag=flag):
                harness = FakeD1BHarness()
                setattr(harness, flag, True)
                with self.assertRaisesRegex(
                    ValueError,
                    "manifest|checkpoint|policy|digest|binding",
                ):
                    run_residual_d1b(
                        _passing_manifest(),
                        "artifacts",
                        _source_roles(),
                        harness.dependencies(),
                    )
                self.assertFalse(harness.clone_calls)
                self.assertFalse(harness.train_calls)

        aliasing = FakeD1BHarness()
        aliasing.alias_clone = True
        before = _policy_digest(aliasing.bc_policy)
        with self.assertRaisesRegex(ValueError, "clone|frozen|distinct"):
            run_residual_d1b(
                _passing_manifest(),
                "artifacts",
                _source_roles(),
                aliasing.dependencies(),
            )
        self.assertEqual(_policy_digest(aliasing.bc_policy), before)
        self.assertFalse(aliasing.train_calls)

        corrupted = FakeD1BHarness()
        corrupted.bad_clone_digest = True
        with self.assertRaisesRegex(ValueError, "clone|frozen|digest"):
            run_residual_d1b(
                _passing_manifest(),
                "artifacts",
                _source_roles(),
                corrupted.dependencies(),
            )
        self.assertFalse(corrupted.train_calls)

    def test_source_roles_are_pairwise_disjoint_and_target_free(self) -> None:
        roles = _source_roles()
        with self.assertRaisesRegex(ValueError, "disjoint|overlap|role"):
            ResidualD1BSourceRoles(
                ppo_training=roles.ppo_training,
                threshold_selection=replace(
                    roles.threshold_selection,
                    sample_ids=(
                        roles.ppo_training.sample_ids[1],
                        *roles.threshold_selection.sample_ids[1:],
                    ),
                ),
                competence_gate=roles.competence_gate,
                evaluation=roles.evaluation,
            )
        with self.assertRaisesRegex(ValueError, "target|source.?only"):
            replace(roles.evaluation, hidden_target_calls=1)

        contaminated = _passing_manifest()
        contaminated["source_evaluation"] = {"target_calls": 1}
        harness = FakeD1BHarness()
        with self.assertRaisesRegex(ValueError, "target|source.?only"):
            run_residual_d1b(
                contaminated,
                "artifacts",
                roles,
                harness.dependencies(),
            )
        self.assertFalse(harness.verify_calls)
        self.assertFalse(harness.train_calls)

        mismatched_audit = _passing_manifest()
        mismatched_audit["source_split_roles"]["role_indices_sha256"][
            "d1b_evaluation"
        ] = "d" * 64
        with self.assertRaisesRegex(ValueError, "role|cohort|digest|binding"):
            run_residual_d1b(
                mismatched_audit,
                "artifacts",
                roles,
                harness.dependencies(),
            )
        self.assertFalse(harness.verify_calls)

    def test_resume_verifies_checkpoint_chain_and_runs_remaining_blocks(
        self,
    ) -> None:
        manifest = _passing_manifest()
        roles = _source_roles()
        harness = FakeD1BHarness()
        complete = run_residual_d1b(
            manifest,
            "artifacts",
            roles,
            harness.dependencies(),
        )
        partial = replace(
            complete.resume_state,
            blocks=complete.resume_state.blocks[:2],
        )
        harness.clone_calls.clear()
        harness.train_calls.clear()
        harness.save_calls.clear()
        harness.load_calls.clear()

        resumed = run_residual_d1b(
            manifest,
            "artifacts",
            roles,
            harness.dependencies(),
            resume_state=partial,
        )

        self.assertFalse(harness.clone_calls)
        self.assertEqual(
            harness.load_calls,
            [block.checkpoint for block in partial.blocks],
        )
        self.assertEqual(
            [call["episode_offset"] for call in harness.train_calls],
            [100, 150],
        )
        self.assertEqual(len(harness.save_calls), 2)
        self.assertEqual(len(resumed.resume_state.blocks), 4)
        self.assertEqual(resumed.resume_state.completed_episodes, 200)

        mismatched = replace(partial, source_roles_digest="d" * 64)
        harness.load_calls.clear()
        with self.assertRaisesRegex(ValueError, "resume|role|binding"):
            run_residual_d1b(
                manifest,
                "artifacts",
                roles,
                harness.dependencies(),
                resume_state=mismatched,
            )
        self.assertFalse(harness.load_calls)

        other_receipt = replace(
            partial.blocks[0].checkpoint,
            reference="other-checkpoint-1",
        )
        spliced = replace(
            partial,
            blocks=(
                replace(partial.blocks[0], checkpoint=other_receipt),
                partial.blocks[1],
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "resume|checkpoint|metadata|binding",
        ):
            run_residual_d1b(
                manifest,
                "artifacts",
                roles,
                harness.dependencies(),
                resume_state=spliced,
            )
        self.assertFalse(harness.load_calls)
        with self.assertRaisesRegex(ValueError, "reference|opaque"):
            replace(partial.blocks[0].checkpoint, reference="../unsafe.pt")
        self.assertFalse(harness.load_calls)

        substituted_receipt = replace(
            partial.blocks[-1].checkpoint,
            reference="safe-but-unbound-checkpoint",
        )
        substituted = replace(
            partial,
            blocks=(
                partial.blocks[0],
                replace(
                    partial.blocks[-1],
                    checkpoint=substituted_receipt,
                ),
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "resume|checkpoint|binding",
        ):
            run_residual_d1b(
                manifest,
                "artifacts",
                roles,
                harness.dependencies(),
                resume_state=substituted,
            )
        self.assertFalse(harness.load_calls)

        reference = partial.blocks[-1].checkpoint.reference
        harness.checkpoint_metadata[reference]["episodes_completed"] = 999
        harness.train_calls.clear()
        with self.assertRaisesRegex(
            ValueError,
            "resume|checkpoint|metadata|binding",
        ):
            run_residual_d1b(
                manifest,
                "artifacts",
                roles,
                harness.dependencies(),
                resume_state=partial,
            )
        self.assertFalse(harness.train_calls)

    def test_checkpoint_receipt_must_bind_exact_block_metadata(self) -> None:
        harness = FakeD1BHarness()
        harness.bad_receipt_metadata = True
        with self.assertRaisesRegex(ValueError, "checkpoint|metadata|digest"):
            run_residual_d1b(
                _passing_manifest(),
                "artifacts",
                _source_roles(),
                harness.dependencies(),
            )
        self.assertEqual(len(harness.train_calls), 1)
        self.assertEqual(
            _policy_digest(harness.bc_policy),
            _digest(
                {
                    "name": "bc",
                    "generation": 0,
                    "threshold": 0.1,
                }
            ),
        )

    def test_deadline_exception_propagates_before_checkpointing(self) -> None:
        harness = FakeD1BHarness()

        def deadline() -> None:
            return None

        def timeout_train(
            policy: object,
            payload: object,
            **kwargs: object,
        ) -> dict[str, object]:
            del policy, payload
            self.assertIs(kwargs["deadline_check"], deadline)
            raise TimeoutError("bounded D1b deadline")

        dependencies = replace(
            harness.dependencies(),
            train_ppo_block=timeout_train,
        )
        with self.assertRaisesRegex(TimeoutError, "bounded D1b deadline"):
            run_residual_d1b(
                _passing_manifest(),
                "artifacts",
                _source_roles(),
                dependencies,
                deadline_check=deadline,
            )
        self.assertFalse(harness.save_calls)
        self.assertFalse(harness.threshold_payloads)
        self.assertFalse(harness.competence_payloads)

    def test_nonzero_target_evidence_fails_closed_at_each_stage(self) -> None:
        for stage in (
            "block",
            "receipt",
            "threshold",
            "competence",
            "source_calls",
            "target_victim",
        ):
            with self.subTest(stage=stage):
                harness = FakeD1BHarness()
                harness.hidden_target_stage = stage
                with self.assertRaisesRegex(
                    ValueError,
                    "target|source.?only|source-call|inconsistent|held.?out",
                ):
                    run_residual_d1b(
                        _passing_manifest(),
                        "artifacts",
                        _source_roles(),
                        harness.dependencies(),
                    )

    def test_existing_ppo_adapter_forwards_opaque_training_payload(self) -> None:
        payload = ResidualD1BTrainingPayload(
            source_victims="source-victims",
            source_samples="source-samples",
            attack_config="attack",
        )
        with patch(
            "rl_transfer.phase2_residual_d1b.train_residual_ranker_ppo",
            return_value={"hidden_target_calls": 0},
        ) as trainer:
            result = existing_residual_ppo_block(
                "policy",
                payload,
                episodes=50,
                seed=1,
                prior_seed=2,
                initial_family_weights=None,
                episode_offset=0,
                initial_instance_offsets=None,
                deadline_check=None,
            )

        self.assertEqual(result, {"hidden_target_calls": 0})
        trainer.assert_called_once_with(
            "policy",
            "source-victims",
            "source-samples",
            "attack",
            episodes=50,
            seed=1,
            prior_seed=2,
            initial_family_weights=None,
            episode_offset=0,
            initial_instance_offsets=None,
            deadline_check=None,
        )


if __name__ == "__main__":
    unittest.main()
