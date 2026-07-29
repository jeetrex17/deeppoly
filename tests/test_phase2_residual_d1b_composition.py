from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from rl_transfer.artifacts import sha256_file
from rl_transfer.phase2_residual_d1 import ResidualD1Request
from rl_transfer.phase2_residual_d1_teacher import D1_HIDDEN_DIM
from rl_transfer.phase2_residual_d1b import (
    D1B_BLOCK_ENDPOINTS,
    D1B_BLOCK_EPISODES,
    ResidualD1BSourceRole,
    ResidualD1BSourceRoles,
    ResidualD1BTrainingPayload,
    VerifiedD1AArtifacts,
)
from rl_transfer.phase2_residual_d1b_artifacts import (
    ResidualD1BBlockStore,
    ResidualD1BStoreBinding,
    canonical_json_digest,
)
from rl_transfer.phase2_residual_d1b_runner import (
    run_residual_d1b_from_datasets,
)
from rl_transfer.recurrent import PPOConfig, RecurrentAttackPolicy
from rl_transfer.residual_ranker import ResidualRankerPolicy
from rl_transfer.verified_artifacts import load_verified_json


SOURCE_FAMILIES = ("classical_cnn", "transformer")


def _seal() -> dict[str, object]:
    return {
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _request(root: Path) -> ResidualD1Request:
    source = root / "source"
    source.mkdir()
    source_manifest = source / "screen_manifest.json"
    source_manifest.write_text('{"sealed":true}\n')
    data = root / "data"
    data.mkdir()
    return ResidualD1Request(
        source_manifest=source_manifest,
        source_root=source,
        output_dir=root / "study" / "d1a",
        data_root=data,
    )


def _policy(seed: int = 17) -> ResidualRankerPolicy:
    return ResidualRankerPolicy(
        RecurrentAttackPolicy(
            12,
            96,
            hidden_dim=D1_HIDDEN_DIM,
            seed=seed,
            config=PPOConfig(update_epochs=1),
            actor_mode="action_conditioned",
            action_grid_size=4,
        ),
        confidence_threshold=0.2,
        prior_temperature=24.0,
        overrides_enabled=True,
    )


def _source_roles() -> ResidualD1BSourceRoles:
    return ResidualD1BSourceRoles(
        ppo_training=ResidualD1BSourceRole(
            "ppo_training",
            tuple(range(0, 200)),
            ResidualD1BTrainingPayload(
                source_victims={family: () for family in SOURCE_FAMILIES},
                source_samples=(),
                attack_config={},
            ),
        ),
        threshold_selection=ResidualD1BSourceRole(
            "threshold_selection",
            tuple(range(200, 250)),
            (),
        ),
        competence_gate=ResidualD1BSourceRole(
            "competence_gate",
            tuple(range(250, 300)),
            (),
        ),
        evaluation=ResidualD1BSourceRole(
            "d1b_evaluation",
            tuple(range(350, 400)),
            "reserved-source-evaluation",
        ),
    )


def _d1a_manifest(
    policy: ResidualRankerPolicy,
    roles: ResidualD1BSourceRoles,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "name": "phase2-d1a-residual-ranker-bc",
        "status": "complete",
        "heldout_family": "modern_cnn",
        "source_families": list(SOURCE_FAMILIES),
        "seed": 17,
        "bc_episodes": 200,
        "ppo_episodes_completed": 0,
        "training_performed": True,
        "teacher_completed": True,
        "checkpoint": {
            "name": "residual_ranker_bc.pt",
            "sha256": "b" * 64,
            "persistent_digest": policy.persistent_digest(),
        },
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
                "train": roles.ppo_training.sample_ids_sha256,
                "threshold": roles.threshold_selection.sample_ids_sha256,
                "competence": roles.competence_gate.sample_ids_sha256,
                "d1a_evaluation": canonical_json_digest(tuple(range(300, 350))),
                "d1b_evaluation": roles.evaluation.sample_ids_sha256,
            },
        },
        "d1_decision": {
            "passed": True,
            "eligible_for_d1b_source_only_ppo": True,
            "authorizes_hidden_target_evaluation": False,
        },
        **_seal(),
    }


def _block_metrics(
    episode_offset: int,
    initial_offsets: object,
) -> dict[str, object]:
    starting = (
        {family: 0 for family in SOURCE_FAMILIES}
        if initial_offsets is None
        else dict(initial_offsets)  # type: ignore[arg-type]
    )
    return {
        "episodes": D1B_BLOCK_EPISODES,
        "trained_episodes": D1B_BLOCK_EPISODES,
        "episode_offset": episode_offset,
        "next_episode_offset": episode_offset + D1B_BLOCK_EPISODES,
        "family_weights": {
            "classical_cnn": 0.45,
            "transformer": 0.55,
        },
        "instance_offsets": {
            family: int(starting[family]) + 25 for family in SOURCE_FAMILIES
        },
        "source_calls": 100,
        "source_calls_by_family": {family: 50 for family in SOURCE_FAMILIES},
        "source_calls_by_victim": {
            "classical-source": 50,
            "transformer-source": 50,
        },
        **_seal(),
    }


def _write_verified_bytes(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


class _CheckpointIO:
    """Small deterministic stand-in for only the tensor serialization seam."""

    def __init__(self) -> None:
        self.saved: dict[
            Path,
            tuple[RecurrentAttackPolicy, dict[str, object]],
        ] = {}

    def save(
        self,
        path: Path,
        policy: RecurrentAttackPolicy,
        metadata: dict[str, object],
    ) -> str:
        digest = _write_verified_bytes(
            path,
            (
                f"synthetic:{path.name}:{policy.persistent_digest()}".encode(
                    "utf-8"
                )
            ),
        )
        self.saved[path] = (copy.deepcopy(policy), copy.deepcopy(metadata))
        return digest

    def load(
        self,
        path: Path,
        device: object,
        **expected: object,
    ) -> tuple[RecurrentAttackPolicy, dict[str, object]]:
        del device, expected
        policy, metadata = self.saved[path]
        return copy.deepcopy(policy), copy.deepcopy(metadata)


class _SyntheticWorkerLoss(BaseException):
    """Model an abrupt process loss that bypasses terminal-manifest handling."""


class ResidualD1BCompositionTests(unittest.TestCase):
    def test_real_runner_core_and_store_resume_four_block_disk_lifecycle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            output = root / "study" / "d1b"
            roles = _source_roles()
            bc_policy = _policy()
            bc_digest_before = bc_policy.persistent_digest()
            d1a = _d1a_manifest(bc_policy, roles)
            verified = VerifiedD1AArtifacts(
                bc_policy=bc_policy,
                manifest_digest=canonical_json_digest(d1a),
                checkpoint_sha256="b" * 64,
                bc_policy_digest=bc_digest_before,
                hidden_target_calls=0,
            )
            context = SimpleNamespace(
                config=SimpleNamespace(
                    attack_config=lambda: SimpleNamespace(
                        recurrent_observation_dim=12,
                        action_dim=96,
                    )
                )
            )
            checkpoint_io = _CheckpointIO()
            attempted_offsets: list[int] = []
            committed_offsets: list[int] = []
            crash_pending = True

            def train_block(
                policy: ResidualRankerPolicy,
                payload: object,
                **kwargs: object,
            ) -> dict[str, object]:
                nonlocal crash_pending
                self.assertIs(payload, roles.ppo_training.payload)
                offset = int(kwargs["episode_offset"])  # type: ignore[arg-type]
                attempted_offsets.append(offset)
                kwargs["deadline_check"]()  # type: ignore[operator]
                if crash_pending and offset == 100:
                    crash_pending = False
                    raise _SyntheticWorkerLoss("simulated worker exit")
                with torch.no_grad():
                    next(policy.backbone.parameters()).add_(
                        0.0001 * (offset + 1)
                    )
                committed_offsets.append(offset)
                return _block_metrics(
                    offset,
                    kwargs["initial_instance_offsets"],
                )

            def select_threshold(
                policy: ResidualRankerPolicy,
                payload: object,
                deadline_check: object,
            ) -> dict[str, object]:
                self.assertIsInstance(policy, ResidualRankerPolicy)
                self.assertIs(payload, roles.threshold_selection.payload)
                deadline_check()  # type: ignore[operator]
                return {
                    "selection_role": "d1b_threshold_selection_only",
                    "threshold": 0.25,
                    "accepted_steps": 50,
                    "overrides_enabled": True,
                    **_seal(),
                }

            def evaluate_competence(
                policy: ResidualRankerPolicy,
                payload: object,
                deadline_check: object,
            ) -> dict[str, object]:
                self.assertIsInstance(policy, ResidualRankerPolicy)
                self.assertIs(payload, roles.competence_gate.payload)
                deadline_check()  # type: ignore[operator]
                return {
                    "target_mode": "all_soft",
                    "accepted_steps": 50,
                    "gated_top1_accuracy": 0.72,
                    "prior_top1_accuracy": 0.64,
                    "soft_cross_entropy": 0.73,
                    "prior_soft_cross_entropy": 0.81,
                    "residual_use_fraction": 0.18,
                    **_seal(),
                }

            def calibrated_checkpoint(
                destination: Path,
                policy: ResidualRankerPolicy,
                **binding: object,
            ) -> dict[str, object]:
                self.assertEqual(
                    set(binding),
                    {
                        "request",
                        "d1a_manifest_digest",
                        "source_roles_digest",
                    },
                )
                digest = _write_verified_bytes(
                    destination / "residual_ranker_ppo.pt",
                    policy.persistent_digest().encode("utf-8"),
                )
                return {
                    "name": "residual_ranker_ppo.pt",
                    "sha256": digest,
                    "persistent_digest": policy.persistent_digest(),
                    "metadata_sha256": "9" * 64,
                }

            def evaluate_source(
                destination: Path,
                inputs: object,
                **callbacks: object,
            ) -> dict[str, object]:
                self.assertEqual(
                    inputs.sample_ids,  # type: ignore[attr-defined]
                    roles.evaluation.sample_ids,
                )
                self.assertEqual(
                    set(callbacks),
                    {"deadline_check", "progress"},
                )
                results = _write_verified_bytes(
                    destination / "source_results.jsonl",
                    b'{"synthetic":true}\n',
                )
                traces = _write_verified_bytes(
                    destination / "source_query_traces.jsonl",
                    b'{"synthetic":true}\n',
                )
                figures = {
                    name: _write_verified_bytes(
                        destination / name,
                        b"<svg/>",
                    )
                    for name in ("asr_by_query.svg", "final_asr.svg")
                }
                return {
                    "source_evaluation": {"synthetic_verified": True},
                    "paired_uncertainty": {"synthetic_verified": True},
                    "raw_evidence_verification": {"synthetic_verified": True},
                    "results_sha256": results,
                    "query_traces_sha256": traces,
                    "figures": figures,
                    "decision": {
                        "passed": True,
                        "selected_method": "residual_ranker_bc_ppo",
                        "authorizes_hidden_target_evaluation": False,
                    },
                    "source_model_calls": 36,
                    **_seal(),
                }

            with (
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "load_d1_source_context",
                    return_value=context,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner._cache_binding",
                    return_value=("binding", "protocol"),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "load_residual_teacher_cache",
                    return_value=object(),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "build_d1b_source_roles",
                    return_value=roles,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.verify_d1a_artifacts",
                    return_value=verified,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "existing_residual_ppo_block",
                    side_effect=train_block,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.select_d1b_threshold",
                    side_effect=select_threshold,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "evaluate_d1b_competence",
                    side_effect=evaluate_competence,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "_calibrated_checkpoint",
                    side_effect=calibrated_checkpoint,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner._evaluate_d1b_source",
                    side_effect=evaluate_source,
                ) as source_evaluator,
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "verify_complete_d1b_children"
                ) as complete_verifier,
                patch(
                    "rl_transfer.phase2_residual_d1b_runner._gpu_memory_record",
                    return_value=None,
                ),
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
                with self.assertRaisesRegex(
                    _SyntheticWorkerLoss,
                    "simulated worker exit",
                ):
                    run_residual_d1b_from_datasets(
                        request,
                        output,
                        d1a,
                        object(),  # type: ignore[arg-type]
                        object(),  # type: ignore[arg-type]
                        dataset_version="synthetic-v1",
                        dataset_content_sha256="6" * 64,
                        runtime_environment={"environment_sha256": "7" * 64},
                        deadline_check=lambda: None,
                        progress=lambda _: None,
                    )

                interrupted = load_verified_json(output / "d1b_manifest.json")
                self.assertEqual(interrupted["status"], "running")
                self.assertEqual(
                    [
                        endpoint
                        for endpoint in D1B_BLOCK_ENDPOINTS
                        if (output / f"ppo_block_{endpoint:03d}.pt").is_file()
                    ],
                    [50, 100],
                )

                completed = run_residual_d1b_from_datasets(
                    request,
                    output,
                    d1a,
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    dataset_version="synthetic-v1",
                    dataset_content_sha256="6" * 64,
                    runtime_environment={"environment_sha256": "7" * 64},
                    deadline_check=lambda: None,
                    progress=lambda _: None,
                )
                attempts_before_noop_resume = tuple(attempted_offsets)
                resumed_complete = run_residual_d1b_from_datasets(
                    request,
                    output,
                    d1a,
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    dataset_version="synthetic-v1",
                    dataset_content_sha256="6" * 64,
                    runtime_environment={"environment_sha256": "7" * 64},
                    deadline_check=lambda: None,
                    progress=lambda _: None,
                )

                binding = ResidualD1BStoreBinding(
                    root=output,
                    device=request.device,
                    observation_dim=12,
                    action_dim=96,
                    hidden_dim=D1_HIDDEN_DIM,
                    d1a_manifest_digest=verified.manifest_digest,
                    d1a_checkpoint_sha256=verified.checkpoint_sha256,
                    bc_policy_digest=verified.bc_policy_digest,
                    source_roles_digest=roles.digest,
                )
                resume = ResidualD1BBlockStore(binding).load_resume_state()

            self.assertEqual(completed["status"], "complete")
            self.assertEqual(resumed_complete, completed)
            self.assertEqual(tuple(attempted_offsets), attempts_before_noop_resume)
            self.assertEqual(committed_offsets, [0, 50, 100, 150])
            self.assertEqual(
                Counter(attempted_offsets),
                Counter({0: 1, 50: 1, 100: 2, 150: 1}),
            )
            self.assertIsNotNone(resume)
            self.assertEqual(resume.completed_episodes, 200)
            self.assertEqual(len(resume.blocks), 4)
            self.assertEqual(
                [block.episodes_completed for block in resume.blocks],
                list(D1B_BLOCK_ENDPOINTS),
            )
            self.assertEqual(completed["ppo_episodes_completed"], 200)
            self.assertEqual(completed["ppo_blocks_completed"], 4)
            self.assertEqual(
                [record["endpoint"] for record in completed["ppo_blocks"]],
                list(D1B_BLOCK_ENDPOINTS),
            )
            self.assertEqual(completed["source_model_calls"], 436)
            self.assertEqual(completed["target_calls"], 0)
            self.assertEqual(completed["hidden_target_calls"], 0)
            self.assertFalse(completed["hidden_target_evaluation_performed"])
            self.assertEqual(bc_policy.persistent_digest(), bc_digest_before)
            self.assertEqual(source_evaluator.call_count, 1)
            self.assertEqual(complete_verifier.call_count, 2)
            for index, endpoint in enumerate(D1B_BLOCK_ENDPOINTS):
                record = load_verified_json(
                    output / f"ppo_block_{endpoint:03d}.receipt.json"
                )
                self.assertEqual(record["hidden_target_calls"], 0)
                self.assertFalse(
                    record["authorizes_hidden_target_evaluation"]
                )
                parent = record["core_metadata"]["parent_checkpoint"]
                if index == 0:
                    self.assertIsNone(parent)
                else:
                    previous = load_verified_json(
                        output
                        / (
                            f"ppo_block_{D1B_BLOCK_ENDPOINTS[index - 1]:03d}"
                            ".receipt.json"
                        )
                    )
                    self.assertEqual(
                        parent["reference"],
                        previous["opaque_reference"],
                    )


if __name__ == "__main__":
    unittest.main()
