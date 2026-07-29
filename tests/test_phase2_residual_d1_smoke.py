from __future__ import annotations

import itertools
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl_transfer.artifacts import sha256_file
from rl_transfer.phase2_residual_d1 import ResidualD1Request
from rl_transfer.phase2_residual_d1_smoke import (
    build_residual_d1_smoke_plan,
    run_residual_d1_gpu_smoke,
)
from rl_transfer.results import ResearchResultRow
from rl_transfer.verified_artifacts import load_verified_json


SOURCE_FAMILIES = ("classical_cnn", "transformer")


class FakeBackbone:
    def to(self, device: object) -> FakeBackbone:
        del device
        return self

    def persistent_digest(self) -> str:
        return "b" * 64


class FakeResidualPolicy:
    def __init__(self, backbone: FakeBackbone, **kwargs: object) -> None:
        del kwargs
        self.backbone = backbone


class ResidualD1SmokeContractTests(unittest.TestCase):
    def _request(self, root: Path) -> ResidualD1Request:
        source = root / "source"
        data = root / "data"
        output = root / "smoke"
        source.mkdir()
        data.mkdir()
        return ResidualD1Request(
            source_manifest=source / "screen_manifest.json",
            source_root=source,
            output_dir=output,
            data_root=data,
        )

    def test_smoke_plan_is_small_source_only_and_never_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = build_residual_d1_smoke_plan(self._request(Path(directory)))

        self.assertTrue(plan["smoke_only"])
        self.assertFalse(plan["research_valid"])
        self.assertEqual(plan["teacher_episodes"], 10)
        self.assertEqual(plan["bc_epochs"], 1)
        self.assertEqual(plan["target_calls"], 0)
        self.assertEqual(plan["hidden_target_calls"], 0)
        self.assertFalse(plan["target_evaluation_performed"])
        self.assertFalse(plan["hidden_target_evaluation_performed"])
        self.assertFalse(plan["authorizes_d1_promotion"])
        self.assertFalse(plan["authorizes_hidden_target_evaluation"])

    def test_real_smoke_refuses_non_cuda_machine_before_data_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(Path(directory))
            with (
                patch(
                    "rl_transfer.phase2_residual_d1_smoke.torch.cuda.is_available",
                    return_value=False,
                ),
                self.assertRaisesRegex(RuntimeError, "CUDA|GPU"),
            ):
                run_residual_d1_gpu_smoke(
                    request,
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    dataset_content_sha256="0" * 64,
                    runtime_environment={},
                )
            self.assertFalse(request.output_dir.exists())

    def test_synthetic_gpu_smoke_exercises_the_complete_source_only_lifecycle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(Path(directory))
            attack = SimpleNamespace(
                recurrent_observation_dim=12,
                action_dim=96,
                grid_size=4,
            )
            context = SimpleNamespace(
                config=SimpleNamespace(
                    attack_config=lambda: attack,
                    policy_learning_rate=0.001,
                    policy_entropy_weight=0.01,
                ),
                teacher_victims={
                    family: ((f"{family}-teacher", object()),)
                    for family in SOURCE_FAMILIES
                },
                train_samples=tuple(object() for _ in range(10)),
                train_indices=tuple(range(10)),
            )
            backbone = FakeBackbone()
            checkpoint_metadata: dict[str, object] = {}

            def save_checkpoint(
                path: Path,
                supplied_backbone: object,
                metadata: dict[str, object],
            ) -> str:
                self.assertIs(supplied_backbone, backbone)
                checkpoint_metadata.update(metadata)
                path.write_bytes(b"synthetic-smoke-checkpoint")
                digest = sha256_file(path)
                path.with_suffix(".pt.sha256").write_text(digest + "\n")
                return digest

            def family_result(
                family: str,
            ) -> tuple[
                dict[str, object],
                list[ResearchResultRow],
                list[dict[str, object]],
            ]:
                victim_id = f"{family}-teacher"
                sample_id = f"cifar10:{family}:{victim_id}:0"
                row = ResearchResultRow(
                    sample_id=sample_id,
                    victim_id=victim_id,
                    victim_family=family,
                    method="smoke_residual_fallback",
                    threat_model="T1",
                    seed=95_017,
                    query_budget=50,
                    clean_correct=True,
                    success=False,
                    query_to_success=None,
                    total_target_calls=1,
                    linf=0.0,
                    l2=0.0,
                    policy_digest="b" * 64,
                    action_trace=(),
                )
                trace = {
                    "method": row.method,
                    "sample_id": sample_id,
                    "victim_id": victim_id,
                    "victim_family": family,
                    "family": family,
                    "heldout_family": "modern_cnn",
                    "source_slice": "exact_source",
                    "total_target_calls": 1,
                    "query_trace": [
                        {
                            "call_index": 1,
                            "sample_id": sample_id,
                            "victim_id": victim_id,
                            "feedback": "scores",
                            "purpose": "initialization",
                            "error": None,
                        }
                    ],
                    "target_calls": 0,
                    "hidden_target_calls": 0,
                }
                condition = {
                    "family": family,
                    "source_model_calls": 1,
                    "target_calls": 0,
                    "hidden_target_calls": 0,
                }
                return condition, [row], [trace]

            runtime = {
                "environment_sha256": "e" * 64,
                "gpu_name": "synthetic",
            }
            clock_values = itertools.count()
            family_outputs = [family_result(family) for family in SOURCE_FAMILIES]
            with (
                patch(
                    "rl_transfer.phase2_residual_d1_smoke.torch.cuda.is_available",
                    return_value=True,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_smoke"
                    "._validated_runtime_environment",
                    return_value=runtime,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_smoke.load_d1_source_context",
                    return_value=context,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_smoke._collect_teacher_blocks",
                    return_value=(
                        (),
                        {
                            "episodes": 10,
                            "source_model_calls": 20,
                            "target_calls": 0,
                            "hidden_target_calls": 0,
                        },
                    ),
                ) as teacher_mock,
                patch(
                    "rl_transfer.phase2_residual_d1_smoke.RecurrentAttackPolicy",
                    return_value=backbone,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_smoke.fit_residual_ranker_bc",
                    return_value={
                        "epochs": 1,
                        "loss": 0.1,
                        "target_calls": 0,
                    },
                ) as bc_mock,
                patch(
                    "rl_transfer.phase2_residual_d1_smoke.save_recurrent_checkpoint",
                    side_effect=save_checkpoint,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_smoke.load_recurrent_checkpoint",
                    side_effect=lambda *args, **kwargs: (
                        backbone,
                        dict(checkpoint_metadata),
                    ),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_smoke.ResidualRankerPolicy",
                    FakeResidualPolicy,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_smoke"
                    ".evaluate_residual_policy_cohort",
                    side_effect=family_outputs,
                ) as evaluator,
            ):
                result = run_residual_d1_gpu_smoke(
                    request,
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    dataset_content_sha256="d" * 64,
                    runtime_environment=runtime,
                    progress=lambda _: None,
                    clock=lambda: float(next(clock_values)),
                )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["result_rows"], 2)
            self.assertEqual(result["query_traces"], 2)
            self.assertEqual(result["target_calls"], 0)
            self.assertEqual(result["hidden_target_calls"], 0)
            self.assertFalse(result["authorizes_d1_promotion"])
            self.assertEqual(
                load_verified_json(request.output_dir / "smoke_manifest.json"),
                result,
            )
            self.assertEqual(teacher_mock.call_args.kwargs["role"], "gpu_smoke_train")
            self.assertIs(
                teacher_mock.call_args.kwargs["deadline_check"],
                bc_mock.call_args.kwargs["deadline_check"],
            )
            self.assertEqual(evaluator.call_count, 2)
            for call in evaluator.call_args_list:
                self.assertIs(
                    call.kwargs["deadline_check"],
                    teacher_mock.call_args.kwargs["deadline_check"],
                )
            self.assertEqual(checkpoint_metadata["target_calls"], 0)
            self.assertEqual(checkpoint_metadata["hidden_target_calls"], 0)
            self.assertFalse(checkpoint_metadata["hidden_target_evaluation_performed"])
            self.assertTrue((request.output_dir / "smoke_results.jsonl").is_file())
            self.assertTrue((request.output_dir / "smoke_query_traces.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
