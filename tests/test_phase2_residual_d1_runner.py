from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from rl_transfer.artifacts import sha256_file
from rl_transfer.phase2_residual_d1 import ResidualD1Request
from rl_transfer.phase2_residual_d1_runner import (
    _existing_d1_manifest,
    _run_residual_d1_from_datasets,
    run_residual_d1_from_datasets,
)
from rl_transfer.results import ResearchResultRow


SOURCE_FAMILIES = ("classical_cnn", "transformer")
METHODS = ("score_greedy", "residual_ranker_bc")


def _runtime() -> dict[str, object]:
    payload: dict[str, object] = {
        "python_version": "3.12.3",
        "torch_version": "test",
        "torchvision_version": "test",
        "cuda_runtime_version": "test",
        "cudnn_version": 1,
        "gpu_name": "synthetic",
        "gpu_total_memory_bytes": 1,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return {**payload, "environment_sha256": digest}


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

    def persistent_digest(self) -> str:
        return "c" * 64


def _write_child(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def _rows_and_traces() -> tuple[
    list[ResearchResultRow],
    list[dict[str, object]],
]:
    rows: list[ResearchResultRow] = []
    traces: list[dict[str, object]] = []
    for family in SOURCE_FAMILIES:
        victim_id = f"{family}-evaluation"
        for method in METHODS:
            sample_id = f"cifar10:{family}:{victim_id}:350"
            row = ResearchResultRow(
                sample_id=sample_id,
                victim_id=victim_id,
                victim_family=family,
                method=method,
                threat_model="T1",
                seed=17,
                query_budget=50,
                clean_correct=True,
                success=True,
                query_to_success=2,
                total_target_calls=2,
                linf=2 / 255,
                l2=0.02,
                policy_digest="c" * 64,
                action_trace=(3,),
            )
            rows.append(row)
            traces.append(
                {
                    "method": method,
                    "sample_id": sample_id,
                    "victim_id": victim_id,
                    "victim_family": family,
                    "total_target_calls": 2,
                    "query_trace": [
                        {
                            "call_index": 1,
                            "sample_id": sample_id,
                            "victim_id": victim_id,
                            "feedback": "scores",
                            "purpose": "initialization",
                            "error": None,
                        },
                        {
                            "call_index": 2,
                            "sample_id": sample_id,
                            "victim_id": victim_id,
                            "feedback": "scores",
                            "purpose": "residual-ranker",
                            "error": None,
                        },
                    ],
                    "target_calls": 0,
                    "hidden_target_calls": 0,
                }
            )
    return rows, traces


class ResidualD1RunnerLifecycleTests(unittest.TestCase):
    def _request(self, root: Path) -> ResidualD1Request:
        source = root / "source"
        source.mkdir()
        manifest = source / "screen_manifest.json"
        manifest.write_text('{"sealed":true}\n')
        data = root / "data"
        data.mkdir()
        output = root / "study" / "d1a"
        output.mkdir(parents=True)
        return ResidualD1Request(
            source_manifest=manifest,
            source_root=source,
            output_dir=output,
            data_root=data,
        )

    def test_complete_d1a_lifecycle_writes_and_reverifies_all_children(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root)
            attack = SimpleNamespace(
                recurrent_observation_dim=12,
                action_dim=96,
                grid_size=4,
            )
            config = SimpleNamespace(
                attack_config=lambda: attack,
                policy_learning_rate=0.001,
                policy_entropy_weight=0.01,
            )
            context = SimpleNamespace(
                config=config,
                role_audit={
                    "pairwise_disjoint": True,
                    "role_sizes": {
                        "train": 200,
                        "threshold": 50,
                        "competence": 50,
                        "d1a_evaluation": 50,
                        "d1b_evaluation": 50,
                    },
                    "role_indices_sha256": {
                        role: str(offset) * 64
                        for offset, role in enumerate(
                            (
                                "train",
                                "threshold",
                                "competence",
                                "d1a_evaluation",
                                "d1b_evaluation",
                            ),
                            start=1,
                        )
                    },
                },
                source_manifest_sha256=sha256_file(request.source_manifest),
            )
            backbone = FakeBackbone()
            saved_metadata: dict[str, object] = {}

            def teacher(*args: object, **kwargs: object):
                del args, kwargs
                examples = _write_child(
                    request.output_dir / "teacher_ranker_examples.jsonl",
                    b"{}\n",
                )
                metadata = _write_child(
                    request.output_dir / "teacher_ranker_manifest.json",
                    b"{}\n",
                )
                return (
                    (),
                    (),
                    (),
                    {
                        "examples_sha256": examples,
                        "metadata_sha256": metadata,
                        "roles": {
                            role: {"source_calls": calls}
                            for role, calls in (
                                ("train", 20),
                                ("threshold_selection", 10),
                                ("competence_gate", 10),
                            )
                        },
                    },
                )

            def save_checkpoint(
                path: Path,
                supplied_backbone: object,
                metadata: dict[str, object],
            ) -> str:
                self.assertIs(supplied_backbone, backbone)
                saved_metadata.update(metadata)
                return _write_child(path, b"synthetic-checkpoint")

            def fake_plots(
                output: Path,
                verification: object,
            ) -> tuple[Path, Path]:
                del verification
                paths = output / "asr_by_query.svg", output / "final_asr.svg"
                for path in paths:
                    _write_child(path, b"<svg/>")
                return paths

            rows, traces = _rows_and_traces()
            conditions = {
                family: {"methods": {method: {} for method in METHODS}}
                for family in SOURCE_FAMILIES
            }
            clock_values = itertools.count()
            external_deadline_calls = 0

            def external_deadline_check() -> None:
                nonlocal external_deadline_calls
                external_deadline_calls += 1

            with (
                patch(
                    "rl_transfer.phase2_residual_d1_runner.torch.cuda.is_available",
                    return_value=False,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_runner.load_d1_source_context",
                    return_value=context,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_runner._teacher_examples",
                    side_effect=teacher,
                ) as teacher_mock,
                patch(
                    "rl_transfer.phase2_residual_d1_runner.RecurrentAttackPolicy",
                    return_value=backbone,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_runner.fit_residual_ranker_bc",
                    return_value={"epochs": 12, "loss": 0.1},
                ) as bc_mock,
                patch(
                    "rl_transfer.phase2_residual_d1_runner.select_confidence_threshold",
                    return_value={
                        "threshold": 0.2,
                        "overrides_enabled": True,
                    },
                ) as threshold_mock,
                patch(
                    "rl_transfer.phase2_residual_d1_runner.ResidualRankerPolicy",
                    FakeResidualPolicy,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_runner"
                    ".evaluate_residual_ranker_examples",
                    return_value={"accepted_steps": 20},
                ) as competence_mock,
                patch(
                    "rl_transfer.phase2_residual_d1_runner.save_recurrent_checkpoint",
                    side_effect=save_checkpoint,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_runner.load_recurrent_checkpoint",
                    side_effect=lambda *args, **kwargs: (
                        backbone,
                        dict(saved_metadata),
                    ),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_runner.evaluate_residual_d1",
                    return_value=(conditions, rows, traces),
                ) as evaluator_mock,
                patch(
                    "rl_transfer.phase2_residual_d1_runner.verify_d1_raw_evidence",
                    return_value={"verified": True},
                ) as raw_verifier,
                patch(
                    "rl_transfer.phase2_residual_d1_runner.verify_d1_recorded_summaries"
                ) as summary_verifier,
                patch(
                    "rl_transfer.phase2_residual_d1_runner.write_d1_evidence_plots",
                    side_effect=fake_plots,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_runner.paired_source_statistics",
                    return_value={"paired": True},
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_runner._decision",
                    return_value={
                        "passed": True,
                        "eligible_for_d1b_source_only_ppo": True,
                        "authorizes_hidden_target_evaluation": False,
                    },
                ),
            ):
                result = _run_residual_d1_from_datasets(
                    request,
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    dataset_version="synthetic",
                    dataset_content_sha256="d" * 64,
                    runtime_environment=_runtime(),
                    progress=lambda _: None,
                    clock=lambda: float(next(clock_values)),
                    external_deadline_check=external_deadline_check,
                )
                resumed = _existing_d1_manifest(request)
                with patch(
                    "rl_transfer.phase2_residual_d1_runner"
                    "._run_residual_d1_from_datasets",
                    side_effect=AssertionError("completed D1a must not rerun"),
                ):
                    public_resumed = run_residual_d1_from_datasets(
                        request,
                        object(),  # type: ignore[arg-type]
                        object(),  # type: ignore[arg-type]
                        dataset_version="synthetic",
                        dataset_content_sha256="d" * 64,
                        runtime_environment=_runtime(),
                        progress=lambda _: None,
                        clock=lambda: float(next(clock_values)),
                        external_deadline_check=external_deadline_check,
                    )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(resumed, result)
            self.assertEqual(public_resumed, result)
            self.assertEqual(result["hidden_target_calls"], 0)
            self.assertEqual(result["source_model_calls"], 48)
            self.assertEqual(
                set(result["figures"]),
                {"asr_by_query.svg", "final_asr.svg"},
            )
            self.assertGreaterEqual(external_deadline_calls, 4)
            propagated = (
                teacher_mock.call_args.kwargs["deadline_check"],
                bc_mock.call_args.kwargs["deadline_check"],
                threshold_mock.call_args.kwargs["deadline_check"],
                competence_mock.call_args.kwargs["deadline_check"],
                evaluator_mock.call_args.kwargs["deadline_check"],
            )
            self.assertTrue(all(item is propagated[0] for item in propagated))
            self.assertEqual(saved_metadata["schema_version"], 3)
            self.assertEqual(
                saved_metadata["kind"],
                "phase2_d1a_source_only_residual_ranker_bc",
            )
            self.assertEqual(
                saved_metadata["source_manifest_sha256"],
                sha256_file(request.source_manifest),
            )
            self.assertEqual(saved_metadata["request_sha256"], request.digest())
            self.assertEqual(saved_metadata["target_calls"], 0)
            self.assertEqual(saved_metadata["hidden_target_calls"], 0)
            self.assertFalse(saved_metadata["target_evaluation_performed"])
            self.assertFalse(saved_metadata["hidden_target_evaluation_performed"])
            self.assertFalse(saved_metadata["target_evaluation_available"])
            self.assertFalse(saved_metadata["authorizes_hidden_target_evaluation"])
            self.assertEqual(raw_verifier.call_count, 4)
            first_verified_rows = raw_verifier.call_args_list[0].args[0]
            self.assertIsInstance(first_verified_rows[0]["action_trace"], list)
            self.assertEqual(summary_verifier.call_count, 4)


if __name__ == "__main__":
    unittest.main()
