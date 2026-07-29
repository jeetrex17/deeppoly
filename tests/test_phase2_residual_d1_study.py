from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from rl_transfer.phase2_residual_d1 import ResidualD1Request
from rl_transfer.phase2_residual_d1_study import (
    ResidualD1StudyStages,
    run_residual_d1_study_from_datasets,
)
from rl_transfer.verified_artifacts import load_verified_json


def _request(root: Path, *, deadline_seconds: float = 28_800.0) -> ResidualD1Request:
    source = root / "source"
    source.mkdir()
    manifest = source / "screen_manifest.json"
    manifest.write_text('{"sealed":true}\n')
    data = root / "data"
    data.mkdir()
    return ResidualD1Request(
        source_manifest=manifest,
        source_root=source,
        output_dir=root / "study" / "d1a",
        data_root=data,
        deadline_seconds=deadline_seconds,
    )


def _d1a(*, passed: bool = True, status: str = "complete") -> dict[str, object]:
    return {
        "schema_version": 3,
        "name": "phase2-d1a-residual-ranker-bc",
        "status": status,
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
        "source_evaluation": {
            "classical_cnn": {
                "methods": {
                    "score_greedy": {
                        "total_target_calls": 2_500,
                        "max_total_target_calls": 50,
                        "hidden_target_calls": 0,
                    }
                }
            }
        },
        "d1_decision": {
            "passed": passed,
            "eligible_for_d1b_source_only_ppo": passed,
            "authorizes_hidden_target_evaluation": False,
        },
    }


def _d1b(*, status: str = "complete") -> dict[str, object]:
    return {
        "schema_version": 3,
        "name": "phase2-d1b-residual-ranker-ppo",
        "status": status,
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
        "source_evaluation": {
            "transformer": {
                "methods": {
                    "residual_ranker_bc_ppo": {
                        "total_target_calls": 2_500,
                        "max_total_target_calls": 50,
                        "hidden_target_calls": 0,
                    }
                }
            }
        },
    }


class FakeStages:
    def __init__(
        self,
        *,
        d1a_result: dict[str, object] | None = None,
        d1b_result: dict[str, object] | None = None,
    ) -> None:
        self.d1a_result = d1a_result or _d1a()
        self.d1b_result = d1b_result or _d1b()
        self.d1a_calls: list[dict[str, object]] = []
        self.d1b_calls: list[dict[str, object]] = []

    def run_d1a(self, *args: object, **kwargs: object) -> dict[str, object]:
        del args
        self.d1a_calls.append(dict(kwargs))
        kwargs["external_deadline_check"]()
        return dict(self.d1a_result)

    def run_d1b(self, *args: object, **kwargs: object) -> dict[str, object]:
        del args
        self.d1b_calls.append(dict(kwargs))
        kwargs["deadline_check"]()
        return dict(self.d1b_result)

    def stages(self) -> ResidualD1StudyStages:
        return ResidualD1StudyStages(
            run_d1a=self.run_d1a,
            run_d1b=self.run_d1b,
        )


def _runtime() -> dict[str, object]:
    return {"environment_sha256": "f" * 64}


class ResidualD1StudyTests(unittest.TestCase):
    def test_one_persisted_deadline_is_shared_across_both_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            harness = FakeStages()

            result = run_residual_d1_study_from_datasets(
                request,
                "train",
                "test",
                dataset_version="synthetic",
                dataset_content_sha256="e" * 64,
                runtime_environment=_runtime(),
                stages=harness.stages(),
                wall_clock=lambda: 1_000.0,
                monotonic_clock=lambda: 500.0,
                progress=lambda _: None,
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["started_epoch_seconds"], 1_000.0)
            self.assertEqual(result["deadline_epoch_seconds"], 29_800.0)
            self.assertEqual(len(harness.d1a_calls), 1)
            self.assertEqual(len(harness.d1b_calls), 1)
            self.assertIs(
                harness.d1a_calls[0]["external_deadline_check"],
                harness.d1b_calls[0]["deadline_check"],
            )
            persisted = load_verified_json(
                Path(directory) / "study" / "study_manifest.json"
            )
            self.assertEqual(persisted, result)
            self.assertEqual(result["target_calls"], 0)
            self.assertFalse(result["authorizes_hidden_target_evaluation"])

    def test_resume_preserves_original_start_and_never_resets_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            first = FakeStages(
                d1a_result=_d1a(passed=False, status="failed"),
            )
            initial = run_residual_d1_study_from_datasets(
                request,
                "train",
                "test",
                dataset_version="synthetic",
                dataset_content_sha256="e" * 64,
                runtime_environment=_runtime(),
                stages=first.stages(),
                wall_clock=lambda: 1_000.0,
                monotonic_clock=lambda: 10.0,
                progress=lambda _: None,
            )
            self.assertEqual(initial["status"], "failed")

            second = FakeStages()
            resumed = run_residual_d1_study_from_datasets(
                request,
                "train",
                "test",
                dataset_version="synthetic",
                dataset_content_sha256="e" * 64,
                runtime_environment=_runtime(),
                stages=second.stages(),
                wall_clock=lambda: 2_000.0,
                monotonic_clock=lambda: 20.0,
                progress=lambda _: None,
            )

            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["started_epoch_seconds"], 1_000.0)
            self.assertEqual(resumed["deadline_epoch_seconds"], 29_800.0)
            self.assertEqual(len(second.d1a_calls), 1)

    def test_resume_rejects_code_or_git_provenance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            first = FakeStages(
                d1a_result=_d1a(passed=False, status="failed"),
            )
            run_residual_d1_study_from_datasets(
                request,
                "train",
                "test",
                dataset_version="synthetic",
                dataset_content_sha256="e" * 64,
                runtime_environment=_runtime(),
                stages=first.stages(),
                wall_clock=lambda: 1_000.0,
                monotonic_clock=lambda: 10.0,
                progress=lambda _: None,
            )

            with (
                patch(
                    "rl_transfer.phase2_residual_d1_study.code_digest",
                    return_value="0" * 64,
                ),
                self.assertRaisesRegex(ValueError, "provenance|code|Git"),
            ):
                run_residual_d1_study_from_datasets(
                    request,
                    "train",
                    "test",
                    dataset_version="synthetic",
                    dataset_content_sha256="e" * 64,
                    runtime_environment=_runtime(),
                    stages=FakeStages().stages(),
                    wall_clock=lambda: 1_001.0,
                    monotonic_clock=lambda: 11.0,
                    progress=lambda _: None,
                )

    def test_d1a_negative_is_preserved_and_d1b_is_only_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            harness = FakeStages(
                d1a_result=_d1a(passed=False),
                d1b_result=_d1b(status="skipped"),
            )

            result = run_residual_d1_study_from_datasets(
                request,
                "train",
                "test",
                dataset_version="synthetic",
                dataset_content_sha256="e" * 64,
                runtime_environment=_runtime(),
                stages=harness.stages(),
                wall_clock=lambda: 1_000.0,
                monotonic_clock=lambda: 10.0,
                progress=lambda _: None,
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["study_outcome"], "valid_d1a_negative")
            self.assertEqual(len(harness.d1b_calls), 1)
            self.assertFalse(result["d1a_source_gate_passed"])

    def test_expired_deadline_stops_before_any_new_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root, deadline_seconds=10.0)
            first = FakeStages(
                d1a_result=_d1a(passed=False, status="failed"),
            )
            run_residual_d1_study_from_datasets(
                request,
                "train",
                "test",
                dataset_version="synthetic",
                dataset_content_sha256="e" * 64,
                runtime_environment=_runtime(),
                stages=first.stages(),
                wall_clock=lambda: 1_000.0,
                monotonic_clock=lambda: 1.0,
                progress=lambda _: None,
            )

            resumed = FakeStages()
            result = run_residual_d1_study_from_datasets(
                request,
                "train",
                "test",
                dataset_version="synthetic",
                dataset_content_sha256="e" * 64,
                runtime_environment=_runtime(),
                stages=resumed.stages(),
                wall_clock=lambda: 1_011.0,
                monotonic_clock=lambda: 2.0,
                progress=lambda _: None,
            )

            self.assertEqual(result["status"], "deadline_reached")
            self.assertFalse(resumed.d1a_calls)
            self.assertFalse(resumed.d1b_calls)
            self.assertEqual(result["deadline_epoch_seconds"], 1_010.0)

    def test_clock_rollback_and_tampered_or_unexpected_root_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            harness = FakeStages(
                d1a_result=_d1a(passed=False, status="failed"),
            )
            run_residual_d1_study_from_datasets(
                request,
                "train",
                "test",
                dataset_version="synthetic",
                dataset_content_sha256="e" * 64,
                runtime_environment=_runtime(),
                stages=harness.stages(),
                wall_clock=lambda: 1_000.0,
                monotonic_clock=lambda: 10.0,
                progress=lambda _: None,
            )

            with self.assertRaisesRegex(ValueError, "clock|rollback"):
                run_residual_d1_study_from_datasets(
                    request,
                    "train",
                    "test",
                    dataset_version="synthetic",
                    dataset_content_sha256="e" * 64,
                    runtime_environment=_runtime(),
                    stages=FakeStages().stages(),
                    wall_clock=lambda: 999.0,
                    monotonic_clock=lambda: 20.0,
                    progress=lambda _: None,
                )

            (root / "study" / "unrelated.txt").write_text("unsafe")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                run_residual_d1_study_from_datasets(
                    request,
                    "train",
                    "test",
                    dataset_version="synthetic",
                    dataset_content_sha256="e" * 64,
                    runtime_environment=_runtime(),
                    stages=FakeStages().stages(),
                    wall_clock=lambda: 1_001.0,
                    monotonic_clock=lambda: 20.0,
                    progress=lambda _: None,
                )

    def test_completed_study_revalidates_child_manifest_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            harness = FakeStages()
            run_residual_d1_study_from_datasets(
                request,
                "train",
                "test",
                dataset_version="synthetic",
                dataset_content_sha256="e" * 64,
                runtime_environment=_runtime(),
                stages=harness.stages(),
                wall_clock=lambda: 1_000.0,
                monotonic_clock=lambda: 10.0,
                progress=lambda _: None,
            )
            (root / "study" / "d1b" / "d1b_manifest.json").write_text("{}")

            with self.assertRaisesRegex(ValueError, "child|checksum|manifest"):
                run_residual_d1_study_from_datasets(
                    request,
                    "train",
                    "test",
                    dataset_version="synthetic",
                    dataset_content_sha256="e" * 64,
                    runtime_environment=_runtime(),
                    stages=FakeStages().stages(),
                    wall_clock=lambda: 1_001.0,
                    monotonic_clock=lambda: 11.0,
                    progress=lambda _: None,
                )


if __name__ == "__main__":
    unittest.main()
