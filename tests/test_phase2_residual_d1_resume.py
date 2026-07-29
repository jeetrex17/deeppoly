from __future__ import annotations

import tempfile
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from rl_transfer.phase2_residual_d1 import ResidualD1Request
from rl_transfer.phase2_residual_d1_runner import (
    _base_manifest,
    _existing_d1_manifest,
    _run_residual_d1_from_datasets,
    _verified_child,
)
from rl_transfer.verified_artifacts import write_verified_json


class ResidualD1ResumeTests(unittest.TestCase):
    def _request(self, root: Path) -> ResidualD1Request:
        source = root / "source"
        source.mkdir()
        data = root / "data"
        data.mkdir()
        output = root / "output"
        output.mkdir()
        return ResidualD1Request(
            source_manifest=source / "screen_manifest.json",
            source_root=source,
            output_dir=output,
            data_root=data,
        )

    def test_running_manifest_contains_the_complete_source_only_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(Path(directory))
            with patch(
                "rl_transfer.phase2_residual_d1_runner._validated_runtime_environment",
                return_value={"environment_sha256": "0" * 64},
            ):
                manifest = _base_manifest(
                    request,
                    dataset_version="synthetic",
                    dataset_content_sha256="1" * 64,
                    runtime_environment={},
                )

        self.assertEqual(manifest["target_calls"], 0)
        self.assertEqual(manifest["hidden_target_calls"], 0)
        self.assertFalse(manifest["target_evaluation_performed"])
        self.assertFalse(manifest["hidden_target_evaluation_performed"])
        self.assertFalse(manifest["authorizes_hidden_target_evaluation"])

    def test_expired_deadline_stops_before_source_context_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(Path(directory))
            ticks = iter((0.0, 28_800.0, 28_800.0))
            with (
                patch(
                    "rl_transfer.phase2_residual_d1_runner._base_manifest",
                    return_value={
                        "status": "running",
                        "target_calls": 0,
                        "hidden_target_calls": 0,
                        "target_evaluation_performed": False,
                        "hidden_target_evaluation_performed": False,
                        "authorizes_hidden_target_evaluation": False,
                    },
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_runner.load_d1_source_context"
                ) as source_loader,
                patch(
                    "rl_transfer.phase2_residual_d1_runner._gpu_memory_record",
                    return_value=None,
                ),
            ):
                result = _run_residual_d1_from_datasets(
                    request,
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    dataset_version="synthetic",
                    dataset_content_sha256="0" * 64,
                    runtime_environment={},
                    progress=lambda _: None,
                    clock=lambda: next(ticks),
                )

        self.assertEqual(result["status"], "deadline_reached")
        source_loader.assert_not_called()

    def test_verified_matching_manifest_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(Path(directory))
            self.assertIsNone(_existing_d1_manifest(request))
            expected = {
                "schema_version": 3,
                "status": "running",
                "request_sha256": request.digest(),
            }
            write_verified_json(
                request.output_dir / "d1_manifest.json",
                expected,
            )

            self.assertEqual(dict(_existing_d1_manifest(request) or {}), expected)

    def test_resume_rejects_mismatch_partial_pair_and_unknown_files(self) -> None:
        cases = ("mismatch", "partial", "unknown")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                request = self._request(Path(directory))
                if case == "mismatch":
                    write_verified_json(
                        request.output_dir / "d1_manifest.json",
                        {
                            "status": "running",
                            "request_sha256": "f" * 64,
                        },
                    )
                elif case == "partial":
                    (request.output_dir / "d1_manifest.json").write_text("{}")
                else:
                    (request.output_dir / "unrelated.txt").write_text("unsafe")

                with self.assertRaisesRegex(
                    ValueError,
                    "binding|incomplete|unexpected",
                ):
                    _existing_d1_manifest(request)

    def test_partial_promoted_artifacts_are_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self._request(Path(directory))
            write_verified_json(
                request.output_dir / "d1_manifest.json",
                {
                    "schema_version": 3,
                    "status": "running",
                    "request_sha256": request.digest(),
                },
            )
            checkpoint = request.output_dir / "residual_ranker_bc.pt"
            checkpoint.write_bytes(b"preserve-me")
            checkpoint.with_suffix(".pt.sha256").write_text("0" * 64 + "\n")

            with self.assertRaisesRegex(ValueError, "cannot be overwritten"):
                _existing_d1_manifest(request)
            self.assertEqual(checkpoint.read_bytes(), b"preserve-me")

    def test_completed_child_requires_content_and_matching_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "child.jsonl"
            path.write_text('{"safe":true}\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".jsonl.sha256").write_text(digest + "\n")

            _verified_child(path, digest)
            path.write_text('{"safe":false}\n')
            with self.assertRaisesRegex(ValueError, "checksum"):
                _verified_child(path, digest)

    def test_resume_and_child_verification_reject_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root)
            outside = root / "outside.json"
            outside.write_text("{}")
            (request.output_dir / "d1_manifest.json").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _existing_d1_manifest(request)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.jsonl"
            outside.write_text('{"safe":true}\n')
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            path = root / "child.jsonl"
            path.symlink_to(outside)
            path.with_suffix(".jsonl.sha256").write_text(digest + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                _verified_child(path, digest)


if __name__ == "__main__":
    unittest.main()
