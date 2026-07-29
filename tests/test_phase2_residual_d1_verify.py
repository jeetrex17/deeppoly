from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from rl_transfer.artifacts import sha256_file
from rl_transfer.phase2_residual_d1 import ResidualD1Request
from rl_transfer.phase2_residual_d1_verify import (
    _study_file_digests,
    _verify_current_provenance,
    _verify_residual_d1_study_locked,
    _write_locked_package,
    verify_residual_d1_study,
)
from rl_transfer.verified_artifacts import write_verified_json


class ResidualD1DeepVerificationTests(unittest.TestCase):
    def test_public_verifier_holds_the_study_lock_for_the_full_audit(self) -> None:
        events: list[str] = []

        @contextmanager
        def fake_lock(path: Path):
            events.append(f"lock:{path.name}")
            yield
            events.append("unlock")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source_manifest = source / "screen_manifest.json"
            source_manifest.write_text("{}")
            data = root / "data"
            data.mkdir()
            with (
                patch(
                    "rl_transfer.phase2_residual_d1_verify.exclusive_file_lock",
                    side_effect=fake_lock,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_verify"
                    "._verify_residual_d1_study_locked",
                    side_effect=lambda *args: (
                        events.append("verify") or {"status": "verified"}
                    ),
                ),
            ):
                result = verify_residual_d1_study(
                    root,
                    source_manifest,
                    source,
                    data,
                )

        self.assertEqual(result["status"], "verified")
        self.assertEqual(events, ["lock:.study.lock", "verify", "unlock"])

    def test_locked_package_matches_verified_snapshot_and_is_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "study"
            (root / "nested").mkdir(parents=True)
            (root / ".study.lock").write_bytes(b"")
            (root / "manifest.json").write_text('{"status":"complete"}\n')
            (root / "nested" / "rows.jsonl").write_text('{"sample":1}\n')
            expected = _study_file_digests(root)
            first = _write_locked_package(
                root,
                parent / "first.tar.gz",
                parent / "first.SHA256SUMS",
                expected,
            )
            second = _write_locked_package(
                root,
                parent / "second.tar.gz",
                parent / "second.SHA256SUMS",
                expected,
            )

            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            self.assertEqual(
                (parent / "first.SHA256SUMS").read_text(),
                (parent / "second.SHA256SUMS").read_text(),
            )
            with tarfile.open(parent / "first.tar.gz", "r:gz") as archive:
                names = set(archive.getnames())
            self.assertEqual(
                names,
                {
                    "study/.study.lock",
                    "study/manifest.json",
                    "study/nested/rows.jsonl",
                },
            )
            self.assertEqual(first["artifact_file_count"], 3)
            self.assertTrue((parent / "first.tar.gz.sha256").is_file())

    def test_locked_package_rejects_a_changed_verified_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "study"
            root.mkdir()
            (root / ".study.lock").write_bytes(b"")
            (root / "manifest.json").write_text("{}")
            expected = _study_file_digests(root)
            changed = {**expected, "manifest.json": "0" * 64}
            with (
                patch(
                    "rl_transfer.phase2_residual_d1_verify._study_file_digests",
                    return_value=changed,
                ),
                self.assertRaisesRegex(ValueError, "snapshot"),
            ):
                _write_locked_package(
                    root,
                    parent / "study.tar.gz",
                    parent / "study.SHA256SUMS",
                    expected,
                )

    def test_locked_deep_verifier_traverses_skipped_and_complete_d1b(
        self,
    ) -> None:
        for d1b_status in ("skipped", "complete"):
            with self.subTest(d1b_status=d1b_status):
                with tempfile.TemporaryDirectory() as directory:
                    parent = Path(directory)
                    root = parent / "study"
                    d1a_root = root / "d1a"
                    d1b_root = root / "d1b"
                    d1a_root.mkdir(parents=True)
                    d1b_root.mkdir()
                    source = parent / "source"
                    source.mkdir()
                    source_manifest = source / "screen_manifest.json"
                    source_manifest.write_text('{"sealed":true}\n')
                    data = parent / "data"
                    data.mkdir()
                    request = ResidualD1Request(
                        source_manifest=source_manifest,
                        source_root=source,
                        output_dir=d1a_root,
                        data_root=data,
                    )
                    runtime = {"environment_sha256": "e" * 64}
                    passed = d1b_status == "complete"
                    seal = {
                        "target_calls": 0,
                        "hidden_target_calls": 0,
                        "target_evaluation_performed": False,
                        "hidden_target_evaluation_performed": False,
                        "target_evaluation_available": False,
                        "authorizes_hidden_target_evaluation": False,
                    }
                    study = {
                        "schema_version": 1,
                        "status": "complete",
                        "deadline_seconds": 28_800.0,
                        "request_sha256": request.digest(),
                        "study_outcome": (
                            "d1b_complete" if passed else "d1a_gate_failed"
                        ),
                        **seal,
                    }
                    d1a = {
                        "schema_version": 3,
                        "status": "complete",
                        "runtime_environment": runtime,
                        "d1_decision": {
                            "passed": passed,
                            "eligible_for_d1b_source_only_ppo": passed,
                            "authorizes_hidden_target_evaluation": False,
                        },
                        **seal,
                    }
                    d1b = {
                        "schema_version": 3,
                        "status": d1b_status,
                        "runtime_environment": runtime,
                        **seal,
                    }
                    write_verified_json(root / "study_manifest.json", study)
                    write_verified_json(d1a_root / "d1_manifest.json", d1a)
                    write_verified_json(d1b_root / "d1b_manifest.json", d1b)
                    attack = SimpleNamespace(
                        recurrent_observation_dim=12,
                        action_dim=96,
                    )
                    context = SimpleNamespace(
                        config=SimpleNamespace(attack_config=lambda: attack)
                    )
                    roles = SimpleNamespace(digest="d" * 64)
                    verified_d1a = SimpleNamespace(
                        manifest_digest="a" * 64,
                        checkpoint_sha256="b" * 64,
                        bc_policy_digest="c" * 64,
                    )
                    complete_resume = SimpleNamespace(
                        completed_episodes=200,
                        blocks=(object(), object(), object(), object()),
                    )
                    provenance = {
                        "git_revision": "revision",
                        "code_digest": "1" * 64,
                        "dataset_content_sha256": "2" * 64,
                        "source_manifest_sha256": "3" * 64,
                        "runtime_environment_sha256": "e" * 64,
                        "worktree_clean": True,
                    }

                    with ExitStack() as stack:
                        stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                ".torch.cuda.is_available",
                                return_value=True,
                            )
                        )
                        stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify._validate_root"
                            )
                        )
                        stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                "._verify_current_provenance",
                                return_value=provenance,
                            )
                        )
                        study_children = stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                "._verify_complete_children"
                            )
                        )
                        d1a_children = stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                "._verify_complete_d1_children"
                            )
                        )
                        stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify.seed_everything"
                            )
                        )
                        stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                "._runtime_environment",
                                return_value=runtime,
                            )
                        )
                        dataset_loader = stack.enter_context(
                            patch(
                                "torchvision.datasets.CIFAR10",
                                side_effect=("train-dataset", "test-dataset"),
                            )
                        )
                        stack.enter_context(
                            patch(
                                "torchvision.transforms.ToTensor",
                                return_value=object(),
                            )
                        )
                        context_loader = stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                ".load_d1_source_context",
                                return_value=context,
                            )
                        )
                        stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify._cache_binding",
                                return_value=("binding", "protocol"),
                            )
                        )
                        stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                ".load_residual_teacher_cache",
                                return_value=object(),
                            )
                        )
                        stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                ".build_d1b_source_roles",
                                return_value=roles,
                            )
                        )
                        stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                ".verify_d1a_artifacts",
                                return_value=verified_d1a,
                            )
                        )
                        d1b_output_validator = stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify._validate_output"
                            )
                        )
                        d1b_children = stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                ".verify_complete_d1b_children"
                            )
                        )
                        store_type = stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                ".ResidualD1BBlockStore"
                            )
                        )
                        final_ppo = stack.enter_context(
                            patch(
                                "rl_transfer.phase2_residual_d1_verify"
                                "._verify_final_ppo",
                                return_value="f" * 64,
                            )
                        )
                        store_type.return_value.load_resume_state.return_value = (
                            complete_resume
                        )
                        result = _verify_residual_d1_study_locked(
                            root,
                            source_manifest,
                            source,
                            data,
                        )

                    self.assertEqual(result["status"], "verified")
                    self.assertEqual(result["d1a_gate_passed"], passed)
                    self.assertEqual(result["d1b_status"], d1b_status)
                    self.assertEqual(
                        result["d1b_verified_blocks"],
                        4 if passed else 0,
                    )
                    self.assertEqual(
                        result["ppo_policy_digest"],
                        "f" * 64 if passed else None,
                    )
                    self.assertEqual(result["target_calls"], 0)
                    self.assertEqual(result["hidden_target_calls"], 0)
                    self.assertIn("artifact_tree_sha256", result)
                    study_children.assert_called_once()
                    d1a_children.assert_called_once()
                    self.assertEqual(dataset_loader.call_count, 2)
                    context_loader.assert_called_once()
                    if passed:
                        d1b_output_validator.assert_called_once()
                        d1b_children.assert_called_once()
                        store_type.assert_called_once()
                        final_ppo.assert_called_once()
                    else:
                        d1b_output_validator.assert_not_called()
                        d1b_children.assert_not_called()
                        store_type.assert_not_called()
                        final_ppo.assert_not_called()

    def test_current_provenance_fails_closed_on_runtime_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source_manifest = source / "screen_manifest.json"
            source_manifest.write_text("{}\n")
            data = root / "data"
            data.mkdir()
            output = root / "study" / "d1a"
            request = ResidualD1Request(
                source_manifest=source_manifest,
                source_root=source,
                output_dir=output,
                data_root=data,
            )
            study = {
                "code_digest": "a" * 64,
                "git_revision": "revision",
                "dataset_content_sha256": "b" * 64,
                "source_manifest_sha256": sha256_file(source_manifest),
                "runtime_environment_sha256": "c" * 64,
            }
            with (
                patch(
                    "rl_transfer.phase2_residual_d1_verify.git_worktree_state",
                    return_value={"dirty": False},
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_verify.code_digest",
                    return_value="a" * 64,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_verify.git_revision",
                    return_value="revision",
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_verify.tree_digest",
                    return_value="b" * 64,
                ),
                self.assertRaisesRegex(ValueError, "runtime provenance"),
            ):
                _verify_current_provenance(
                    study,
                    request,
                    {"environment_sha256": "d" * 64},
                )

if __name__ == "__main__":
    unittest.main()
