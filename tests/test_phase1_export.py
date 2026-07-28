import csv
import gzip
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from rl_transfer.phase1_export import export_phase1_evidence
from rl_transfer.phase1_export_validation import validate_portable_value
from rl_transfer.verified_artifacts import write_verified_json


_FINGERPRINT = hashlib.sha256(b"phase-1-fixture").hexdigest()
_LEARNED = "gradient_bc_groupdro_ppo_stochastic"
_CONTROL = "score_greedy"


def _metrics(asr: float, auc: float, entropy: float) -> dict[str, object]:
    return {
        "eligible": 100,
        "successes": round(asr * 100),
        "asr_at_budgets": {"0": 0.0, "25": asr / 2, "50": asr},
        "asr_query_auc": auc,
        "normalized_action_entropy": entropy,
        "query_budget": 50,
        "max_total_target_calls": 50,
        "frozen": True,
        "initialization_included": True,
        "by_victim": {
            "victim-0": {
                "eligible": 100,
                "successes": round(asr * 100),
                "asr_at_budgets": {"0": 0.0, "50": asr},
                "asr_query_auc": auc,
            }
        },
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    study_root = root / "source"
    freeze = (
        "torch==2.13.0\n"
        "torchvision==0.28.0\n"
        "-e git+https://github.com/jeetrex17/deeppoly.git@"
        "47bd57e9c6826a9e09203de2adacef64a75ace4e"
        "#egg=rl_transfer_research\n"
    )
    study_root.mkdir()
    (study_root / "pip_freeze.txt").write_text(freeze)
    run_dir = study_root / "runs" / _FINGERPRINT[:12]
    run_dir.mkdir(parents=True)
    results = run_dir / "source_results.jsonl"
    traces = run_dir / "source_query_traces.jsonl"
    results.write_text(
        '{"method":"learned","success":true,"victim_family":"modern_cnn"}\n'
    )
    traces.write_text('{"method":"learned","actions":[1,2],"family":"modern_cnn"}\n')
    evaluation = {
        "exact_source": {
            "modern_cnn": {
                _LEARNED: _metrics(0.12, 0.06, 0.70),
                _CONTROL: _metrics(0.05, 0.025, 0.25),
            }
        },
        "seen_family_new_instance": {
            "modern_cnn": {
                _LEARNED: _metrics(0.10, 0.05, 0.71),
                _CONTROL: _metrics(0.04, 0.020, 0.24),
            }
        },
    }
    source_cache = {
        "binding": {
            "code_digest": "a" * 64,
            "config_digest": "b" * 64,
            "data_role_digests": {"source_gate": "c" * 64},
            "policy_checkpoints": {
                "main": {
                    "path": "/home/private/research/policy.pt",
                    "sha256": "d" * 64,
                }
            },
            "source_victim_checkpoints": {"victim-0": "e" * 64},
            "split_digest": "f" * 64,
        },
        "evaluation_elapsed_seconds": 12.5,
        "query_traces_sha256": hashlib.sha256(traces.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
        "source_evaluation": evaluation,
    }
    write_verified_json(run_dir / "source_evaluation.json", source_cache)
    behavior_cloning = {
        "elapsed_seconds": 4.0,
        "enabled": True,
        "fit": {
            "accepted_steps": 1200,
            "epochs": 3,
            "final_accuracy": 0.04,
            "final_loss": 4.30,
            "frequency_nll": 4.28,
            "majority_accuracy": 0.035,
            "uniform_accuracy": 1 / 96,
            "uniform_nll": 4.564348191467836,
        },
        "gate": {"passed": False},
        "validation": {
            "accepted_steps": 200,
            "accuracy": 0.038,
            "frequency_nll": 4.31,
            "majority_accuracy": 0.036,
            "nll": 4.34,
            "uniform_accuracy": 1 / 96,
            "uniform_nll": 4.564348191467836,
        },
    }
    run = {
        "schema_version": 1,
        "name": "fixture-modern-seed-17",
        "fingerprint": _FINGERPRINT,
        "run_dir": "/home/private/research/runs/" + _FINGERPRINT[:12],
        "status": "source_complete",
        "seed": 17,
        "target_family": "classical_cnn",
        "source_families": ["modern_cnn"],
        "elapsed_seconds": 100.0,
        "source_evaluation_elapsed_seconds": 12.5,
        "research_valid": False,
        "target_calls": 0,
        "target_evaluation_performed": False,
        "config_digest": "b" * 64,
        "split_digest": "f" * 64,
        "data_role_digests": {"source_gate": "c" * 64},
        "victim_bank_digest": "1" * 64,
        "victim_accuracy_gate": {"passed": True},
        "victim_instances": {
            "classical_cnn": [{"victim_id": "held-out-victim"}],
            "modern_cnn": [{"victim_id": "source-victim"}],
        },
        "policy": {
            "checkpoint": "/home/private/research/policy.pt",
            "checkpoint_sha256": "d" * 64,
            "training": {
                "episodes": 100,
                "trained_episodes": 90,
                "source_calls": 4000,
                "behavior_cloning": behavior_cloning,
            },
        },
        "source_competence_gate": {
            "passed": False,
            "thresholds": {
                "minimum_asr_gain": 0.05,
                "minimum_auc_gain": 0.02,
            },
            "slices": {},
        },
        "source_evaluation": evaluation,
        "source_evaluation_audits": {
            "exact_source": {
                "modern_cnn": {
                    "passed": True,
                    "row_count": 200,
                    "expected_cohort_verified": True,
                }
            }
        },
    }
    write_verified_json(run_dir / "manifest.json", run)
    study = {
        "schema_version": 1,
        "name": "fixture-phase-1",
        "status": "source_learning_failed",
        "research_valid": False,
        "publication_candidate": False,
        "elapsed_seconds": 100.0,
        "source_phase_elapsed_seconds": 100.0,
        "target_calls": 0,
        "target_evaluation_performed": False,
        "study_code_digest": "a" * 64,
        "runtime_environment": {
            "cuda_runtime": "13.0",
            "cudnn_version": 92000,
            "nvidia_driver": "580.126.09",
            "pip_freeze_path": "source/pip_freeze.txt",
            "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
            "requirements_sha256": hashlib.sha256(
                Path("requirements/rtx-publication.txt").read_bytes()
            ).hexdigest(),
        },
        "config": {
            "seeds": [17],
            "target_families": ["classical_cnn"],
            "primary_control": _CONTROL,
            "base_config": {"attack": {"max_queries": 50}},
        },
        "source_competence_gate": {
            "passed": False,
            "completed_runs": 1,
            "expected_runs": 1,
            "grid_complete": True,
            "failures": ["source competence threshold not met"],
        },
        "source_runs": [run],
    }
    write_verified_json(study_root / "study_manifest.json", study)
    return study_root, root / "bundle"


class Phase1EvidenceExportTests(unittest.TestCase):
    def test_export_is_deterministic_portable_and_self_verifying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            first = export_phase1_evidence(source, bundle)
            first_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in bundle.iterdir()
            }
            second = export_phase1_evidence(source, bundle)
            second_hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in bundle.iterdir()
            }
            self.assertEqual(first, second)
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(first["integrity"]["verified_runs"], 1)
            self.assertEqual(first["target_evaluation"]["target_calls"], 0)
            expected = {
                "README.md",
                "PROVENANCE.md",
                "SHA256SUMS",
                "summary.json",
                "dependency_freeze.txt",
                "environment_summary.json",
                "run_summary.csv",
                "condition_metrics.csv",
                "method_summary.csv",
                "input_checksums.csv",
                "raw_compact_evidence.json.gz",
                "raw_source_records.tar.gz",
                "method_performance.svg",
                "heldout_family_asr.svg",
                "bc_diagnostics.svg",
                "runtime.svg",
            }
            self.assertEqual({path.name for path in bundle.iterdir()}, expected)
            environment = json.loads((bundle / "environment_summary.json").read_text())
            self.assertIsNone(environment["run_start_manifest"]["git_revision"])
            self.assertIsNone(environment["run_start_manifest"]["gpu_model"])
            self.assertEqual(environment["dependencies"]["torch"], "2.13.0")
            self.assertEqual(
                environment["dependencies"]["torchvision"],
                "0.28.0",
            )
            self.assertEqual(
                environment["code_mapping"]["editable_install_commit"],
                "47bd57e9c6826a9e09203de2adacef64a75ace4e",
            )
            readme = (bundle / "README.md").read_text()
            self.assertIn("held-out attack-evaluation calls were 0", readme)
            self.assertNotIn("target victims were not evaluated", readme)
            checksums = (bundle / "SHA256SUMS").read_text().splitlines()
            self.assertEqual(len(checksums), len(expected) - 1)
            for line in checksums:
                digest, filename = line.split("  ", maxsplit=1)
                self.assertEqual(
                    hashlib.sha256((bundle / filename).read_bytes()).hexdigest(),
                    digest,
                )
            raw = gzip.decompress(
                (bundle / "raw_compact_evidence.json.gz").read_bytes()
            ).decode()
            self.assertNotIn("/home/", raw)
            self.assertNotIn(".pt", raw)
            self.assertNotIn("password", raw.lower())
            compact = json.loads(raw)
            self.assertEqual(compact["runs"][0]["fingerprint"], _FINGERPRINT)
            with tarfile.open(
                bundle / "raw_source_records.tar.gz",
                mode="r:gz",
            ) as archive:
                members = archive.getmembers()
                self.assertEqual(
                    [member.name for member in members],
                    [
                        f"runs/{_FINGERPRINT[:12]}/source_results.jsonl",
                        (f"runs/{_FINGERPRINT[:12]}/source_query_traces.jsonl"),
                    ],
                )
                for member in members:
                    self.assertEqual(member.mtime, 0)
                    self.assertEqual(member.uid, 0)
                    self.assertEqual(member.gid, 0)
                    self.assertEqual(member.mode, 0o644)
                    self.assertFalse(member.name.startswith("/"))
                    self.assertNotIn("..", Path(member.name).parts)
                results_member = archive.extractfile(members[0])
                self.assertIsNotNone(results_member)
                self.assertEqual(
                    results_member.read(),
                    (
                        b'{"method":"learned","success":true,'
                        b'"victim_family":"modern_cnn"}\n'
                    ),
                )
            with (bundle / "run_summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row["exact_source_asr"]), 0.12)
            self.assertAlmostEqual(float(row["asr_gain_vs_score_greedy"]), 0.07)

    def test_rejects_tampered_study_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            manifest = source / "study_manifest.json"
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                export_phase1_evidence(source, bundle)

    def test_rejects_tampered_dependency_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            freeze = source / "pip_freeze.txt"
            freeze.write_text(freeze.read_text() + "unsafe==1\n")
            with self.assertRaisesRegex(ValueError, "freeze checksum"):
                export_phase1_evidence(source, bundle)

    def test_rejects_credentialed_or_local_dependency_pins(self) -> None:
        unsafe_pins = (
            (
                "-e git+https://user:password@github.com/example/repo.git@"
                "47bd57e9c6826a9e09203de2adacef64a75ace4e"
                "#egg=unsafe"
            ),
            "-e file:///home/research/private-package",
        )
        for unsafe_pin in unsafe_pins:
            with self.subTest(unsafe_pin=unsafe_pin):
                with tempfile.TemporaryDirectory() as directory:
                    source, bundle = _write_fixture(Path(directory))
                    freeze = source / "pip_freeze.txt"
                    freeze.write_text(
                        f"torch==2.13.0\ntorchvision==0.28.0\n{unsafe_pin}\n"
                    )
                    manifest_path = source / "study_manifest.json"
                    study = json.loads(manifest_path.read_text())
                    study["runtime_environment"]["pip_freeze_sha256"] = hashlib.sha256(
                        freeze.read_bytes()
                    ).hexdigest()
                    write_verified_json(manifest_path, study)
                    with self.assertRaisesRegex(
                        ValueError,
                        "non-portable or sensitive|not a safe pin",
                    ):
                        export_phase1_evidence(source, bundle)

    def test_rejects_tampered_per_run_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            traces = source / "runs" / _FINGERPRINT[:12] / "source_query_traces.jsonl"
            traces.write_text(traces.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "query trace"):
                export_phase1_evidence(source, bundle)

    def test_rejects_path_traversal_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            manifest_path = source / "study_manifest.json"
            study = json.loads(manifest_path.read_text())
            study["source_runs"][0]["fingerprint"] = "../escape"
            write_verified_json(manifest_path, study)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                export_phase1_evidence(source, bundle)

    def test_rejects_machine_paths_inside_raw_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            run_dir = source / "runs" / _FINGERPRINT[:12]
            traces = run_dir / "source_query_traces.jsonl"
            traces.write_text(
                '{"family":"modern_cnn","private_path":"/home/private/model"}\n'
            )
            cache_path = run_dir / "source_evaluation.json"
            cache = json.loads(cache_path.read_text())
            cache["query_traces_sha256"] = hashlib.sha256(
                traces.read_bytes()
            ).hexdigest()
            write_verified_json(cache_path, cache)
            with self.assertRaisesRegex(
                ValueError,
                "non-portable or sensitive",
            ):
                export_phase1_evidence(source, bundle)

    def test_rejects_structured_secrets_and_private_network_values(self) -> None:
        forbidden = (
            {"password": "not-safe"},
            {"access_token": "not-safe"},
            {"Authorization": "Bearer token-value"},
            {"auth_header": "Bearer token-value"},
            {"endpoint": "http://10.23.4.5/private"},
            {"endpoint": "http://[::1]/private"},
            {"path_hint": "copied from /home/research/private"},
            {"path_hint": "~/private"},
            {"endpoint": "https://user:password@example.com"},
        )
        for value in forbidden:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "non-portable or sensitive",
                ):
                    validate_portable_value(value)
        validate_portable_value(
            {
                "checkpoint_sha256": "a" * 64,
                "public_reference": "https://8.8.8.8/example",
                "source_calls": 100,
            }
        )

    def test_rejects_raw_rows_from_the_held_out_target_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            run_dir = source / "runs" / _FINGERPRINT[:12]
            results = run_dir / "source_results.jsonl"
            results.write_text('{"victim_family":"classical_cnn","success":false}\n')
            cache_path = run_dir / "source_evaluation.json"
            cache = json.loads(cache_path.read_text())
            cache["results_sha256"] = hashlib.sha256(results.read_bytes()).hexdigest()
            write_verified_json(cache_path, cache)
            with self.assertRaisesRegex(ValueError, "held-out target family"):
                export_phase1_evidence(source, bundle)

    def test_rejects_nested_target_family_references_in_source_traces(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            run_dir = source / "runs" / _FINGERPRINT[:12]
            traces = run_dir / "source_query_traces.jsonl"
            traces.write_text(
                '{"family":"modern_cnn","sample_id":"cifar10:classical_cnn:victim:1"}\n'
            )
            cache_path = run_dir / "source_evaluation.json"
            cache = json.loads(cache_path.read_text())
            cache["query_traces_sha256"] = hashlib.sha256(
                traces.read_bytes()
            ).hexdigest()
            write_verified_json(cache_path, cache)
            with self.assertRaisesRegex(ValueError, "held-out target family"):
                export_phase1_evidence(source, bundle)

    def test_rejects_a_source_bundle_after_target_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            manifest_path = source / "study_manifest.json"
            study = json.loads(manifest_path.read_text())
            study["target_calls"] = 1
            write_verified_json(manifest_path, study)
            with self.assertRaisesRegex(ValueError, "target-free"):
                export_phase1_evidence(source, bundle)

    def test_rejects_unmanaged_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            bundle.mkdir()
            (bundle / "unreviewed.txt").write_text("do not include me")
            with self.assertRaisesRegex(ValueError, "unmanaged entries"):
                export_phase1_evidence(source, bundle)


if __name__ == "__main__":
    unittest.main()
