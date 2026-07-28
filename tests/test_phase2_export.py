import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from rl_transfer.verified_artifacts import write_verified_json


LEARNED = (
    "soft_gradient_bc_action_conditioned_groupdro_ppo_stochastic"
)
CONTROL = "score_greedy"
FAMILIES = ("classical_cnn", "modern_cnn", "transformer")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(
    asr: float,
    auc: float,
    *,
    entropy: float,
) -> dict[str, object]:
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
    }


def _audit() -> dict[str, object]:
    return {
        "passed": True,
        "expected_cohort_verified": True,
        "call_budget_verified": True,
        "frozen_policy_verified": True,
    }


def _write_run(
    study_root: Path,
    *,
    target_family: str,
    seed: int,
) -> dict[str, object]:
    fingerprint = hashlib.sha256(
        f"{target_family}:{seed}".encode()
    ).hexdigest()
    run_dir = study_root / "runs" / fingerprint[:12]
    run_dir.mkdir(parents=True)
    source_families = [
        family for family in FAMILIES if family != target_family
    ]
    source_evaluation: dict[str, object] = {}
    source_audits: dict[str, object] = {}
    for slice_index, slice_name in enumerate(
        ("exact_source", "seen_family_new_instance")
    ):
        source_evaluation[slice_name] = {}
        source_audits[slice_name] = {}
        for family_index, family in enumerate(source_families):
            offset = 0.005 * (slice_index + family_index)
            source_evaluation[slice_name][family] = {
                LEARNED: _metrics(
                    0.08 + offset,
                    0.035 + offset / 2,
                    entropy=0.96,
                ),
                CONTROL: _metrics(
                    0.09 + offset,
                    0.040 + offset / 2,
                    entropy=0.0,
                ),
                "random_action": _metrics(
                    0.05 + offset,
                    0.020 + offset / 2,
                    entropy=0.99,
                ),
            }
            source_audits[slice_name][family] = _audit()

    policy_path = run_dir / "policy.pt"
    policy_path.write_bytes(f"policy:{target_family}".encode())
    policy_sha = _sha256(policy_path)
    policy_path.with_suffix(".pt.sha256").write_text(
        policy_sha + "\n"
    )
    validation = {
        "accepted_steps": 3200,
        "target_mode": "soft",
        "baseline_provenance": "evaluated_labels_validation_oracle",
        "baseline_estimator": "empirical_best_constant_no_smoothing",
        "top1_accuracy": 0.02,
        "top5_accuracy": 0.08,
        "soft_cross_entropy": 4.55,
        "validation_oracle_top1_accuracy": 0.03,
        "validation_oracle_top5_accuracy": 0.12,
        "validation_oracle_soft_cross_entropy": 4.50,
    }
    blocks = [
        {
            "episode_offset": offset,
            "episodes": 50,
            "trained_episodes": 42 + index,
            "source_calls": 2100 + 10 * index,
            "elapsed_seconds": 8.0 + index,
            "ppo": {
                "loss": 0.10 - index * 0.01,
                "policy_loss": -0.05,
                "value_loss": 0.30,
                "entropy": 4.40,
            },
            "family_diagnostics": {
                family: {
                    "eligible_episodes": 20,
                    "successful_episodes": index,
                    "episode_return": {"mean": -0.4 + 0.1 * index},
                    "margin_reduction": {"mean": 0.03 + 0.01 * index},
                }
                for family in source_families
            },
        }
        for index, offset in enumerate((0, 50))
    ]
    victim_instances = {
        family: [
            {
                "victim_id": f"{family}-fixture-0",
                "checkpoint": f"victim_cache/{family}.pt",
                "checkpoint_sha256": hashlib.sha256(
                    family.encode()
                ).hexdigest(),
                "source_validation_accuracy": (
                    0.78 if family == "transformer" else 0.93
                ),
                "resumed": True,
            }
        ]
        for family in source_families
    }
    run = {
        "schema_version": 1,
        "name": f"phase2-{target_family}-{seed}",
        "fingerprint": fingerprint,
        "run_dir": fingerprint[:12],
        "status": "source_complete",
        "seed": seed,
        "target_family": target_family,
        "source_families": source_families,
        "research_valid": False,
        "target_calls": 0,
        "target_evaluation_performed": False,
        "validation_roles_disjoint": True,
        "config_digest": "a" * 64,
        "split_digest": "b" * 64,
        "data_role_digests": {"source_gate": "c" * 64},
        "victim_bank_digest": "d" * 64,
        "victim_cache_digest": "e" * 64,
        "elapsed_seconds": 1.0,
        "source_evaluation_elapsed_seconds": 205.0,
        "runtime": {
            "code_digest": "f" * 64,
            "git_revision": "1" * 40,
            "python": "3.12.3",
            "torch": "2.13.0+cu130",
            "cuda_runtime": "13.0",
            "cudnn_version": 92000,
            "cuda_device_name": "NVIDIA GeForce RTX 2080 Ti",
            "determinism": "deterministic algorithms requested",
        },
        "victim_instances": victim_instances,
        "victim_accuracy_gate": {"passed": True},
        "victim_access_audit": {
            "source_victims_only": True,
            "victim_cache_only": True,
            "constructed_families": source_families,
            "untouched_families": [target_family],
            "model_instances_by_family": {
                family: 0 if family == target_family else 1
                for family in FAMILIES
            },
            "validation_evaluations_by_family": {
                family: 0 if family == target_family else 1
                for family in FAMILIES
            },
            "heldout_family": target_family,
            "heldout_family_model_calls": 0,
            "heldout_family_validation_calls": 0,
            "passed": True,
        },
        "policy": {
            "checkpoint": "policy.pt",
            "checkpoint_sha256": policy_sha,
            "checkpoints": {
                "main": {
                    "method_id": LEARNED,
                    "path": "policy.pt",
                    "sha256": policy_sha,
                }
            },
            "persistent_digest": "2" * 64,
            "training": {
                "episodes": 600,
                "completed_episodes": 600,
                "trained_episodes": 500,
                "source_calls": 24000,
                "blocks": blocks,
                "behavior_cloning": {
                    "enabled": True,
                    "elapsed_seconds": 230.0,
                    "validation": validation,
                    "gate": {
                        "objective": "soft_gradient_distillation",
                        "passed": False,
                    },
                },
            },
        },
        "source_competence_gate": {
            "passed": False,
            "thresholds": {
                "minimum_asr_gain": 0.05,
                "minimum_auc_gain": 0.02,
            },
        },
        "source_evaluation": source_evaluation,
        "source_evaluation_audits": source_audits,
    }
    results = run_dir / "source_results.jsonl"
    traces = run_dir / "source_query_traces.jsonl"
    results.write_text(
        json.dumps(
            {
                "method": LEARNED,
                "success": False,
                "family": source_families[0],
                "sample_id": "fixture-0",
            },
            sort_keys=True,
        )
        + "\n"
    )
    traces.write_text(
        json.dumps(
            {
                "method": LEARNED,
                "actions": [1, 2],
                "family": source_families[0],
                "sample_id": "fixture-0",
            },
            sort_keys=True,
        )
        + "\n"
    )
    write_verified_json(
        run_dir / "source_evaluation.json",
        {
            "source_evaluation": source_evaluation,
            "results_sha256": _sha256(results),
            "query_traces_sha256": _sha256(traces),
        },
    )
    write_verified_json(run_dir / "manifest.json", run)
    return run


def _write_fixture(root: Path) -> tuple[Path, Path]:
    study_root = root / "source"
    study_root.mkdir()
    freeze_path = study_root / "pip_freeze.txt"
    freeze_path.write_text(
        "torch==2.13.0\n"
        "torchvision==0.28.0\n"
        "-e git+https://github.com/example/project.git@"
        + "1" * 40
        + "#egg=rl_transfer_research\n"
    )
    victim_cache = study_root / "runs" / "victim_cache"
    victim_cache.mkdir(parents=True)
    for family in FAMILIES:
        checkpoint = victim_cache / f"{family}.pt"
        checkpoint.write_bytes(family.encode())
        checkpoint.with_suffix(".pt.sha256").write_text(
            _sha256(checkpoint) + "\n"
        )
    runs = [
        _write_run(
            study_root,
            target_family=family,
            seed=17,
        )
        for family in FAMILIES
    ]
    attempts = study_root / "attempt_logs"
    attempts.mkdir()
    (attempts / "attempt-1.log").write_text(
        "known JSON-key mismatch; no target calls\n"
    )
    screen = {
        "schema_version": 1,
        "name": "cifar10-rtx-phase2-action-conditioned-screen",
        "status": "screen_complete",
        "research_valid": False,
        "publication_candidate": False,
        "target_calls": 0,
        "target_evaluation_performed": False,
        "study_code_digest": "f" * 64,
        "protocol_sha256": "3" * 64,
        "base_config_digest": "4" * 64,
        "dataset_version": "torchvision-fixture",
        "runtime_environment": {
            **runs[0]["runtime"],
            "pip_freeze_sha256": _sha256(freeze_path),
        },
        "config": {
            "seeds": [17],
            "target_families": list(FAMILIES),
            "minimum_mean_score_asr_gain": 0.01,
            "minimum_mean_score_auc_gain": 0.005,
        },
        "victim_cache_reuse": {
            "all_verified": True,
            "authentication": "pinned_phase1_study_manifest",
            "cache_fingerprints": ["5" * 64],
            "checkpoint_count": 9,
            "study_manifest_sha256": "6" * 64,
        },
        "source_runs": runs,
        "screen_promotion_decision": {
            "passed": False,
            "eligible_for_stage_2b": False,
            "publication_candidate": False,
            "target_evaluation_authorized": False,
            "grid_complete": True,
            "expected_cells": 3,
            "completed_cells": 3,
            "condition_count": 12,
            "strict_publication_source_gate_passes": 0,
            "metrics": {
                "mean_bc_top5_or_hard_accuracy_gain": -0.04,
                "mean_bc_loss_improvement_over_validation_oracle": -0.05,
                "mean_asr_gain_over_score_greedy": -0.01,
                "mean_auc_gain_over_score_greedy": -0.005,
                "positive_asr_and_auc_condition_fraction": 0.0,
            },
            "requirements": {
                "grid_complete": True,
                "mean_bc_accuracy_gain": False,
                "mean_bc_nll_improvement": False,
                "mean_score_asr_gain": False,
                "mean_score_auc_gain": False,
                "positive_condition_fraction": False,
            },
        },
    }
    write_verified_json(study_root / "screen_manifest.json", screen)
    return study_root, root / "bundle"


class Phase2EvidenceExportTests(unittest.TestCase):
    def test_export_is_deterministic_portable_and_self_verifying(
        self,
    ) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            first = export_phase2_evidence(source, bundle)
            first_hashes = {
                path.name: _sha256(path)
                for path in bundle.iterdir()
            }
            second = export_phase2_evidence(source, bundle)
            second_hashes = {
                path.name: _sha256(path)
                for path in bundle.iterdir()
            }

            self.assertEqual(first, second)
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(first["status"], "screen_complete")
            self.assertFalse(first["promotion"]["passed"])
            self.assertEqual(first["target_evaluation"]["target_calls"], 0)
            self.assertEqual(first["integrity"]["verified_runs"], 3)
            self.assertEqual(
                {path.name for path in bundle.iterdir()},
                {
                    "README.md",
                    "PROVENANCE.md",
                    "SHA256SUMS",
                    "summary.json",
                    "environment_summary.json",
                    "dependency_freeze.txt",
                    "condition_metrics.csv",
                    "fold_summary.csv",
                    "bc_diagnostics.csv",
                    "training_blocks.csv",
                    "victim_accuracy.csv",
                    "input_checksums.csv",
                    "attempt_log_checksums.csv",
                    "raw_compact_evidence.json.gz",
                    "raw_source_records.tar.gz",
                    "source_asr_by_method.svg",
                    "gain_vs_score_greedy.svg",
                    "bc_diagnostics.svg",
                    "runtime.svg",
                },
            )
            with gzip.open(
                bundle / "raw_compact_evidence.json.gz",
                "rt",
            ) as handle:
                compact = handle.read()
            self.assertNotIn("/home/", compact)
            self.assertNotIn("/Users/", compact)
            self.assertNotIn(".pt", compact)
            self.assertIn("policy_checkpoint_sha256", compact)
            with tarfile.open(
                bundle / "raw_source_records.tar.gz",
                "r:gz",
            ) as archive:
                members = archive.getnames()
            self.assertEqual(len(members), 6)
            self.assertTrue(
                all(not Path(member).is_absolute() for member in members)
            )
            checksums = {}
            for line in (bundle / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                checksums[name] = digest
            self.assertNotIn("SHA256SUMS", checksums)
            self.assertEqual(
                checksums,
                {
                    path.name: _sha256(path)
                    for path in bundle.iterdir()
                    if path.name != "SHA256SUMS"
                },
            )

    def test_export_rejects_tampered_raw_results(self) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            results = next(
                (source / "runs").glob("*/source_results.jsonl")
            )
            results.write_text('{"tampered": true}\n')

            with self.assertRaisesRegex(
                ValueError,
                "source result rows failed checksum",
            ):
                export_phase2_evidence(source, bundle)

    def test_export_rejects_any_target_access(self) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            manifest_path = source / "screen_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["target_calls"] = 1
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "target-free",
            ):
                export_phase2_evidence(source, bundle)

    def test_export_rejects_sensitive_dependency_freeze_text(
        self,
    ) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            (source / "pip_freeze.txt").write_text(
                "torch==2.13.0\npassword=do-not-export\n"
            )

            with self.assertRaisesRegex(
                ValueError,
                "dependency freeze",
            ):
                export_phase2_evidence(source, bundle)

    def test_dependency_freeze_comments_are_not_exported(self) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            freeze = source / "pip_freeze.txt"
            freeze.write_text(
                freeze.read_text()
                + "# PRIVATE-CREDENTIAL=do-not-publish\n"
            )
            manifest_path = source / "screen_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["runtime_environment"]["pip_freeze_sha256"] = (
                _sha256(freeze)
            )
            write_verified_json(manifest_path, manifest)

            export_phase2_evidence(source, bundle)

            exported = (bundle / "dependency_freeze.txt").read_text()
            self.assertNotIn("PRIVATE-CREDENTIAL", exported)
            self.assertNotIn("do-not-publish", exported)

    def test_export_requires_dependency_freeze_binding(self) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            manifest_path = source / "screen_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["runtime_environment"]["pip_freeze_sha256"]
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "dependency freeze checksum",
            ):
                export_phase2_evidence(source, bundle)

    def test_export_binds_policy_manifest_hash_to_checkpoint(
        self,
    ) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            screen_path = source / "screen_manifest.json"
            screen = json.loads(screen_path.read_text())
            run = screen["source_runs"][0]
            run["policy"]["checkpoint_sha256"] = "0" * 64
            run["policy"]["checkpoints"]["main"]["sha256"] = "0" * 64
            run_path = (
                source / "runs" / run["run_dir"] / "manifest.json"
            )
            write_verified_json(run_path, run)
            write_verified_json(screen_path, screen)

            with self.assertRaisesRegex(
                ValueError,
                "policy checkpoint checksum",
            ):
                export_phase2_evidence(source, bundle)

    def test_export_binds_victim_manifest_hash_to_checkpoint(
        self,
    ) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            screen_path = source / "screen_manifest.json"
            screen = json.loads(screen_path.read_text())
            run = screen["source_runs"][0]
            family = run["source_families"][0]
            run["victim_instances"][family][0][
                "checkpoint_sha256"
            ] = "0" * 64
            run_path = (
                source / "runs" / run["run_dir"] / "manifest.json"
            )
            write_verified_json(run_path, run)
            write_verified_json(screen_path, screen)

            with self.assertRaisesRegex(
                ValueError,
                "victim checkpoint checksum",
            ):
                export_phase2_evidence(source, bundle)

    def test_export_neutralizes_spreadsheet_formula_text(self) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, bundle = _write_fixture(Path(directory))
            screen_path = source / "screen_manifest.json"
            screen = json.loads(screen_path.read_text())
            run = screen["source_runs"][0]
            family = run["source_families"][0]
            run["victim_instances"][family][0]["victim_id"] = "=1+1"
            run_path = (
                source / "runs" / run["run_dir"] / "manifest.json"
            )
            write_verified_json(run_path, run)
            write_verified_json(screen_path, screen)

            export_phase2_evidence(source, bundle)

            csv_text = (bundle / "victim_accuracy.csv").read_text()
            self.assertIn("'=1+1", csv_text)
            self.assertNotIn(",=1+1,", csv_text)

    def test_output_cannot_overlap_source_archive(self) -> None:
        from rl_transfer.phase2_export import export_phase2_evidence

        with tempfile.TemporaryDirectory() as directory:
            source, _ = _write_fixture(Path(directory))

            with self.assertRaisesRegex(ValueError, "overlap"):
                export_phase2_evidence(source, source / "bundle")

    def test_cli_reports_the_verified_export(self) -> None:
        script_path = (
            Path(__file__).parents[1]
            / "scripts"
            / "export_phase2_evidence.py"
        )
        spec = importlib.util.spec_from_file_location(
            "phase2_export_evidence_script",
            script_path,
        )
        if spec is None or spec.loader is None:
            self.fail("could not load the Phase 2 export CLI")
        export_phase2_evidence = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export_phase2_evidence)

        summary = {
            "status": "screen_complete",
            "promotion": {"passed": False},
            "target_evaluation": {"target_calls": 0},
            "integrity": {"verified_runs": 3},
        }
        output = io.StringIO()
        with (
            mock.patch.object(
                export_phase2_evidence,
                "export_phase2_evidence",
                return_value=summary,
            ) as export,
            redirect_stdout(output),
        ):
            export_phase2_evidence.main(
                ["--source", "source", "--output", "bundle"]
            )

        export.assert_called_once_with(Path("source"), Path("bundle"))
        reported = json.loads(output.getvalue())
        self.assertEqual(reported["status"], "screen_complete")
        self.assertFalse(reported["promotion_passed"])
        self.assertEqual(reported["target_calls"], 0)
        self.assertEqual(reported["verified_runs"], 3)


if __name__ == "__main__":
    unittest.main()
