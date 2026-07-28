import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from rl_transfer.cifar_config import MacPilotConfig
from rl_transfer.phase2_config import FAMILIES, Phase2ScreenConfig
from rl_transfer.phase2_screen import (
    build_phase2_dry_run,
    run_phase2_screen_from_datasets,
    screen_promotion_decision,
)
from rl_transfer.verified_artifacts import write_verified_json


CONFIG_PATH = Path("configs/rl_transfer/cifar10_rtx_phase2_screen.json")


def _config_payload(**overrides: object) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "name": "cifar10-rtx-phase2-action-conditioned-screen",
        "research_valid": False,
        "base_config": "configs/rl_transfer/cifar10_rtx_phase2_base.json",
        "output_dir": "output/rl_transfer/cifar10_rtx_phase2_screen",
        "device": "cuda",
        "seeds": [17],
        "target_families": list(FAMILIES),
        "resume": True,
        "split_seed": 20260727,
        "victim_seed": 1000000,
        "victim_cache_source": (
            "output/rl_transfer/cifar10_rtx_publication/"
            "cifar10-rtx-publication/runs/victim_cache"
        ),
        "victim_study_manifest": (
            "output/rl_transfer/cifar10_rtx_publication/"
            "cifar10-rtx-publication/study_manifest.json"
        ),
        "victim_study_manifest_sha256": (
            "791140871a987ec400cca083aea9b1192d8e73f2a5e70e5504d"
            "cfcae7f85911d"
        ),
        "require_verified_victim_cache": True,
        "max_wall_clock_minutes": 60,
        "estimated_minutes_per_cell": 12,
        "minimum_mean_bc_accuracy_gain": 0.01,
        "minimum_mean_bc_nll_improvement": 0.02,
        "minimum_mean_score_asr_gain": 0.01,
        "minimum_mean_score_auc_gain": 0.005,
        "minimum_positive_condition_fraction": 0.67,
    }
    return {**payload, **overrides}


def _base_config(**overrides: object) -> MacPilotConfig:
    payload = json.loads(
        Path(
            "configs/rl_transfer/cifar10_rtx_phase2_base.json"
        ).read_text()
    )
    payload.update(overrides)
    return MacPilotConfig(**payload)


def _audit() -> dict[str, object]:
    return {
        "passed": True,
        "expected_cohort_verified": True,
        "call_budget_verified": True,
        "frozen_policy_verified": True,
    }


def _source_metrics(
    learned_asr: float,
    learned_auc: float,
    score_asr: float,
    score_auc: float,
) -> dict[str, object]:
    return {
        "soft_gradient_bc_action_conditioned_groupdro_ppo_stochastic": {
            "asr_at_budgets": {"0": 0.0, "50": learned_asr},
            "asr_query_auc": learned_auc,
        },
        "score_greedy": {
            "asr_at_budgets": {"0": 0.0, "50": score_asr},
            "asr_query_auc": score_auc,
        },
    }


def _screen_run(
    family: str,
    seed: int,
    *,
    bc_accuracy_gain: float = 0.03,
    bc_nll_improvement: float = 0.04,
    asr_gain: float = 0.03,
    auc_gain: float = 0.01,
) -> dict[str, object]:
    source_families = tuple(item for item in FAMILIES if item != family)
    methods = _source_metrics(
        0.10 + asr_gain,
        0.04 + auc_gain,
        0.10,
        0.04,
    )
    validation = {
        "accuracy": 0.04 + bc_accuracy_gain,
        "validation_oracle_top1_accuracy": 0.04,
        "nll": 4.2 - bc_nll_improvement,
        "validation_oracle_nll": 4.2,
        "baseline_provenance": "evaluated_labels_validation_oracle",
        "baseline_estimator": "empirical_best_constant_no_smoothing",
    }
    return {
        "status": "source_complete",
        "seed": seed,
        "target_family": family,
        "target_evaluation_performed": False,
        "target_calls": 0,
        "validation_roles_disjoint": True,
        "victim_access_audit": {
            "source_victims_only": True,
            "victim_cache_only": True,
            "constructed_families": list(source_families),
            "untouched_families": [family],
            "model_instances_by_family": {
                item: 0 if item == family else 1
                for item in FAMILIES
            },
            "validation_evaluations_by_family": {
                item: 0 if item == family else 1
                for item in FAMILIES
            },
            "heldout_family": family,
            "heldout_family_model_calls": 0,
            "heldout_family_validation_calls": 0,
            "passed": True,
        },
        "victim_accuracy_gate": {"passed": True},
        "source_evaluation_audits": {
            source_slice: {
                source_family: _audit()
                for source_family in source_families
            }
            for source_slice in (
                "exact_source",
                "seen_family_new_instance",
            )
        },
        "source_evaluation": {
            source_slice: {
                source_family: methods
                for source_family in source_families
            }
            for source_slice in (
                "exact_source",
                "seen_family_new_instance",
            )
        },
        "source_competence_gate": {"passed": False},
        "policy": {
            "checkpoint_sha256": "a" * 64,
            "training": {
                "behavior_cloning": {
                    "enabled": True,
                    "validation": validation,
                    "gate": {"passed": False},
                }
            },
        },
    }


def _write_source_artifact_fixture(
    output: Path,
    *,
    binding_matches: bool,
) -> tuple[dict[str, object], MacPilotConfig]:
    run_dir = output / "cell"
    run_dir.mkdir()
    derived = replace(
        _base_config(),
        name="phase2-binding-fixture",
        output_dir="output/rl_transfer/phase2-fixture/runs",
        target_family="classical_cnn",
        seed=17,
    )
    run = _screen_run("classical_cnn", 17)
    victim_instances: dict[str, list[dict[str, object]]] = {}
    for family in ("modern_cnn", "transformer"):
        family_instances: list[dict[str, object]] = []
        for index in range(3):
            victim_id = f"{family}-{index}"
            victim_path = (
                output
                / "victim_cache"
                / "fixture"
                / f"{victim_id}.pt"
            )
            victim_path.parent.mkdir(parents=True, exist_ok=True)
            victim_path.write_bytes(
                f"verified-victim:{victim_id}".encode()
            )
            victim_sha = hashlib.sha256(
                victim_path.read_bytes()
            ).hexdigest()
            victim_path.with_suffix(".pt.sha256").write_text(
                victim_sha + "\n"
            )
            family_instances.append(
                {
                    "victim_id": victim_id,
                    "checkpoint": (
                        f"victim_cache/fixture/{victim_id}.pt"
                    ),
                    "checkpoint_sha256": victim_sha,
                }
            )
        victim_instances[family] = family_instances
    policy_path = run_dir / "policy.pt"
    policy_path.write_bytes(b"verified-policy-fixture")
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    policy_path.with_suffix(".pt.sha256").write_text(policy_sha + "\n")
    policy_checkpoints = {
        "main": {
            "path": "policy.pt",
            "sha256": policy_sha,
        }
    }
    run.update(
        {
            "run_dir": "cell",
            "config": asdict(derived),
            "fingerprint": "f" * 64,
            "config_digest": derived.digest(),
            "split_digest": "s" * 64,
            "data_role_digests": {"source_gate": "d" * 64},
            "runtime": {"code_digest": "c" * 64},
            "source_families": ["modern_cnn", "transformer"],
            "victim_instances": victim_instances,
        }
    )
    run["policy"].update(
        {
            "checkpoint": "policy.pt",
            "checkpoint_sha256": policy_sha,
            "checkpoints": policy_checkpoints,
        }
    )
    persisted_keys = (
        "fingerprint",
        "config_digest",
        "split_digest",
        "seed",
        "target_family",
        "target_evaluation_performed",
        "target_calls",
        "victim_access_audit",
    )
    write_verified_json(
        run_dir / "manifest.json",
        {key: run[key] for key in persisted_keys},
    )
    results = run_dir / "source_results.jsonl"
    traces = run_dir / "source_query_traces.jsonl"
    results.write_text('{"fixture": "result"}\n')
    traces.write_text('{"fixture": "trace"}\n')
    source_checkpoints = {
        instance["victim_id"]: instance["checkpoint_sha256"]
        for family in run["source_families"]
        for instance in victim_instances[family]
    }
    binding = {
        "config_digest": run["config_digest"],
        "code_digest": run["runtime"]["code_digest"],
        "split_digest": run["split_digest"],
        "data_role_digests": run["data_role_digests"],
        "policy_checkpoints": policy_checkpoints,
        "source_victim_checkpoints": source_checkpoints,
    }
    write_verified_json(
        run_dir / "source_evaluation.json",
        {
            "binding": binding if binding_matches else {"tampered": True},
            "source_evaluation": run["source_evaluation"],
            "results_sha256": hashlib.sha256(
                results.read_bytes()
            ).hexdigest(),
            "query_traces_sha256": hashlib.sha256(
                traces.read_bytes()
            ).hexdigest(),
        },
    )
    return run, derived


class Phase2ConfigurationTests(unittest.TestCase):
    def test_committed_screen_is_short_source_only_and_reuses_cache(self) -> None:
        config = Phase2ScreenConfig.from_json(CONFIG_PATH)
        base = MacPilotConfig.from_json(Path(config.base_config))

        self.assertEqual(config.target_families, FAMILIES)
        self.assertEqual(len(config.seeds), 1)
        self.assertLessEqual(len(config.seeds), 3)
        self.assertTrue(config.resume)
        self.assertTrue(config.require_verified_victim_cache)
        self.assertLessEqual(config.max_wall_clock_minutes, 120)
        self.assertFalse(base.train_ablation_policies)
        self.assertLessEqual(base.policy_episodes, 1500)
        self.assertLessEqual(base.source_evaluation_images, 100)
        self.assertGreater(base.behavior_cloning_episodes, 0)
        self.assertEqual(base.policy_actor_mode, "action_conditioned")
        self.assertEqual(base.image_patch_feature_mode, "statistics")
        self.assertEqual(base.behavior_cloning_soft_temperature, 0.5)
        self.assertEqual(base.policy_evaluation_temperature, 1.0)

    def test_invalid_or_expansive_screen_contracts_are_rejected(self) -> None:
        cases = (
            ("seeds", []),
            ("seeds", [1, 2, 3, 4]),
            ("seeds", [1, 1]),
            ("seeds", [-1]),
            ("target_families", list(reversed(FAMILIES))),
            ("resume", False),
            ("research_valid", True),
            ("require_verified_victim_cache", False),
            ("victim_cache_source", "../escape"),
            ("max_wall_clock_minutes", 241),
            ("minimum_positive_condition_fraction", 0.49),
        )
        for field, value in cases:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    Phase2ScreenConfig(
                        **_config_payload(**{field: value})
                    )

    def test_dry_run_has_an_explicit_bounded_runtime_and_no_target_surface(
        self,
    ) -> None:
        config = Phase2ScreenConfig(**_config_payload())
        plan = build_phase2_dry_run(config, _base_config())

        self.assertEqual(plan["mode"], "source_only_screen")
        self.assertEqual(plan["total_cells"], 3)
        self.assertEqual(plan["pending_cells"], 3)
        self.assertEqual(plan["estimated_remaining_minutes"], 36)
        self.assertEqual(plan["maximum_invocation_minutes"], 60)
        self.assertFalse(plan["target_evaluation_available"])
        self.assertNotIn("phase", plan)


class ScreenPromotionTests(unittest.TestCase):
    def test_screen_decision_is_separate_from_publication_and_target_gates(
        self,
    ) -> None:
        config = Phase2ScreenConfig(**_config_payload())
        runs = [
            _screen_run(family, 17)
            for family in FAMILIES
        ]

        decision = screen_promotion_decision(runs, config)

        self.assertTrue(decision["passed"])
        self.assertTrue(decision["eligible_for_stage_2b"])
        self.assertFalse(decision["publication_candidate"])
        self.assertFalse(decision["target_evaluation_authorized"])
        self.assertEqual(decision["decision_scope"], "source-only screening")
        self.assertEqual(decision["condition_count"], 12)

    def test_screen_decision_fails_closed_for_missing_grid_or_weak_signal(
        self,
    ) -> None:
        config = Phase2ScreenConfig(**_config_payload())
        missing = [_screen_run(family, 17) for family in FAMILIES[:-1]]
        weak = [
            _screen_run(
                family,
                17,
                bc_accuracy_gain=0.0,
                bc_nll_improvement=0.0,
                asr_gain=-0.01,
                auc_gain=-0.005,
            )
            for family in FAMILIES
        ]

        self.assertFalse(
            screen_promotion_decision(missing, config)["passed"]
        )
        decision = screen_promotion_decision(weak, config)
        self.assertFalse(decision["passed"])
        self.assertFalse(decision["target_evaluation_authorized"])

    def test_soft_target_screen_uses_soft_ce_and_top5_diagnostics(
        self,
    ) -> None:
        config = Phase2ScreenConfig(**_config_payload())
        runs = [_screen_run(family, 17) for family in FAMILIES]
        for run in runs:
            validation = run["policy"]["training"]["behavior_cloning"][
                "validation"
            ]
            validation.update(
                {
                    "target_mode": "soft",
                    "top1_accuracy": 0.0,
                    "validation_oracle_top1_accuracy": 1.0,
                    "top5_accuracy": 0.18,
                    "validation_oracle_top5_accuracy": 0.14,
                    "soft_cross_entropy": 4.0,
                    "validation_oracle_soft_cross_entropy": 4.05,
                    "accuracy": 0.0,
                    "majority_accuracy": 1.0,
                    "nll": 10.0,
                }
            )

        decision = screen_promotion_decision(runs, config)

        self.assertTrue(decision["passed"])
        self.assertAlmostEqual(
            decision["metrics"][
                "mean_bc_top5_or_hard_accuracy_gain"
            ],
            0.04,
        )
        self.assertAlmostEqual(
            decision["metrics"][
                "mean_bc_loss_improvement_over_validation_oracle"
            ],
            0.05,
        )

    def test_screen_decision_rejects_target_access_or_failed_raw_audit(
        self,
    ) -> None:
        config = Phase2ScreenConfig(**_config_payload())
        accessed = [_screen_run(family, 17) for family in FAMILIES]
        accessed[0]["target_calls"] = 1
        with self.assertRaises(ValueError):
            screen_promotion_decision(accessed, config)

        bad_audit = [_screen_run(family, 17) for family in FAMILIES]
        bad_audit[0]["source_evaluation_audits"]["exact_source"][
            "modern_cnn"
        ]["passed"] = False
        with self.assertRaises(ValueError):
            screen_promotion_decision(bad_audit, config)

        bad_oracle = [_screen_run(family, 17) for family in FAMILIES]
        bad_oracle[0]["policy"]["training"]["behavior_cloning"][
            "validation"
        ]["baseline_provenance"] = "training_labels_empirical_constant"
        with self.assertRaisesRegex(ValueError, "provenance"):
            screen_promotion_decision(bad_oracle, config)


class Phase2RunnerTests(unittest.TestCase):
    def test_cache_fingerprint_mismatch_refuses_victim_retraining(self) -> None:
        config = Phase2ScreenConfig(**_config_payload())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch(
                    "rl_transfer.phase2_screen.resolve_within_repository",
                    side_effect=lambda value, **_kwargs: (
                        root / Path(value).name
                    ),
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.MacPilotConfig.from_json",
                    return_value=_base_config(),
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.mirror_verified_victim_cache",
                    return_value={
                        "all_verified": True,
                        "cache_fingerprints": ["a" * 64],
                    },
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.expected_victim_cache_fingerprint",
                    return_value="b" * 64,
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.run_cifar_pilot_from_datasets",
                ) as pilot,
                self.assertRaisesRegex(
                    ValueError,
                    "refusing to retrain victims",
                ),
            ):
                run_phase2_screen_from_datasets(
                    config,
                    object(),
                    object(),
                    dataset_version="fixture-v1",
                )
        pilot.assert_not_called()

    def test_runner_never_requests_target_evaluation(self) -> None:
        config = Phase2ScreenConfig(**_config_payload())
        produced = {
            (family, 17): _screen_run(family, 17)
            for family in FAMILIES
        }

        def fake_run(derived, *_args, **kwargs):
            self.assertIs(kwargs["evaluate_target"], False)
            self.assertIs(kwargs["source_victims_only"], True)
            self.assertIs(kwargs["victim_cache_only"], True)
            self.assertIs(kwargs["portable_paths"], True)
            return produced[(derived.target_family, derived.seed)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch(
                    "rl_transfer.phase2_screen.resolve_within_repository",
                    side_effect=lambda value, **_kwargs: (
                        root / Path(value).name
                    ),
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.MacPilotConfig.from_json",
                    return_value=_base_config(),
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.mirror_verified_victim_cache",
                    return_value={
                        "all_verified": True,
                        "checkpoint_count": 9,
                        "cache_fingerprints": ["f" * 64],
                    },
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.expected_victim_cache_fingerprint",
                    return_value="f" * 64,
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.run_cifar_pilot_from_datasets",
                    side_effect=fake_run,
                ) as pilot,
                mock.patch(
                    "rl_transfer.phase2_screen.validate_source_run_artifacts"
                ),
            ):
                result = run_phase2_screen_from_datasets(
                    config,
                    object(),
                    object(),
                    dataset_version="fixture-v1",
                )

        self.assertEqual(pilot.call_count, 3)
        self.assertEqual(result["status"], "screen_complete")
        self.assertFalse(result["target_evaluation_performed"])
        self.assertEqual(result["target_calls"], 0)
        self.assertFalse(
            result["screen_promotion_decision"][
                "target_evaluation_authorized"
            ]
        )
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("/home/", serialized)
        self.assertNotIn("/Users/", serialized)

    def test_deadline_stops_between_cells_and_preserves_resumable_manifest(
        self,
    ) -> None:
        config = Phase2ScreenConfig(**_config_payload())
        clocks = iter((0.0, 0.0, 3601.0, 3601.0))
        first = _screen_run(FAMILIES[0], 17)
        captured: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch(
                    "rl_transfer.phase2_screen.resolve_within_repository",
                    side_effect=lambda value, **_kwargs: (
                        root / Path(value).name
                    ),
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.MacPilotConfig.from_json",
                    return_value=_base_config(),
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.mirror_verified_victim_cache",
                    return_value={
                        "all_verified": True,
                        "cache_fingerprints": ["f" * 64],
                    },
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.expected_victim_cache_fingerprint",
                    return_value="f" * 64,
                ),
                mock.patch(
                    "rl_transfer.phase2_screen.run_cifar_pilot_from_datasets",
                    return_value=first,
                ) as pilot,
                mock.patch(
                    "rl_transfer.phase2_screen.validate_source_run_artifacts"
                ),
                mock.patch(
                    "rl_transfer.phase2_screen._write_manifest",
                    side_effect=lambda _path, payload: captured.append(
                        dict(payload)
                    ),
                ),
            ):
                result = run_phase2_screen_from_datasets(
                    config,
                    object(),
                    object(),
                    dataset_version="fixture-v1",
                    clock=lambda: next(clocks),
                )

        self.assertEqual(pilot.call_count, 1)
        self.assertEqual(result["status"], "screen_deadline_reached")
        self.assertEqual(len(result["source_runs"]), 1)
        self.assertFalse(result["screen_promotion_decision"]["passed"])
        self.assertEqual(captured[-1]["status"], "screen_deadline_reached")

    def test_tampered_or_mismatched_resume_manifest_is_not_ignored(
        self,
    ) -> None:
        from rl_transfer.phase2_screen import load_resumable_screen_manifest

        config = Phase2ScreenConfig(**_config_payload())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_verified_json(
                path,
                {
                    "schema_version": 1,
                    "status": "screen_running",
                    "config": {
                        **asdict(config),
                        "seeds": list(config.seeds),
                        "target_families": list(config.target_families),
                    },
                    "base_config_digest": "b" * 64,
                    "study_code_digest": "c" * 64,
                    "dataset_version": "fixture-v1",
                    "source_runs": [],
                },
            )
            with self.assertRaises(ValueError):
                load_resumable_screen_manifest(
                    path,
                    config=config,
                    base_config_digest="x" * 64,
                    code_digest="c" * 64,
                    dataset_version="fixture-v1",
                    protocol_sha256="p" * 64,
                )
            path.write_text(path.read_text() + " ")
            with self.assertRaises(ValueError):
                load_resumable_screen_manifest(
                    path,
                    config=config,
                    base_config_digest="b" * 64,
                    code_digest="c" * 64,
                    dataset_version="fixture-v1",
                    protocol_sha256="p" * 64,
                )

    def test_matching_verified_manifest_is_resumable(self) -> None:
        from rl_transfer.phase2_screen import load_resumable_screen_manifest

        config = Phase2ScreenConfig(**_config_payload())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            payload = {
                "schema_version": 1,
                "status": "screen_failed",
                "research_valid": False,
                "publication_candidate": False,
                "config": {
                    **asdict(config),
                    "seeds": list(config.seeds),
                    "target_families": list(config.target_families),
                },
                "base_config_digest": "b" * 64,
                "study_code_digest": "c" * 64,
                "dataset_version": "fixture-v1",
                "protocol_sha256": "p" * 64,
                "target_evaluation_performed": False,
                "target_calls": 0,
                "source_runs": [],
            }
            write_verified_json(path, payload)

            loaded = load_resumable_screen_manifest(
                path,
                config=config,
                base_config_digest="b" * 64,
                code_digest="c" * 64,
                dataset_version="fixture-v1",
                protocol_sha256="p" * 64,
            )

        self.assertEqual(loaded, payload)

    def test_artifact_validator_rejects_source_cache_binding_mismatch(
        self,
    ) -> None:
        from rl_transfer.phase2_screen import validate_source_run_artifacts

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run, derived = _write_source_artifact_fixture(
                output,
                binding_matches=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "binding mismatch",
            ):
                validate_source_run_artifacts(
                    run,
                    derived_config=derived,
                    run_output_dir=output,
                )

    def test_artifact_validator_accepts_complete_verified_pilot_bundle(
        self,
    ) -> None:
        from rl_transfer.phase2_screen import validate_source_run_artifacts

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run, derived = _write_source_artifact_fixture(
                output,
                binding_matches=True,
            )

            validate_source_run_artifacts(
                run,
                derived_config=derived,
                run_output_dir=output,
            )

    def test_portable_source_bundle_validates_after_root_relocation(
        self,
    ) -> None:
        from rl_transfer.phase2_screen import validate_source_run_artifacts

        with (
            tempfile.TemporaryDirectory() as source_directory,
            tempfile.TemporaryDirectory() as destination_directory,
        ):
            source = Path(source_directory) / "runs"
            source.mkdir()
            run, derived = _write_source_artifact_fixture(
                source,
                binding_matches=True,
            )
            relocated = Path(destination_directory) / "copied-runs"
            shutil.copytree(source, relocated)

            validate_source_run_artifacts(
                run,
                derived_config=derived,
                run_output_dir=relocated,
            )

            serialized = json.dumps(run, sort_keys=True)
            self.assertNotIn(source_directory, serialized)
            self.assertNotIn(destination_directory, serialized)
if __name__ == "__main__":
    unittest.main()
