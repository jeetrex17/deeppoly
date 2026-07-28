from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from rl_transfer.artifacts import sha256_file
from rl_transfer.cifar_config import MacPilotConfig
from rl_transfer.phase2_calibration_manifest import (
    CALIBRATION_MAX_SECONDS,
    CALIBRATION_TEMPERATURES,
    Phase2CalibrationRequest,
)
from rl_transfer.phase2_calibration_screen import (
    build_calibration_dry_run,
    run_calibration_from_datasets,
    select_calibration_temperature,
)
from rl_transfer.verified_artifacts import load_verified_json


FOLDS = ("classical_cnn", "modern_cnn", "transformer")


def _request(
    root: Path,
    **overrides: object,
) -> Phase2CalibrationRequest:
    source_root = root / "phase2-source"
    source_root.mkdir(parents=True, exist_ok=True)
    source_manifest = source_root / "screen_manifest.json"
    source_manifest.write_text("{}")
    values: dict[str, object] = {
        "source_manifest": source_manifest,
        "source_root": source_root,
        "output_dir": root / "calibration",
        "data_root": root / "data",
        "seeds": (17,),
        "folds": FOLDS,
        "temperatures": CALIBRATION_TEMPERATURES,
        "deadline_seconds": CALIBRATION_MAX_SECONDS,
    }
    values.update(overrides)
    return Phase2CalibrationRequest(**values)


def _source_mapping(
    request: Phase2CalibrationRequest,
    *,
    target_calls: int = 0,
) -> dict[str, object]:
    folds = tuple(
        {
            "seed": 17,
            "heldout_family": heldout_family,
            "checkpoint_path": (
                request.source_root / "runs" / f"fold-{index}" / "policy.pt"
            ),
            "checkpoint_sha256": f"{index + 1:064x}",
            "persistent_digest": f"{index + 4:064x}",
            "score_rows_path": (
                request.source_root / "runs" / f"fold-{index}" / "source_results.jsonl"
            ),
        }
        for index, heldout_family in enumerate(FOLDS)
    )
    return {
        "manifest_path": request.source_manifest,
        "manifest_sha256": "a" * 64,
        "dataset_version": "in-memory",
        "dataset_content_sha256": None,
        "target_calls": target_calls,
        "target_evaluation_performed": False,
        "folds": folds,
    }


def _temperature_values(
    *,
    default: tuple[float, float] = (0.09, 0.03),
    overrides: dict[float, tuple[float, float]] | None = None,
) -> dict[float, tuple[float, float]]:
    values = {temperature: default for temperature in CALIBRATION_TEMPERATURES}
    values.update(overrides or {})
    return values


def _fold_summary(
    heldout_family: str,
    values: dict[float, tuple[float, float]],
    *,
    score: tuple[float, float] = (0.10, 0.04),
    target_calls: int = 0,
) -> dict[str, object]:
    return {
        "seed": 17,
        "heldout_family": heldout_family,
        "complete": True,
        "score_greedy": {"asr": score[0], "auc": score[1]},
        "temperatures": {
            str(temperature): {
                "asr": asr,
                "auc": auc,
                "frozen": True,
            }
            for temperature, (asr, auc) in values.items()
        },
        "raw_evidence_audited": True,
        "target_calls": target_calls,
        "target_evaluation_performed": False,
    }


class _SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)
        self._last = values[-1]

    def __call__(self) -> float:
        self._last = next(self._values, self._last)
        return self._last


class Phase2CalibrationRequestTests(unittest.TestCase):
    def test_request_locks_seed_folds_temperatures_and_fifteen_minutes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)

            self.assertEqual(
                CALIBRATION_TEMPERATURES,
                (0.25, 0.5, 0.75, 1.0, 1.5),
            )
            self.assertEqual(CALIBRATION_MAX_SECONDS, 900.0)
            self.assertEqual(request.seeds, (17,))
            self.assertEqual(request.folds, FOLDS)
            self.assertEqual(
                request.temperatures,
                CALIBRATION_TEMPERATURES,
            )
            self.assertEqual(request.deadline_seconds, 900.0)

            invalid = (
                {"seeds": (19,)},
                {"folds": FOLDS[:2]},
                {"folds": tuple(reversed(FOLDS))},
                {"temperatures": (0.5, 1.0)},
                {"deadline_seconds": 0.0},
                {"deadline_seconds": 900.01},
            )
            for overrides in invalid:
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    _request(root, **overrides)

    def test_output_tree_cannot_overlap_phase2_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "phase2-source"
            overlapping_outputs = (
                source_root,
                source_root / "calibration",
                root,
            )

            for output_dir in overlapping_outputs:
                with (
                    self.subTest(output_dir=output_dir),
                    self.assertRaisesRegex(
                        ValueError,
                        "output|source|overlap",
                    ),
                ):
                    _request(root, output_dir=output_dir)

    def test_download_root_cannot_overlap_sealed_artifact_trees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "phase2-source"

            with self.assertRaisesRegex(ValueError, "data|overlap"):
                _request(
                    root,
                    data_root=source_root / "data",
                    download=True,
                )


class Phase2CalibrationDryRunTests(unittest.TestCase):
    def test_dry_run_is_training_free_and_has_no_target_surface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            source = _source_mapping(request)

            plan = build_calibration_dry_run(request, source)

            self.assertEqual(
                plan["mode"],
                "source_only_phase2_checkpoint_calibration_diagnostic",
            )
            self.assertEqual(
                plan["selected_policy_seeds"],
                [17],
            )
            self.assertEqual(plan["selected_folds"], list(FOLDS))
            self.assertEqual(
                plan["temperatures"],
                list(CALIBRATION_TEMPERATURES),
            )
            self.assertEqual(
                plan["maximum_wall_clock_seconds"],
                900.0,
            )
            self.assertFalse(plan["training_performed"])
            self.assertFalse(plan["target_evaluation_available"])
            self.assertEqual(plan["target_calls"], 0)


class CalibrationTemperatureSelectionTests(unittest.TestCase):
    def test_temperature_is_useful_only_when_it_matches_score_in_every_fold(
        self,
    ) -> None:
        winner_values = (
            (0.11, 0.05),
            (0.10, 0.04),
            (0.12, 0.045),
        )
        summaries = [
            _fold_summary(
                fold,
                _temperature_values(overrides={0.5: winner_values[index]}),
            )
            for index, fold in enumerate(FOLDS)
        ]

        decision = select_calibration_temperature(summaries)

        self.assertTrue(decision["complete"])
        self.assertTrue(decision["calibration_useful"])
        self.assertFalse(decision["stop_temperature_only_work"])
        self.assertEqual(decision["selected_temperature"], 0.5)
        self.assertEqual(
            tuple(decision["qualifying_temperatures"]),
            (0.5,),
        )

    def test_macro_average_cannot_hide_one_failed_fold(self) -> None:
        candidate_values = (
            (0.20, 0.08),
            (0.20, 0.08),
            (0.099, 0.08),
        )
        summaries = [
            _fold_summary(
                fold,
                _temperature_values(overrides={0.5: candidate_values[index]}),
            )
            for index, fold in enumerate(FOLDS)
        ]

        decision = select_calibration_temperature(summaries)

        self.assertTrue(decision["complete"])
        self.assertFalse(decision["calibration_useful"])
        self.assertTrue(decision["stop_temperature_only_work"])
        self.assertIsNone(decision["selected_temperature"])
        self.assertEqual(decision["qualifying_temperatures"], [])

    def test_selector_requires_all_three_complete_folds(self) -> None:
        summaries = [
            _fold_summary(
                fold,
                _temperature_values(overrides={0.5: (0.11, 0.05)}),
            )
            for fold in FOLDS[:2]
        ]

        decision = select_calibration_temperature(summaries)

        self.assertFalse(decision["complete"])
        self.assertFalse(decision["calibration_useful"])
        self.assertTrue(decision["stop_temperature_only_work"])
        self.assertIsNone(decision["selected_temperature"])

    def test_exact_metric_tie_is_deterministic_and_uses_lower_temperature(
        self,
    ) -> None:
        summaries = [
            _fold_summary(
                fold,
                _temperature_values(
                    overrides={
                        0.5: (0.12, 0.06),
                        0.75: (0.12, 0.06),
                    }
                ),
            )
            for fold in reversed(FOLDS)
        ]

        first = select_calibration_temperature(summaries)
        second = select_calibration_temperature(tuple(reversed(summaries)))

        self.assertEqual(first["selected_temperature"], 0.5)
        self.assertEqual(second["selected_temperature"], 0.5)
        self.assertEqual(
            tuple(first["qualifying_temperatures"]),
            (0.5, 0.75),
        )


class Phase2CalibrationRunnerTests(unittest.TestCase):
    def test_runner_refuses_to_overwrite_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            request.output_dir.mkdir(parents=True)
            (request.output_dir / "prior.json").write_text("{}")

            with self.assertRaisesRegex(ValueError, "output|overwrite"):
                run_calibration_from_datasets(
                    request,
                    object(),
                    object(),
                    source_loader=lambda _request: _source_mapping(request),
                    fold_evaluator=mock.Mock(),
                    clock=lambda: 0.0,
                )

    def test_injected_runner_stops_at_deadline_and_writes_verified_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            source = _source_mapping(request)
            evaluated: list[str] = []

            def evaluator(*, fold, **_kwargs):
                heldout_family = fold["heldout_family"]
                evaluated.append(heldout_family)
                summary = _fold_summary(
                    heldout_family,
                    _temperature_values(overrides={0.5: (0.11, 0.05)}),
                )
                rows = [
                    {
                        "seed": 17,
                        "heldout_family": heldout_family,
                        "method": "learned_temperature",
                        "temperature": 0.5,
                        "target_calls": 0,
                    }
                ]
                traces = [
                    {
                        "sample_id": f"{heldout_family}:0",
                        "method": "learned_temperature",
                        "temperature": 0.5,
                        "target_calls": 0,
                    }
                ]
                return summary, rows, traces

            result = run_calibration_from_datasets(
                request,
                object(),
                object(),
                source_loader=lambda actual_request: (
                    source
                    if actual_request is request
                    else self.fail("runner replaced the request")
                ),
                fold_evaluator=evaluator,
                clock=_SequenceClock(0.0, 0.0, 901.0, 901.0),
            )

            self.assertEqual(evaluated, [FOLDS[0]])
            self.assertEqual(result["status"], "deadline_reached")
            self.assertTrue(result["deadline_reached"])
            self.assertEqual(result["completed_folds"], 1)
            self.assertEqual(result["selected_folds"], 3)
            self.assertFalse(result["training_performed"])
            self.assertFalse(result["target_evaluation_available"])
            self.assertFalse(result["target_evaluation_performed"])
            self.assertEqual(result["target_calls"], 0)

            manifest_path = request.output_dir / "calibration_manifest.json"
            manifest = load_verified_json(manifest_path)
            self.assertEqual(manifest, result)
            self.assertEqual(
                manifest["source_manifest_sha256"],
                source["manifest_sha256"],
            )

            rows_path = request.output_dir / "calibration_results.jsonl"
            traces_path = request.output_dir / "calibration_query_traces.jsonl"
            for path in (rows_path, traces_path):
                with self.subTest(path=path):
                    self.assertTrue(path.is_file())
                    sidecar = path.with_suffix(path.suffix + ".sha256")
                    self.assertTrue(sidecar.is_file())
                    self.assertEqual(
                        sidecar.read_text().strip(),
                        sha256_file(path),
                    )
            self.assertEqual(
                len(rows_path.read_text().splitlines()),
                1,
            )
            self.assertEqual(
                len(traces_path.read_text().splitlines()),
                1,
            )
            persisted_row = json.loads(rows_path.read_text())
            self.assertEqual(
                persisted_row["heldout_family"],
                FOLDS[0],
            )
            self.assertEqual(persisted_row["target_calls"], 0)

    def test_method_deadline_preserves_audited_partial_calls(self) -> None:
        from rl_transfer.phase2_calibration_evaluation import (
            CalibrationDeadlineReached,
        )

        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            source = _source_mapping(request)
            partial_row = {
                "method": "temperature_0p25",
                "total_target_calls": 5,
                "hidden_target_calls": 0,
            }
            error = CalibrationDeadlineReached(
                "deadline",
                rows=(partial_row,),
                traces=({"method": "temperature_0p25"},),
                partial_fold={
                    "seed": 17,
                    "heldout_family": FOLDS[0],
                    "source_model_calls": 5,
                    "raw_evidence_audited": True,
                    "target_calls": 0,
                    "target_evaluation_performed": False,
                    "complete": False,
                },
            )

            result = run_calibration_from_datasets(
                request,
                object(),
                object(),
                source_loader=lambda _request: source,
                fold_evaluator=mock.Mock(side_effect=error),
                clock=lambda: 0.0,
            )

            self.assertEqual(result["status"], "deadline_reached")
            self.assertEqual(result["completed_folds"], 0)
            self.assertEqual(result["partial_folds"], 1)
            self.assertEqual(result["source_model_calls"], 5)
            self.assertEqual(
                len(
                    (request.output_dir / "calibration_results.jsonl")
                    .read_text()
                    .splitlines()
                ),
                1,
            )

    def test_runner_rejects_source_or_fold_target_calls(self) -> None:
        cases = ("source", "fold")
        for violation in cases:
            with self.subTest(violation=violation):
                with tempfile.TemporaryDirectory() as directory:
                    request = _request(Path(directory))
                    source = _source_mapping(
                        request,
                        target_calls=1 if violation == "source" else 0,
                    )
                    evaluator = mock.Mock(
                        return_value=(
                            _fold_summary(
                                FOLDS[0],
                                _temperature_values(),
                                target_calls=(1 if violation == "fold" else 0),
                            ),
                            [],
                            [],
                        )
                    )

                    with self.assertRaisesRegex(ValueError, "target"):
                        run_calibration_from_datasets(
                            request,
                            object(),
                            object(),
                            source_loader=lambda _request: source,
                            fold_evaluator=evaluator,
                            clock=lambda: 0.0,
                        )

                    self.assertFalse(
                        (request.output_dir / "calibration_manifest.json").exists()
                    )
                    if violation == "source":
                        evaluator.assert_not_called()

    def test_runner_rejects_an_unaudited_fold_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            summary = _fold_summary(
                FOLDS[0],
                _temperature_values(),
            )
            del summary["raw_evidence_audited"]

            with self.assertRaisesRegex(ValueError, "audit|evidence"):
                run_calibration_from_datasets(
                    request,
                    object(),
                    object(),
                    source_loader=lambda _request: _source_mapping(request),
                    fold_evaluator=mock.Mock(return_value=(summary, [], [])),
                    clock=lambda: 0.0,
                )

    def test_calibration_api_exposes_no_target_evaluation_entrypoint(
        self,
    ) -> None:
        prohibited = {
            "evaluate_target",
            "target_dataset",
            "target_evaluator",
            "target_family",
        }
        request_parameters = set(inspect.signature(Phase2CalibrationRequest).parameters)
        runner_parameters = set(
            inspect.signature(run_calibration_from_datasets).parameters
        )

        self.assertTrue(prohibited.isdisjoint(request_parameters))
        self.assertTrue(prohibited.isdisjoint(runner_parameters))

        source = Path("rl_transfer/phase2_calibration_screen.py").read_text()
        self.assertNotIn("target_dataset", source)
        self.assertNotIn("target_evaluator", source)
        self.assertNotIn("run_target_evaluation", source)


class Phase2CalibrationReplayTests(unittest.TestCase):
    def test_temperature_one_replay_must_match_verified_source_evidence(
        self,
    ) -> None:
        from rl_transfer.phase2_calibration_evaluation import (
            verify_temperature_one_reproduction,
        )

        metrics = {
            "eligible": 25,
            "successes": 3,
            "asr_at_budgets": {"0": 0.0, "50": 0.12},
            "asr_query_auc": 0.05,
            "eligible_sample_ids_sha256": "a" * 64,
            "action_histogram": {"0": 10, "1": 15},
        }

        verify_temperature_one_reproduction(metrics, dict(metrics))
        for field, changed in (
            ("successes", 4),
            ("asr_query_auc", 0.051),
            ("eligible_sample_ids_sha256", "b" * 64),
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(
                    ValueError,
                    "reproduce",
                ),
            ):
                verify_temperature_one_reproduction(
                    {**metrics, field: changed},
                    metrics,
                )

    def test_temperature_method_names_are_stable(self) -> None:
        from rl_transfer.phase2_calibration_evaluation import (
            temperature_method,
        )

        self.assertEqual(temperature_method(0.25), "temperature_0p25")
        self.assertEqual(temperature_method(1.0), "temperature_1p00")
        self.assertEqual(temperature_method(1.5), "temperature_1p50")

    def test_fold_aggregation_is_macro_over_four_conditions(self) -> None:
        from rl_transfer.phase2_calibration_evaluation import _pool_metrics

        pooled = _pool_metrics(
            (
                {
                    "eligible": 100,
                    "successes": 100,
                    "asr_query_auc": 1.0,
                    "frozen": True,
                    "source_model_calls": 500,
                },
                {
                    "eligible": 1,
                    "successes": 0,
                    "asr_query_auc": 0.0,
                    "frozen": True,
                    "source_model_calls": 5,
                },
            )
        )

        self.assertEqual(pooled["eligible"], 101)
        self.assertEqual(pooled["successes"], 100)
        self.assertEqual(pooled["asr"], 0.5)
        self.assertEqual(pooled["auc"], 0.5)
        self.assertEqual(pooled["source_model_calls"], 505)

    def test_temperature_one_replay_matches_exact_sample_rows(self) -> None:
        from rl_transfer.phase2_calibration_evaluation import (
            verify_temperature_one_rows,
        )
        from rl_transfer.results import ResearchResultRow

        row = ResearchResultRow(
            sample_id="sample",
            victim_id="victim",
            victim_family="modern_cnn",
            method="original",
            threat_model="T1",
            seed=17,
            query_budget=50,
            clean_correct=True,
            success=True,
            query_to_success=5,
            total_target_calls=5,
            linf=0.01,
            l2=0.02,
            policy_digest="p",
            action_trace=(1, 2, 3, 4),
        )
        replayed = replace(row, method="temperature_1p00")

        digest = verify_temperature_one_rows((replayed,), (row,))
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        with self.assertRaisesRegex(ValueError, "sample|reproduce"):
            verify_temperature_one_rows(
                (replace(replayed, linf=0.02),),
                (row,),
            )

    def test_fold_replay_pools_all_verified_source_conditions(self) -> None:
        from rl_transfer.phase2_calibration_evaluation import (
            evaluate_calibration_fold,
        )
        from rl_transfer.results import ResearchResultRow

        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            base = MacPilotConfig.from_json(
                Path("configs/rl_transfer/cifar10_rtx_phase2_base.json")
            )
            config = replace(base, target_family="classical_cnn")
            learned = "soft_gradient_bc_action_conditioned_groupdro_ppo_stochastic"
            source_families = ("modern_cnn", "transformer")
            metric = {
                "eligible": 1,
                "successes": 1,
                "asr_at_budgets": {0: 0.0, 50: 1.0},
                "asr_query_auc": 0.5,
                "eligible_sample_ids_sha256": "e" * 64,
                "action_histogram": {"0": 1},
                "frozen": True,
                "policy_digest_before": "p",
                "policy_digest_after": "p",
                "source_model_calls": 5,
            }
            verified_family = {
                "score_greedy": {
                    **metric,
                    "successes": 0,
                    "asr_query_auc": 0.25,
                },
                learned: dict(metric),
            }
            run = {
                "config": config.__dict__,
                "policy": {
                    "checkpoints": {
                        "main": {"method_id": learned.removesuffix("_stochastic")}
                    }
                },
                "source_evaluation": {
                    source_slice: {
                        family: verified_family for family in source_families
                    }
                    for source_slice in (
                        "exact_source",
                        "seen_family_new_instance",
                    )
                },
            }
            fold = {
                "heldout_family": "classical_cnn",
                "source_families": source_families,
                "checkpoint_sha256": "c" * 64,
                "persistent_digest": "p",
                "score_rows_path": Path(directory) / "source_results.jsonl",
                "run_manifest": run,
            }

            class FrozenPolicy:
                def persistent_digest(self) -> str:
                    return "p"

            populations = {
                source_slice: {
                    family: ((f"{source_slice}-{family}", nn.Identity()),)
                    for family in source_families
                }
                for source_slice in (
                    "exact_source",
                    "seen_family_new_instance",
                )
            }
            original_rows = []
            for slice_offset, source_slice in enumerate(populations):
                for family_offset, family in enumerate(source_families):
                    victim_id = populations[source_slice][family][0][0]
                    original_rows.append(
                        ResearchResultRow(
                            sample_id=f"cifar10:{family}:{victim_id}:1",
                            victim_id=victim_id,
                            victim_family=family,
                            method=learned,
                            threat_model="T1",
                            seed=(800_017 + 10_000 * slice_offset + family_offset),
                            query_budget=50,
                            clean_correct=True,
                            success=True,
                            query_to_success=5,
                            total_target_calls=5,
                            linf=0.01,
                            l2=0.02,
                            policy_digest="p",
                            action_trace=(0,),
                        )
                    )

            def selected_evaluator(methods, *_args, **_kwargs):
                method = next(iter(methods))
                victims = _args[0]
                condition_seed = _args[4]
                family = _args[5]
                victim_id = victims[0][0]
                row = ResearchResultRow(
                    sample_id=f"cifar10:{family}:{victim_id}:1",
                    victim_id=victim_id,
                    victim_family=family,
                    method=method,
                    threat_model="T1",
                    seed=condition_seed,
                    query_budget=50,
                    clean_correct=True,
                    success=True,
                    query_to_success=5,
                    total_target_calls=5,
                    linf=0.01,
                    l2=0.02,
                    policy_digest="p",
                    action_trace=(0,),
                )
                return [row], [{"method": method}], {method: dict(metric)}

            with (
                mock.patch(
                    "rl_transfer.phase2_calibration_evaluation._load_policy",
                    return_value=FrozenPolicy(),
                ),
                mock.patch(
                    "rl_transfer.phase2_calibration_evaluation._load_source_victims",
                    return_value=populations,
                ),
                mock.patch(
                    "rl_transfer.phase2_calibration_evaluation._source_indices",
                    return_value=(1,),
                ),
                mock.patch(
                    "rl_transfer.phase2_calibration_evaluation.dataset_samples",
                    return_value=((torch.zeros(3, 4, 4), 0),),
                ),
                mock.patch(
                    "rl_transfer.phase2_calibration_evaluation.read_jsonl",
                    return_value=original_rows,
                ),
                mock.patch(
                    "rl_transfer.phase2_calibration_evaluation.audit_evaluation",
                    return_value={"passed": True},
                ),
                mock.patch(
                    "rl_transfer.phase2_calibration_evaluation.FrozenTemperaturePolicy",
                    side_effect=lambda policy, temperature: policy,
                ),
                mock.patch(
                    "rl_transfer.phase2_calibration_evaluation.evaluate_method_set",
                    side_effect=selected_evaluator,
                ) as evaluator,
            ):
                summary, rows, traces = evaluate_calibration_fold(
                    fold=fold,
                    request=request,
                    train_dataset=object(),
                    test_dataset=object(),
                    absolute_deadline=900.0,
                    clock=lambda: 0.0,
                    progress=lambda _message: None,
                )

            self.assertTrue(summary["complete"])
            self.assertTrue(summary["temperature_one_reproduced"])
            self.assertEqual(summary["source_model_calls"], 100)
            self.assertEqual(len(summary["conditions"]), 4)
            self.assertEqual(len(rows), 20)
            self.assertEqual(len(traces), 20)
            self.assertEqual(evaluator.call_count, 20)
            self.assertEqual(summary["temperatures"]["1.0"]["asr"], 1.0)
            self.assertEqual(
                summary["temperatures"]["1.0"]["auc_gain_vs_score"],
                0.25,
            )
            self.assertTrue(all(row["hidden_target_calls"] == 0 for row in rows))

    def test_verified_phase2_manifest_loads_only_source_folds(self) -> None:
        from rl_transfer.phase2_calibration_manifest import (
            load_phase2_calibration_source,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            base = MacPilotConfig.from_json(
                Path("configs/rl_transfer/cifar10_rtx_phase2_base.json")
            )
            runs = []
            checkpoint_digest = "c" * 64
            for index, heldout in enumerate(FOLDS):
                run_dir = request.source_root / "runs" / f"fold-{index}"
                run_dir.mkdir(parents=True)
                (run_dir / "policy.pt").write_bytes(b"checkpoint")
                source_families = tuple(family for family in FOLDS if family != heldout)
                config = replace(base, target_family=heldout)
                runs.append(
                    {
                        "seed": 17,
                        "target_family": heldout,
                        "source_families": list(source_families),
                        "run_dir": f"fold-{index}",
                        "config": config.__dict__,
                        "policy": {
                            "checkpoint": "policy.pt",
                            "checkpoint_sha256": checkpoint_digest,
                            "persistent_digest": "d" * 64,
                        },
                    }
                )
            study = {
                "schema_version": 1,
                "status": "screen_complete",
                "research_valid": False,
                "target_calls": 0,
                "target_evaluation_performed": False,
                "dataset_version": ("in-memory;content-sha256=" + "f" * 64),
                "source_runs": runs,
            }

            with (
                mock.patch(
                    "rl_transfer.phase2_calibration_manifest.load_verified_json",
                    return_value=study,
                ),
                mock.patch(
                    "rl_transfer.phase2_calibration_manifest."
                    "validate_source_run_artifacts"
                ) as validator,
                mock.patch(
                    "rl_transfer.phase2_calibration_manifest.sha256_file",
                    side_effect=lambda path: (
                        checkpoint_digest
                        if Path(path).name == "policy.pt"
                        else "a" * 64
                    ),
                ),
            ):
                source = load_phase2_calibration_source(request)

            self.assertEqual(len(source["folds"]), 3)
            self.assertEqual(validator.call_count, 3)
            self.assertEqual(source["target_calls"], 0)
            self.assertEqual(source["dataset_content_sha256"], "f" * 64)
            self.assertEqual(
                tuple(fold["heldout_family"] for fold in source["folds"]),
                FOLDS,
            )


class Phase2CalibrationCliTests(unittest.TestCase):
    def test_cli_dry_run_prints_the_sealed_plan(self) -> None:
        from rl_transfer import phase2_calibration_cli

        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            source = _source_mapping(request)
            arguments = (
                "--source-manifest",
                str(request.source_manifest),
                "--source-root",
                str(request.source_root),
                "--output-dir",
                str(request.output_dir),
                "--dry-run",
            )
            with (
                mock.patch.object(
                    phase2_calibration_cli,
                    "load_phase2_calibration_source",
                    return_value=source,
                ),
                mock.patch("builtins.print") as printer,
            ):
                status = phase2_calibration_cli.main(arguments)

            self.assertEqual(status, 0)
            payload = json.loads(printer.call_args.args[0])
            self.assertEqual(payload["target_calls"], 0)
            self.assertFalse(payload["training_performed"])

    def test_cli_fails_before_dataset_loading_without_cuda(self) -> None:
        from rl_transfer import phase2_calibration_cli

        with tempfile.TemporaryDirectory() as directory:
            request = _request(Path(directory))
            source = _source_mapping(request)
            arguments = (
                "--source-manifest",
                str(request.source_manifest),
                "--source-root",
                str(request.source_root),
                "--output-dir",
                str(request.output_dir),
            )
            with (
                mock.patch.object(
                    phase2_calibration_cli,
                    "load_phase2_calibration_source",
                    return_value=source,
                ),
                mock.patch.object(
                    phase2_calibration_cli.torch.cuda,
                    "is_available",
                    return_value=False,
                ),
                self.assertRaisesRegex(RuntimeError, "CUDA"),
            ):
                phase2_calibration_cli.main(arguments)


if __name__ == "__main__":
    unittest.main()
