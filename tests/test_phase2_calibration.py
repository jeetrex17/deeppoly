from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl_transfer.artifacts import sha256_file
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


if __name__ == "__main__":
    unittest.main()
