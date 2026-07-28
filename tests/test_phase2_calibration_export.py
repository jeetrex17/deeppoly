from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl_transfer.phase2_calibration_export import (
    CALIBRATION_EVIDENCE_FILES,
    export_phase2_calibration_evidence,
)
from rl_transfer.phase2_calibration_export_validation import (
    validate_calibration_evidence,
)
from rl_transfer.phase2_calibration_manifest import (
    CALIBRATION_TEMPERATURES,
    FOLDS,
)
from rl_transfer.phase2_temperature_output import write_verified_jsonl
from rl_transfer.research_metrics import asr_query_auc
from rl_transfer.verified_artifacts import write_verified_json


_RAW_SOURCE_FILES = {
    "calibration_manifest.json",
    "calibration_manifest.json.sha256",
    "calibration_results.jsonl",
    "calibration_results.jsonl.sha256",
    "calibration_query_traces.jsonl",
    "calibration_query_traces.jsonl.sha256",
}
_RAW_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "docs/research/cifar10_rtx_phase2_calibration"
    / "raw_calibration_records.tar.gz"
)


def _write_source(root: Path) -> Path:
    source = root / "source"
    source.mkdir(parents=True)
    with tarfile.open(_RAW_FIXTURE, mode="r:gz") as archive:
        members = archive.getmembers()
        if {member.name for member in members} != _RAW_SOURCE_FILES or any(
            not member.isfile()
            or Path(member.name).name != member.name
            or member.size < 1
            for member in members
        ):
            raise AssertionError("calibration fixture archive is unsafe")
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise AssertionError("calibration fixture member is unreadable")
            (source / member.name).write_bytes(handle.read())
    return source


class Phase2CalibrationExportTests(unittest.TestCase):
    def test_export_writes_verified_portable_evidence_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            output = root / "evidence"
            failed_log = root / "failed.log"
            rerun_log = root / "rerun.log"
            failed_log.write_text("serialization-key false negative\n")
            rerun_log.write_text("complete, target_calls=0\n")

            summary = export_phase2_calibration_evidence(
                source,
                output,
                attempt_logs=(failed_log, rerun_log),
            )

            self.assertEqual(
                {path.name for path in output.iterdir()},
                CALIBRATION_EVIDENCE_FILES,
            )
            self.assertEqual(summary["status"], "complete")
            self.assertFalse(summary["decision"]["tested_global_temperature_qualified"])
            self.assertTrue(
                summary["decision"]["stop_tested_global_temperature_protocol"]
            )
            self.assertEqual(summary["integrity"]["result_rows"], 9_000)
            self.assertEqual(summary["integrity"]["query_traces"], 60)
            self.assertEqual(summary["target_evaluation"]["target_calls"], 0)
            self.assertEqual(
                summary["integrity"]["temperature_one_reproduced_folds"],
                3,
            )
            self.assertAlmostEqual(
                summary["score_greedy"]["macro_asr"],
                0.06435270435270436,
            )
            self.assertAlmostEqual(
                summary["score_greedy"]["macro_auc"],
                0.029945386045386047,
            )

            checksums = {}
            for line in (output / "SHA256SUMS").read_text().splitlines():
                digest, filename = line.split("  ", 1)
                checksums[filename] = digest
            for path in output.iterdir():
                if path.name == "SHA256SUMS":
                    continue
                self.assertEqual(
                    checksums[path.name],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )

            with (output / "temperature_summary.csv").open(newline="") as handle:
                temperatures = list(csv.DictReader(handle))
            self.assertEqual(len(temperatures), 5)
            self.assertEqual(temperatures[0]["temperature"], "0.25")
            self.assertEqual(temperatures[-1]["temperature"], "1.5")
            self.assertEqual(
                temperatures[0]["conditions_observed_nonnegative_both"],
                "0",
            )
            self.assertIn("macro_asr", temperatures[0])
            self.assertIn("macro_auc", temperatures[0])

            published_text = "\n".join(
                path.read_text()
                for path in output.iterdir()
                if path.suffix in {".md", ".json", ".csv", ".svg", ""}
                and path.name != "SHA256SUMS"
            )
            self.assertNotIn(str(root), published_text)
            self.assertNotIn("target_family_model", published_text)

    def test_export_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            first = root / "first"
            second = root / "second"

            export_phase2_calibration_evidence(source, first)
            export_phase2_calibration_evidence(source, second)

            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )

    def test_export_rejects_tampered_raw_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            with (source / "calibration_results.jsonl").open("a") as handle:
                handle.write("{}\n")

            with self.assertRaisesRegex(ValueError, "checksum"):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                )

    def test_export_rejects_resigned_duplicate_rows_and_traces(self) -> None:
        for filename, manifest_count_key in (
            ("calibration_results.jsonl", "source_model_calls"),
            ("calibration_query_traces.jsonl", None),
        ):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = _write_source(root)
                    path = source / filename
                    records = [
                        json.loads(line) for line in path.read_text().splitlines()
                    ]
                    duplicate = dict(records[0])
                    write_verified_jsonl(path, [*records, duplicate])
                    if manifest_count_key is not None:
                        manifest_path = source / "calibration_manifest.json"
                        manifest = json.loads(manifest_path.read_text())
                        manifest[manifest_count_key] += duplicate["total_target_calls"]
                        manifest["results_sha256"] = hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        write_verified_json(manifest_path, manifest)
                    else:
                        manifest_path = source / "calibration_manifest.json"
                        manifest = json.loads(manifest_path.read_text())
                        manifest["query_traces_sha256"] = hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        write_verified_json(manifest_path, manifest)

                    with self.assertRaisesRegex(
                        ValueError,
                        "duplicate|cardinality|row|trace|identity",
                    ):
                        export_phase2_calibration_evidence(
                            source,
                            root / "evidence",
                        )

    def test_export_rejects_raw_seed_outside_the_condition_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            results_path = source / "calibration_results.jsonl"
            records = [
                json.loads(line) for line in results_path.read_text().splitlines()
            ]
            records[0]["seed"] = 0
            results_sha = write_verified_jsonl(results_path, records)
            manifest_path = source / "calibration_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["results_sha256"] = results_sha
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "seed|temperature|summary|audit|raw",
            ):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                )

    def test_export_rejects_target_calls_and_unmanaged_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            manifest_path = source / "calibration_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["target_calls"] = 1
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(ValueError, "target"):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                )

            source = _write_source(root / "second")
            output = root / "managed"
            output.mkdir()
            (output / "notes.txt").write_text("unmanaged")
            with self.assertRaisesRegex(ValueError, "unmanaged"):
                export_phase2_calibration_evidence(source, output)

    def test_export_rejects_false_frozen_condition_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            manifest_path = source / "calibration_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            first_fold = manifest["fold_summaries"][0]
            first_condition = next(iter(first_fold["conditions"].values()))
            first_condition["temperatures"]["0.25"]["frozen"] = False
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "temperature|frozen|raw|summary|audit",
            ):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                )

    def test_export_rejects_temperature_one_row_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            manifest_path = source / "calibration_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["fold_summaries"][0]["temperature_one_row_sha256"] = "0" * 64
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "temperature-one row digest|inconsistent",
            ):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                )

    def test_export_rejects_score_curve_inconsistent_with_victims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            manifest_path = source / "calibration_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            conditions = manifest["fold_summaries"][0]["conditions"]
            score = next(iter(conditions.values()))["score_greedy"]
            curve = {budget: 0.0 for budget in range(51)}
            curve[50] = score["successes"] / score["eligible"]
            score["asr_at_budgets"] = {
                str(budget): value for budget, value in curve.items()
            }
            score["asr_query_auc"] = asr_query_auc(curve)
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "aggregate curve|score|inconsistent",
            ):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                )

    def test_export_rejects_condition_image_cohort_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            manifest_path = source / "calibration_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            conditions = manifest["fold_summaries"][0]["conditions"]
            next(iter(conditions.values()))["indices_sha256"] = "0" * 64
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "image cohort|source image|different",
            ):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                )

    def test_semantic_audit_rejects_cross_fold_cohort_mismatch(self) -> None:
        grouped = {
            (heldout, source_slice, family, temperature): []
            for heldout in FOLDS
            for source_slice in (
                "exact_source",
                "seen_family_new_instance",
            )
            for family in FOLDS
            if family != heldout
            for temperature in CALIBRATION_TEMPERATURES
        }
        folds = [{"heldout_family": family} for family in FOLDS]
        with (
            mock.patch(
                "rl_transfer.phase2_calibration_export_validation._group_results",
                return_value=(grouped, 0),
            ),
            mock.patch(
                "rl_transfer.phase2_calibration_export_validation._validate_fold",
                side_effect=("cohort-a", "cohort-b", "cohort-a"),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "folds use different source image cohorts",
            ):
                validate_calibration_evidence(
                    folds,
                    {},
                    [],
                    [],
                )

    def test_export_recomputes_the_signed_calibration_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            manifest_path = source / "calibration_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            decision = manifest["calibration_decision"]
            self.assertIsInstance(decision, dict)
            decision.update(
                {
                    "calibration_useful": True,
                    "stop_temperature_only_work": False,
                    "selected_temperature": 0.5,
                    "qualifying_temperatures": [0.5],
                    "reason": "signed but contradicted by the fold metrics",
                }
            )
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "decision|qualif|recomput|inconsistent",
            ):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                )

    def test_export_rejects_signed_aggregates_that_disagree_with_folds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            manifest_path = source / "calibration_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            decision = manifest["calibration_decision"]
            self.assertIsInstance(decision, dict)
            aggregates = decision["aggregates"]
            self.assertIsInstance(aggregates, dict)
            aggregate = aggregates["0.25"]
            self.assertIsInstance(aggregate, dict)
            aggregate["mean_asr_gain_vs_score"] = 0.5
            aggregate_folds = aggregate["folds"]
            self.assertIsInstance(aggregate_folds, list)
            for fold in aggregate_folds:
                self.assertIsInstance(fold, dict)
                fold["asr"] = 0.57
                fold["asr_gain_vs_score"] = 0.5
            write_verified_json(manifest_path, manifest)

            with self.assertRaisesRegex(
                ValueError,
                "aggregate|fold|temperature|inconsistent",
            ):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                )

    def test_export_rejects_nested_private_path_before_archive_creation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            manifest_path = source / "calibration_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            folds = manifest["fold_summaries"]
            self.assertIsInstance(folds, list)
            first_fold = folds[0]
            self.assertIsInstance(first_fold, dict)
            conditions = first_fold["conditions"]
            self.assertIsInstance(conditions, dict)
            first_condition = next(iter(conditions.values()))
            self.assertIsInstance(first_condition, dict)
            first_condition["debug_metadata"] = {
                "worker": {
                    "checkpoint_path": (
                        "/Users/researcher/private/checkpoints/policy.pt"
                    )
                }
            }
            write_verified_json(manifest_path, manifest)

            with mock.patch(
                "rl_transfer.phase2_calibration_export.raw_calibration_archive",
                return_value=b"archive-must-not-be-created",
            ) as archive:
                with self.assertRaisesRegex(
                    ValueError,
                    "portable|sensitive|absolute|private|path",
                ):
                    export_phase2_calibration_evidence(
                        source,
                        root / "evidence",
                    )
                archive.assert_not_called()

    def test_export_rejects_sensitive_attempt_log_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            log = root / "password=not-a-secret.log"
            log.write_text("diagnostic output\n")

            with self.assertRaisesRegex(
                ValueError,
                "portable|sensitive",
            ):
                export_phase2_calibration_evidence(
                    source,
                    root / "evidence",
                    attempt_logs=(log,),
                )

    def test_export_rejects_malformed_locked_request_and_provenance(self) -> None:
        cases = (
            (("request", "seeds"), [0]),
            (("request", "device"), "cpu"),
            (("request", "temperatures"), [1.0]),
            (("calibration_code_digest",), "0" * 64),
            (("calibration_git_revision",), "0" * 40),
            (("dataset_content_sha256",), "0" * 64),
            (("source_manifest_sha256",), "0" * 64),
            (("calibration_git_worktree", "dirty"), False),
            (("runtime_environment", "gpu_name"), "Fake GPU"),
        )
        for path, value in cases:
            with self.subTest(path=path):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = _write_source(root)
                    manifest_path = source / "calibration_manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    destination = manifest
                    for key in path[:-1]:
                        destination = destination[key]
                    destination[path[-1]] = value
                    write_verified_json(manifest_path, manifest)

                    with self.assertRaisesRegex(
                        ValueError,
                        "request|digest|revision|provenance|locked",
                    ):
                        export_phase2_calibration_evidence(
                            source,
                            root / "evidence",
                        )

    def test_export_script_reports_the_calibration_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_source(root)
            output = root / "evidence"
            script = Path("scripts/export_phase2_calibration_evidence.py")
            specification = importlib.util.spec_from_file_location(
                "export_phase2_calibration_evidence",
                script,
            )
            self.assertIsNotNone(specification)
            self.assertIsNotNone(specification.loader)
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                module.main(
                    [
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                    ]
                )

            report = json.loads(stdout.getvalue())
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["stop_tested_global_temperature_protocol"])
            self.assertEqual(report["target_calls"], 0)


class RawCalibrationArchiveTests(unittest.TestCase):
    def test_archive_rejects_unsafe_member_names_before_opening_tar(self) -> None:
        from rl_transfer.phase2_calibration_export_archive import (
            raw_calibration_archive,
        )

        unsafe_names = (
            "../calibration_manifest.json",
            "/private/calibration_manifest.json",
            "nested/calibration_manifest.json",
        )
        for name in unsafe_names:
            with self.subTest(name=name):
                with mock.patch(
                    "rl_transfer.phase2_calibration_export_archive.tarfile.open"
                ) as open_archive:
                    with self.assertRaisesRegex(
                        ValueError,
                        "archive|member|name|path|portable",
                    ):
                        raw_calibration_archive({name: b"{}"})
                    open_archive.assert_not_called()

    def test_archive_rejects_non_bytes_before_opening_tar(self) -> None:
        from rl_transfer.phase2_calibration_export_archive import (
            raw_calibration_archive,
        )

        with mock.patch(
            "rl_transfer.phase2_calibration_export_archive.tarfile.open"
        ) as open_archive:
            with self.assertRaisesRegex(TypeError, "bytes|payload"):
                raw_calibration_archive(
                    {"calibration_manifest.json": "not bytes"}  # type: ignore[dict-item]
                )
            open_archive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
