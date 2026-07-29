from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest

from rl_transfer.phase2_residual_d1_evidence import (
    load_verified_jsonl_records,
    verify_d1_raw_evidence,
    verify_d1_recorded_summaries,
    write_d1_evidence_plots,
)
from rl_transfer.phase2_residual_d1_teacher import _write_verified_jsonl


def _row(family: str, method: str, *, success: bool) -> dict[str, object]:
    victim_id = f"{family}-source-victim"
    sample_id = f"cifar10:{family}:{victim_id}:123"
    return {
        "sample_id": sample_id,
        "victim_id": victim_id,
        "victim_family": family,
        "method": method,
        "threat_model": "T1",
        "seed": 17,
        "query_budget": 50,
        "clean_correct": True,
        "success": success,
        "query_to_success": 2 if success else None,
        "total_target_calls": 2,
        "linf": 2 / 255,
        "l2": 0.02,
        "policy_digest": "a" * 64,
        "action_trace": [3],
        "heldout_family": "modern_cnn",
        "source_slice": "seen_family_new_instance",
        "target_calls": 0,
        "hidden_target_calls": 0,
    }


def _trace(row: dict[str, object]) -> dict[str, object]:
    return {
        "method": row["method"],
        "sample_id": row["sample_id"],
        "victim_id": row["victim_id"],
        "family": row["victim_family"],
        "victim_family": row["victim_family"],
        "clean_correct": True,
        "success": row["success"],
        "query_to_success": row["query_to_success"],
        "total_target_calls": 2,
        "linf": row["linf"],
        "l2": row["l2"],
        "actions": [3],
        "policy_digest_before": "a" * 64,
        "policy_digest_after": "a" * 64,
        "query_trace": [
            {
                "call_index": 1,
                "sample_id": row["sample_id"],
                "victim_id": row["victim_id"],
                "feedback": "scores",
                "purpose": "initialization",
                "error": None,
            },
            {
                "call_index": 2,
                "sample_id": row["sample_id"],
                "victim_id": row["victim_id"],
                "feedback": "scores",
                "purpose": "residual-ranker-fallback",
                "error": None,
            },
        ],
        "heldout_family": "modern_cnn",
        "source_slice": "seen_family_new_instance",
        "target_calls": 0,
        "hidden_target_calls": 0,
    }


class ResidualD1EvidenceTests(unittest.TestCase):
    def _records(self):
        rows = tuple(
            _row(
                family,
                method,
                success=(family == "classical_cnn" and method == "residual_ranker_bc"),
            )
            for family in ("classical_cnn", "transformer")
            for method in ("score_greedy", "residual_ranker_bc")
        )
        return rows, tuple(_trace(row) for row in rows)

    def test_raw_rows_and_full_traces_recompute_verified_metrics_and_plots(
        self,
    ) -> None:
        rows, traces = self._records()

        verified = verify_d1_raw_evidence(
            rows,
            traces,
            expected_methods=("score_greedy", "residual_ranker_bc"),
        )

        self.assertTrue(verified["verified"])
        self.assertTrue(verified["cohorts_exactly_paired"])
        self.assertEqual(verified["rows"], 4)
        self.assertEqual(verified["full_query_traces"], 4)
        self.assertEqual(verified["hidden_target_calls"], 0)
        self.assertEqual(
            verified["macro"]["residual_ranker_bc"]["final_asr"],
            0.5,
        )
        conditions = {
            family: {"methods": copy.deepcopy(verified["by_family"][family])}
            for family in ("classical_cnn", "transformer")
        }
        verify_d1_recorded_summaries(verified, conditions)
        conditions["classical_cnn"]["methods"]["score_greedy"]["successes"] = 99
        with self.assertRaisesRegex(ValueError, "raw recomputation"):
            verify_d1_recorded_summaries(verified, conditions)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_d1_evidence_plots(Path(directory), verified)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue(
                all(
                    path.with_suffix(path.suffix + ".sha256").is_file()
                    for path in paths
                )
            )
            self.assertTrue(all(path.read_text().startswith("<svg") for path in paths))

    def test_verified_jsonl_rejects_tampering_and_evidence_rejects_cohort_drift(
        self,
    ) -> None:
        rows, traces = self._records()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            _write_verified_jsonl(path, rows)
            self.assertEqual(len(load_verified_jsonl_records(path)), 4)
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "checksum|framing"):
                load_verified_jsonl_records(path)

        drifted = (
            *rows[:-1],
            {**rows[-1], "sample_id": f"{rows[-1]['sample_id']}-drift"},
        )
        with self.assertRaisesRegex(ValueError, "trace|match|cohort"):
            verify_d1_raw_evidence(
                drifted,
                traces,
                expected_methods=("score_greedy", "residual_ranker_bc"),
            )

    def test_raw_evidence_rejects_hidden_target_calls(self) -> None:
        rows, traces = self._records()
        contaminated_rows = (
            {**rows[0], "hidden_target_calls": 1},
            *rows[1:],
        )
        with self.assertRaisesRegex(ValueError, "hidden-target|hidden_target"):
            verify_d1_raw_evidence(
                contaminated_rows,
                traces,
                expected_methods=("score_greedy", "residual_ranker_bc"),
            )

        contaminated_traces = (
            {**traces[0], "hidden_target_calls": 1},
            *traces[1:],
        )
        with self.assertRaisesRegex(ValueError, "hidden-target|hidden_target"):
            verify_d1_raw_evidence(
                rows,
                contaminated_traces,
                expected_methods=("score_greedy", "residual_ranker_bc"),
            )

    def test_raw_evidence_rejects_invalid_actions_and_query_events(self) -> None:
        rows, traces = self._records()
        invalid_rows = (
            {**rows[0], "action_trace": [96]},
            *rows[1:],
        )
        with self.assertRaisesRegex(ValueError, "attack contract"):
            verify_d1_raw_evidence(
                invalid_rows,
                traces,
                expected_methods=("score_greedy", "residual_ranker_bc"),
            )

        for field, value in (
            ("purpose", ""),
            ("purpose", "initialization"),
            ("error", "victim failure"),
        ):
            with self.subTest(field=field, value=value):
                events = copy.deepcopy(traces[0]["query_trace"])
                events[1][field] = value
                invalid_traces = (
                    {**traces[0], "query_trace": events},
                    *traces[1:],
                )
                with self.assertRaisesRegex(ValueError, "query trace"):
                    verify_d1_raw_evidence(
                        rows,
                        invalid_traces,
                        expected_methods=(
                            "score_greedy",
                            "residual_ranker_bc",
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
