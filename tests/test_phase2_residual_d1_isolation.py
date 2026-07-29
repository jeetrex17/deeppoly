from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from rl_transfer.phase2_residual_d1 import (
    validate_residual_source_records,
    validate_source_only_payload,
)
from rl_transfer.phase2_residual_d1_cli import _request, build_parser
from rl_transfer.phase2_residual_d1_source import validate_d1_role_indices


class ResidualSourceIsolationTests(unittest.TestCase):
    _SOURCE_RECORDS = (
        {
            "victim_family": "classical_cnn",
            "victim_id": "classical-source-0",
            "heldout_family": "modern_cnn",
            "target_calls": 0,
            "hidden_target_calls": 0,
        },
        {
            "victim_family": "transformer",
            "victim_id": "transformer-source-0",
            "heldout_family": "modern_cnn",
            "target_calls": 0,
            "hidden_target_calls": 0,
        },
    )

    def test_target_calls_are_rejected(self) -> None:
        leaked = (
            {**self._SOURCE_RECORDS[0], "target_calls": 1},
            self._SOURCE_RECORDS[1],
        )

        with self.assertRaisesRegex(ValueError, "target|source.?only|call"):
            validate_residual_source_records(
                leaked,
                heldout_family="modern_cnn",
            )

    def test_audited_source_query_fields_are_not_hidden_target_calls(self) -> None:
        payload = {
            "methods": {
                "score_greedy": {
                    "total_target_calls": 123,
                    "max_total_target_calls": 50,
                    "hidden_target_calls": 0,
                }
            },
            "required_target_calls": 0,
            "target_calls": 0,
            "target_evaluation_performed": False,
        }

        validate_source_only_payload(payload, "real-shaped source summary")
        contaminated = {
            **payload,
            "methods": {
                "score_greedy": {
                    **payload["methods"]["score_greedy"],
                    "hidden_target_calls": 1,
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "hidden_target_calls"):
            validate_source_only_payload(contaminated, "contaminated summary")

    def test_heldout_victim_contamination_is_rejected(self) -> None:
        contaminated = (
            self._SOURCE_RECORDS[0],
            {
                "victim_family": "modern_cnn",
                "victim_id": "heldout-modern-0",
                "heldout_family": "modern_cnn",
                "target_calls": 0,
                "hidden_target_calls": 0,
            },
        )

        with self.assertRaisesRegex(
            ValueError,
            "held.?out|contamin|source.?only|modern_cnn",
        ):
            validate_residual_source_records(
                contaminated,
                heldout_family="modern_cnn",
            )


class ResidualRoleCohortTests(unittest.TestCase):
    _ROLES = {
        "train": tuple(range(0, 200)),
        "threshold": tuple(range(200, 250)),
        "competence": tuple(range(250, 300)),
        "d1a_evaluation": tuple(range(300, 350)),
        "d1b_evaluation": tuple(range(350, 400)),
    }

    def test_all_five_roles_are_exactly_sized_and_pairwise_disjoint(self) -> None:
        audit = validate_d1_role_indices(self._ROLES)

        self.assertTrue(audit["pairwise_disjoint"])
        self.assertEqual(audit["role_sizes"]["train"], 200)
        self.assertEqual(audit["role_sizes"]["d1b_evaluation"], 50)
        self.assertEqual(set(audit["role_indices_sha256"]), set(self._ROLES))

    def test_every_pairwise_overlap_and_duplicate_is_rejected(self) -> None:
        names = tuple(self._ROLES)
        for offset, left in enumerate(names):
            for right in names[offset + 1 :]:
                with self.subTest(left=left, right=right):
                    contaminated = {
                        **self._ROLES,
                        right: (
                            self._ROLES[left][0],
                            *self._ROLES[right][1:],
                        ),
                    }
                    with self.assertRaisesRegex(
                        ValueError,
                        "overlap|disjoint|role",
                    ):
                        validate_d1_role_indices(contaminated)
        with self.assertRaisesRegex(ValueError, "duplicate|role"):
            validate_d1_role_indices(
                {
                    **self._ROLES,
                    "threshold": (
                        self._ROLES["threshold"][0],
                        self._ROLES["threshold"][0],
                        *self._ROLES["threshold"][2:],
                    ),
                }
            )


class ResidualD1CliTests(unittest.TestCase):
    def test_parser_exposes_only_source_side_inputs(self) -> None:
        parser = build_parser()
        destinations = {action.dest for action in parser._actions}

        self.assertTrue(
            {
                "source_manifest",
                "source_root",
                "output_dir",
                "data_root",
                "deadline_seconds",
                "download",
                "dry_run",
                "smoke_test",
            }.issubset(destinations)
        )
        self.assertTrue(
            destinations.isdisjoint(
                {
                    "target_manifest",
                    "target_root",
                    "target_family",
                    "target_victim",
                    "heldout_family",
                    "seed",
                    "source_images",
                    "bc_episodes",
                    "ppo_episodes",
                }
            )
        )

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--source-manifest",
                    "/tmp/source/manifest.json",
                    "--output-dir",
                    "/tmp/residual-d1",
                    "--target-family",
                    "modern_cnn",
                ]
            )

    def test_cli_routes_full_study_to_d1a_subdir_but_keeps_smoke_separate(
        self,
    ) -> None:
        parser = build_parser()
        arguments = parser.parse_args(
            [
                "--source-manifest",
                "/tmp/source/screen_manifest.json",
                "--source-root",
                "/tmp/source",
                "--output-dir",
                "/tmp/d1-study",
                "--data-root",
                "/tmp/data",
            ]
        )

        full = _request(arguments, smoke_test=False)
        smoke = _request(arguments, smoke_test=True)

        self.assertEqual(
            full.output_dir,
            Path("/tmp/d1-study/d1a").resolve(),
        )
        self.assertEqual(
            smoke.output_dir,
            Path("/tmp/d1-study").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
