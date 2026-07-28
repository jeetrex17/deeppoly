import tempfile
import unittest
from pathlib import Path

from phase2_temperature_support import (
    fold_summary,
    phase1_fixture,
    result_row,
    stage_a_request,
)
from rl_transfer.phase2_temperature_screen import (
    FOLDS,
    load_phase1_source_selection,
    select_fixed_source_development_indices,
    select_stage_a_temperature,
)
from rl_transfer.verified_artifacts import (
    load_verified_json,
    write_verified_json,
)


class Phase1SelectionTests(unittest.TestCase):
    def test_loader_verifies_manifest_and_selects_only_requested_fold(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = phase1_fixture(root, FOLDS)
            request = stage_a_request(
                manifest_path,
                root / "stage-a",
                folds=(FOLDS[1],),
            )

            selection = load_phase1_source_selection(request)

            self.assertEqual(len(selection.folds), 1)
            self.assertEqual(
                selection.folds[0].heldout_family,
                FOLDS[1],
            )
            self.assertEqual(selection.folds[0].seed, 17)
            self.assertEqual(
                tuple(selection.folds[0].source_victims),
                (FOLDS[0], FOLDS[2]),
            )
            self.assertTrue(selection.folds[0].policy_path.is_file())
            self.assertEqual(selection.target_calls, 0)

    def test_tampered_manifest_and_target_access_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = phase1_fixture(root, FOLDS)
            request = stage_a_request(manifest_path, root / "stage-a")
            manifest_path.write_text(manifest_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_phase1_source_selection(request)

            manifest_path = phase1_fixture(root, FOLDS)
            payload = load_verified_json(manifest_path)
            payload["source_runs"][0]["target_calls"] = 1
            write_verified_json(manifest_path, payload)
            with self.assertRaisesRegex(ValueError, "target"):
                load_phase1_source_selection(request)

    def test_request_rejects_non_preregistered_or_expansive_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            invalid = (
                {"seeds": (29,)},
                {"folds": ("unknown",)},
                {"temperatures": (0.5, 1.0)},
                {"eligible_images_per_family": 65},
                {"deadline_seconds": 601.0},
                {"device": "mps"},
            )
            for overrides in invalid:
                with self.subTest(overrides=overrides), self.assertRaises(
                    ValueError
                ):
                    stage_a_request(
                        manifest,
                        root / "out",
                        **overrides,
                    )


class FixedDevelopmentCohortTests(unittest.TestCase):
    def test_indices_are_first_64_clean_correct_for_every_exact_victim(
        self,
    ) -> None:
        victim_ids = ("source-0", "source-1")
        candidate_indices = tuple(range(80))
        rows = []
        for index in candidate_indices:
            for victim_id in victim_ids:
                rows.append(
                    result_row(
                        victim_id,
                        "modern_cnn",
                        index,
                        clean_correct=not (
                            victim_id == "source-1" and index in {2, 5}
                        ),
                    )
                )

        selected = select_fixed_source_development_indices(
            rows,
            family="modern_cnn",
            exact_source_victim_ids=victim_ids,
            candidate_indices=candidate_indices,
            count=64,
        )

        expected = tuple(
            index for index in candidate_indices if index not in {2, 5}
        )[:64]
        self.assertEqual(selected, expected)

    def test_inconsistent_methods_or_insufficient_cohort_fail_closed(
        self,
    ) -> None:
        victim_ids = ("source-0", "source-1")
        rows = [
            result_row(
                victim_id,
                "modern_cnn",
                index,
                clean_correct=True,
            )
            for index in range(10)
            for victim_id in victim_ids
        ]
        rows.append(
            result_row(
                "source-0",
                "modern_cnn",
                0,
                clean_correct=False,
                method="random_action",
            )
        )
        with self.assertRaises(ValueError):
            select_fixed_source_development_indices(
                rows,
                family="modern_cnn",
                exact_source_victim_ids=victim_ids,
                candidate_indices=tuple(range(10)),
                count=10,
            )


class TemperatureSelectionTests(unittest.TestCase):
    def _complete(
        self,
        *,
        candidate_asr: float,
        candidate_auc: float,
    ) -> list[dict[str, object]]:
        values = {
            0.25: (0.09, 0.035, 0.5),
            0.5: (candidate_asr, candidate_auc, 0.5),
            0.75: (0.10, 0.040, 0.5),
            1.0: (0.10, 0.040, 0.5),
            1.5: (0.095, 0.039, 0.5),
        }
        return [fold_summary(fold, values) for fold in FOLDS]

    def test_nondefault_replaces_default_only_after_locked_gain_rule(
        self,
    ) -> None:
        decision = select_stage_a_temperature(
            self._complete(candidate_asr=0.106, candidate_auc=0.039)
        )
        self.assertEqual(decision["selected_temperature"], 0.5)
        self.assertEqual(
            decision["diagnostic_temperature_for_phase1_checkpoint"],
            0.5,
        )
        self.assertEqual(
            decision["decision_scope"],
            "phase1_checkpoint_diagnostic_only",
        )
        self.assertTrue(decision["diagnostic_only"])
        self.assertFalse(decision["applies_to_new_phase2_architecture"])
        self.assertFalse(
            decision["authorizes_phase2_deployment_temperature"]
        )
        self.assertEqual(decision["ranking_tie_band"], 0.002)
        self.assertEqual(
            decision["ranking_tie_reference"],
            "best_macro_asr_gain_vs_score",
        )
        self.assertEqual(
            decision["ranking_rule"],
            (
                "Rank by macro ASR gain versus matched score greedy. Treat "
                "candidates within 0.002 of the best macro ASR gain as tied, "
                "then rank by higher macro AUC gain versus score greedy, "
                "then lower temperature."
            ),
        )
        self.assertTrue(decision["replaced_default"])
        self.assertTrue(decision["complete"])

        fallback = select_stage_a_temperature(
            self._complete(candidate_asr=0.104, candidate_auc=0.041)
        )
        self.assertEqual(fallback["selected_temperature"], 1.0)
        self.assertFalse(fallback["replaced_default"])

    def test_ties_use_auc_then_lower_temperature_and_bad_entropy_fails(
        self,
    ) -> None:
        values = {
            0.25: (0.1060, 0.041, 0.5),
            0.5: (0.1070, 0.040, 0.5),
            0.75: (0.09, 0.03, 0.5),
            1.0: (0.10, 0.04, 0.5),
            1.5: (0.09, 0.03, 0.5),
        }
        summaries = [fold_summary(fold, values) for fold in FOLDS]
        decision = select_stage_a_temperature(summaries)
        self.assertEqual(decision["ranked_winner"], 0.25)

        equal_auc_values = {
            **values,
            0.25: (0.1060, 0.040, 0.5),
            0.5: (0.1070, 0.040, 0.5),
        }
        equal_auc_summaries = [
            fold_summary(fold, equal_auc_values) for fold in FOLDS
        ]
        lower_temperature = select_stage_a_temperature(
            equal_auc_summaries
        )
        self.assertEqual(lower_temperature["ranked_winner"], 0.25)

        summaries[0]["temperatures"]["0.25"][
            "normalized_action_entropy"
        ] = 0.99
        fallback = select_stage_a_temperature(summaries)
        self.assertEqual(fallback["selected_temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()
