import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from phase2_temperature_support import (
    fold_summary,
    phase1_fixture,
    phase1_selection,
    result_row,
    stage_a_request,
)
from rl_transfer.artifacts import sha256_file
from rl_transfer.phase2_temperature_evaluation import (
    _StageDeadlineReached,
    evaluate_temperature_fold,
)
from rl_transfer.phase2_temperature_screen import (
    FOLDS,
    STAGE_A_TEMPERATURES,
    Phase1SourceFold,
    build_stage_a_dry_run,
    load_phase1_source_selection,
    run_temperature_screen_from_datasets,
    write_verified_jsonl,
)
from rl_transfer.research_protocol import FrozenEpisodeResult
from rl_transfer.verified_artifacts import load_verified_json


class TemperatureScreenRunnerTests(unittest.TestCase):
    def test_runner_stops_scheduling_at_deadline_and_writes_verified_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = stage_a_request(
                root / "manifest.json",
                root / "out",
            )
            selection = phase1_selection(root)
            values = {
                temperature: (0.1, 0.04, 0.5)
                for temperature in STAGE_A_TEMPERATURES
            }
            clocks = iter((0.0, 0.0, 100.0, 700.0, 700.0))
            evaluated: list[str] = []

            def evaluator(*, fold, **_kwargs):
                evaluated.append(fold.heldout_family)
                return fold_summary(fold.heldout_family, values), []

            with mock.patch(
                "rl_transfer.phase2_temperature_screen."
                "load_phase1_source_selection",
                return_value=selection,
            ):
                result = run_temperature_screen_from_datasets(
                    request,
                    object(),
                    object(),
                    fold_evaluator=evaluator,
                    clock=lambda: next(clocks),
                )

            self.assertEqual(evaluated, list(FOLDS[:2]))
            self.assertEqual(result["status"], "deadline_reached")
            self.assertEqual(result["target_calls"], 0)
            self.assertFalse(result["target_evaluation_performed"])
            summary = load_verified_json(root / "out" / "stage_a.json")
            self.assertEqual(summary["status"], "deadline_reached")
            self.assertEqual(
                summary["name"],
                "phase1-checkpoint-diagnostic-temperature-screen",
            )
            self.assertEqual(
                summary["decision_scope"],
                "phase1_checkpoint_diagnostic_only",
            )
            self.assertTrue(summary["diagnostic_only"])
            self.assertFalse(
                summary["applies_to_new_phase2_architecture"]
            )
            self.assertFalse(
                summary["authorizes_phase2_deployment_temperature"]
            )
            self.assertIn(
                "phase1_diagnostic_temperature_decision",
                summary,
            )
            self.assertNotIn("temperature_selection", summary)
            portable_paths = (
                summary["phase1_manifest"],
                summary["results_path"],
                summary["request"]["phase1_manifest"],
                summary["request"]["phase1_root"],
                summary["request"]["output_dir"],
                summary["request"]["data_root"],
            )
            for portable_path in portable_paths:
                with self.subTest(portable_path=portable_path):
                    self.assertFalse(Path(portable_path).is_absolute())
                    self.assertNotIn("..", Path(portable_path).parts)
            persisted = json.dumps(summary, sort_keys=True)
            self.assertNotIn("/home/", persisted)
            self.assertNotIn("/Users/", persisted)
            rows_path = root / "out" / "stage_a_results.jsonl"
            self.assertTrue(rows_path.is_file())
            self.assertTrue(
                rows_path.with_suffix(".jsonl.sha256").is_file()
            )

    def test_evaluator_checks_deadline_inside_source_sample_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(
                Path(
                    "configs/rl_transfer/cifar10_rtx_bc_ppo.json"
                ).read_text()
            )
            config.update(
                {
                    "seed": 17,
                    "target_family": FOLDS[0],
                }
            )
            policy_digest = "d" * 64
            fold = Phase1SourceFold(
                seed=17,
                heldout_family=FOLDS[0],
                source_families=(FOLDS[1], FOLDS[2]),
                fingerprint="a" * 64,
                run_dir=root / "run",
                policy_path=root / "policy.pt",
                source_results_path=root / "rows.jsonl",
                run_manifest={
                    "config": config,
                    "policy": {
                        "checkpoint_sha256": "e" * 64,
                        "persistent_digest": policy_digest,
                    },
                },
                source_victims={
                    FOLDS[1]: (),
                    FOLDS[2]: (),
                },
            )
            request = stage_a_request(
                root / "manifest.json",
                root / "out",
            )
            runtime_policy = mock.Mock()
            runtime_policy.persistent_digest.return_value = "runtime"
            victims = {
                family: (("victim-0", object()), ("victim-1", object()))
                for family in fold.source_families
            }
            indices = {
                family: (0, 1) for family in fold.source_families
            }
            dataset = (
                (torch.zeros((3, 32, 32)), 0),
                (torch.ones((3, 32, 32)), 0),
            )
            clocks = iter((0.0, 0.0, 601.0))

            def score_result(*args, **_kwargs):
                return FrozenEpisodeResult(
                    sample_id=args[3],
                    victim_id=args[4],
                    family=args[5],
                    clean_correct=True,
                    success=False,
                    query_to_success=None,
                    total_target_calls=50,
                    linf=0.0,
                    l2=0.0,
                    actions=tuple(0 for _ in range(49)),
                    policy_digest_before="score",
                    policy_digest_after="score",
                    query_trace=(),
                )

            with (
                mock.patch(
                    "rl_transfer.phase2_temperature_evaluation."
                    "_load_policy",
                    return_value=runtime_policy,
                ),
                mock.patch(
                    "rl_transfer.phase2_temperature_evaluation."
                    "_phase1_policy_digest",
                    return_value=policy_digest,
                ),
                mock.patch(
                    "rl_transfer.phase2_temperature_evaluation."
                    "_load_exact_source_victims",
                    return_value=victims,
                ),
                mock.patch(
                    "rl_transfer.phase2_temperature_evaluation."
                    "_fixed_indices_by_family",
                    return_value=indices,
                ),
                mock.patch(
                    "rl_transfer.phase2_temperature_evaluation."
                    "run_score_greedy_episode",
                    side_effect=score_result,
                ) as score,
            ):
                with self.assertRaises(_StageDeadlineReached) as raised:
                    evaluate_temperature_fold(
                        fold=fold,
                        request=request,
                        train_dataset=dataset,
                        test_dataset=dataset,
                        absolute_deadline=600.0,
                        clock=lambda: next(clocks),
                        progress=lambda _message: None,
                    )

            self.assertEqual(score.call_count, 1)
            self.assertEqual(len(raised.exception.rows), 1)
            partial = raised.exception.partial_fold
            self.assertIsNotNone(partial)
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["partial_row_count"], 1)
            self.assertEqual(partial["source_model_calls"], 50)
            self.assertEqual(partial["target_calls"], 0)
            self.assertEqual(
                partial["interrupted_method"],
                "score_greedy",
            )

    def test_runner_preserves_verified_rows_from_mid_block_deadline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = stage_a_request(
                root / "manifest.json",
                root / "out",
            )
            selection = phase1_selection(root)
            partial_row = result_row(
                "source-0",
                FOLDS[1],
                0,
                clean_correct=True,
            )
            partial_fold = {
                "seed": 17,
                "heldout_family": FOLDS[0],
                "complete": False,
                "source_model_calls": 50,
                "target_calls": 0,
                "target_evaluation_performed": False,
            }

            def interrupted(**_kwargs):
                raise _StageDeadlineReached(
                    "deadline",
                    rows=(partial_row,),
                    partial_fold=partial_fold,
                )

            clocks = iter((0.0, 0.0, 1.0))
            with mock.patch(
                "rl_transfer.phase2_temperature_screen."
                "load_phase1_source_selection",
                return_value=selection,
            ):
                result = run_temperature_screen_from_datasets(
                    request,
                    object(),
                    object(),
                    fold_evaluator=interrupted,
                    clock=lambda: next(clocks),
                )

            self.assertEqual(result["status"], "deadline_reached")
            self.assertEqual(result["completed_folds"], 0)
            self.assertEqual(result["partial_folds"], 1)
            self.assertTrue(result["partial_results_preserved"])
            self.assertEqual(result["source_model_calls"], 50)
            self.assertEqual(result["target_calls"], 0)
            rows_path = root / "out" / "stage_a_results.jsonl"
            self.assertEqual(len(rows_path.read_text().splitlines()), 1)
            self.assertEqual(
                sha256_file(rows_path),
                rows_path.with_suffix(
                    ".jsonl.sha256"
                ).read_text().strip(),
            )

    def test_dry_run_and_cli_never_load_datasets_or_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = phase1_fixture(root, (FOLDS[0],))
            request = stage_a_request(
                manifest,
                root / "out",
                folds=(FOLDS[0],),
            )
            selection = load_phase1_source_selection(request)
            plan = build_stage_a_dry_run(request, selection)
            self.assertEqual(
                plan["mode"],
                "source_only_phase1_checkpoint_temperature_diagnostic",
            )
            self.assertEqual(
                plan["decision_scope"],
                "phase1_checkpoint_diagnostic_only",
            )
            self.assertFalse(
                plan["applies_to_new_phase2_architecture"]
            )
            self.assertFalse(
                plan["authorizes_phase2_deployment_temperature"]
            )
            self.assertFalse(Path(plan["phase1_manifest"]).is_absolute())
            self.assertEqual(plan["ranking_tie_band"], 0.002)
            self.assertEqual(
                plan["ranking_tie_reference"],
                "best_macro_asr_gain_vs_score",
            )
            self.assertEqual(plan["target_calls"], 0)
            self.assertFalse(plan["target_evaluation_available"])

            from rl_transfer import phase2_temperature_cli

            stream = io.StringIO()
            with (
                mock.patch(
                    "rl_transfer.phase2_temperature_cli."
                    "load_phase1_source_selection",
                    return_value=selection,
                ),
                mock.patch(
                    "rl_transfer.phase2_temperature_cli."
                    "run_temperature_screen_from_datasets"
                ) as run,
                redirect_stdout(stream),
            ):
                exit_code = phase2_temperature_cli.main(
                    [
                        "--phase1-manifest",
                        str(manifest),
                        "--phase1-root",
                        str(root),
                        "--output-dir",
                        str(root / "out"),
                        "--fold",
                        FOLDS[0],
                        "--dry-run",
                    ]
                )
            self.assertEqual(exit_code, 0)
            run.assert_not_called()
            self.assertIn(
                "source_only_phase1_checkpoint_temperature_diagnostic",
                stream.getvalue(),
            )

    def test_verified_jsonl_is_atomic_and_checksum_protected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            digest = write_verified_jsonl(
                path,
                ({"row": 1}, {"row": 2}),
            )
            self.assertEqual(digest, sha256_file(path))
            self.assertEqual(
                path.with_suffix(".jsonl.sha256").read_text().strip(),
                digest,
            )
            self.assertFalse(tuple(path.parent.glob("*.tmp")))

            path.write_text(path.read_text() + "{}\n")
            self.assertNotEqual(
                sha256_file(path),
                path.with_suffix(".jsonl.sha256").read_text().strip(),
            )

    def test_stage_a_module_has_no_target_evaluation_surface(self) -> None:
        source = Path(
            "rl_transfer/phase2_temperature_screen.py"
        ).read_text()
        self.assertNotIn("target_evidence", source)
        self.assertNotIn("seen_family_new_instance", source)


if __name__ == "__main__":
    unittest.main()
