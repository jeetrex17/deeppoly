import json
import tempfile
import unittest
from pathlib import Path

from rl_transfer.study_report import compact_study_manifest, write_study_report


class StudyReportTests(unittest.TestCase):
    def test_compacts_completed_quick_study_and_writes_visual_report(self) -> None:
        def evaluation(value: float) -> dict[str, object]:
            return {
                "eligible": 100,
                "successes": round(value * 100),
                "asr_at_budgets": {"0": 0.0, "25": value},
                "asr_query_auc": value / 2,
                "normalized_action_entropy": 0.5,
                "query_budget": 25,
                "max_total_target_calls": 25,
                "initialization_included": True,
                "eligible_sample_ids_sha256": "eligible-fixture-digest",
                "policy_digest_before": "fixture-policy-digest",
                "policy_digest_after": "fixture-policy-digest",
                "frozen": True,
            }

        study = {
            "status": "complete",
            "name": "fixture-study",
            "research_valid": False,
            "elapsed_seconds": 12.0,
            "config": {"seeds": [7], "target_families": ["transformer"]},
            "promotion_gate": {"passed": False},
            "aggregate": {
                "transformer": {
                    method: {
                        "final_asr": {"mean": value},
                        "asr_query_auc": {"mean": value / 2},
                        "action_entropy": {"mean": 0.5},
                    }
                    for method, value in (
                        ("groupdro_recurrent_ppo_stochastic", 0.2),
                        ("random_action", 0.1),
                        ("bandit_action", 0.15),
                        ("score_greedy", 0.18),
                    )
                }
            },
            "runs": [
                {
                    "status": "complete",
                    "seed": 7,
                    "target_family": "transformer",
                    "source_families": ["classical_cnn", "modern_cnn"],
                    "elapsed_seconds": 12.0,
                    "target_test_accuracy": 0.5,
                    "victim_accuracy_gate": {"passed": True},
                    "victim_instances": {
                        "transformer": [{"source_validation_accuracy": 0.5}]
                    },
                    "policy": {
                        "training": {
                            "episodes": 10,
                            "trained_episodes": 8,
                            "source_calls": 100,
                            "source_calls_by_family": {},
                            "source_calls_by_victim": {},
                            "final_family_weights": {},
                        }
                    },
                    "evaluation": {
                        "groupdro_recurrent_ppo_stochastic": evaluation(0.2),
                        "random_action": evaluation(0.1),
                        "bandit_action": evaluation(0.15),
                        "score_greedy": evaluation(0.18),
                    },
                }
            ],
        }
        compact = compact_study_manifest(study)
        self.assertEqual(len(compact["runs"]), 1)
        self.assertNotIn("query_traces", json.dumps(compact))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "study.json"
            source.write_text(json.dumps(study))
            outputs = write_study_report(source, Path(directory) / "report")
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            self.assertIn("<svg", outputs["figure"].read_text())
            self.assertIn("Victim gate", outputs["markdown"].read_text())
            self.assertIn("Score greedy", outputs["markdown"].read_text())
            self.assertNotIn("deliberately short victim fitting", outputs["markdown"].read_text())

    def test_report_recomputes_summary_and_sanitizes_output_stem(self) -> None:
        evaluation = {
            "eligible": 10,
            "successes": 1,
            "asr_at_budgets": {"0": 0.0, "5": 0.1},
            "asr_query_auc": 0.05,
            "normalized_action_entropy": 0.5,
            "query_budget": 5,
            "max_total_target_calls": 5,
            "initialization_included": True,
            "eligible_sample_ids_sha256": "eligible-fixture-digest",
            "policy_digest_before": "digest",
            "policy_digest_after": "digest",
            "frozen": True,
        }
        run = {
            "status": "complete",
            "seed": 7,
            "target_family": "transformer",
            "source_families": ["classical_cnn", "modern_cnn"],
            "elapsed_seconds": 1.0,
            "target_test_accuracy": 0.5,
            "victim_accuracy_gate": {
                "passed": True,
                "thresholds": {"transformer": 0.4},
            },
            "victim_instances": {
                "transformer": [{"source_validation_accuracy": 0.5}]
            },
            "policy": {
                "checkpoint_sha256": "checkpoint",
                "training": {
                    "episodes": 1,
                    "trained_episodes": 1,
                    "source_calls": 1,
                    "source_calls_by_family": {},
                    "source_calls_by_victim": {},
                    "final_family_weights": {},
                },
            },
            "evaluation": {
                method: dict(evaluation)
                for method in (
                    "groupdro_recurrent_ppo_stochastic",
                    "random_action",
                    "bandit_action",
                    "score_greedy",
                )
            },
        }
        study = {
            "status": "complete",
            "name": "../../unsafe study",
            "research_valid": False,
            "elapsed_seconds": 1.0,
            "config": {"seeds": [7], "target_families": ["transformer"]},
            "promotion_gate": {"passed": True},
            "aggregate": {"tampered": {}},
            "runs": [run],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "study.json"
            output = Path(directory) / "report"
            source.write_text(json.dumps(study))
            paths = write_study_report(source, output)
            self.assertTrue(all(path.resolve().is_relative_to(output.resolve()) for path in paths.values()))
            compact = json.loads(paths["json"].read_text())
            self.assertNotIn("tampered", compact["aggregate"])

    def test_rejects_an_incomplete_study(self) -> None:
        with self.assertRaises(ValueError):
            compact_study_manifest({"status": "running", "runs": []})


if __name__ == "__main__":
    unittest.main()
