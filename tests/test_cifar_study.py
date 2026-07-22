import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from rl_transfer.cifar_study import CIFARStudyConfig, run_study_from_datasets, summarize_study


class BalancedTensorDataset(Dataset):
    def __init__(self, per_class: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.targets = [label for label in range(10) for _ in range(per_class)]
        self.images = torch.rand((len(self.targets), 3, 32, 32), generator=generator)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


def _run(
    seed: int,
    learned: float,
    random: float,
    bandit: float,
    score_greedy: float,
    family: str = "transformer",
) -> dict[str, object]:
    def metrics(value: float, entropy: float) -> dict[str, object]:
        return {
            "eligible": 1000,
            "successes": round(value * 1000),
            "asr_at_budgets": {"0": 0.0, "25": value},
            "asr_query_auc": value / 2,
            "normalized_action_entropy": entropy,
            "query_budget": 25,
            "max_total_target_calls": 25,
            "initialization_included": True,
            "eligible_sample_ids_sha256": "eligible-fixture-digest",
            "policy_digest_before": f"{family}-{seed}-{entropy}",
            "policy_digest_after": f"{family}-{seed}-{entropy}",
            "frozen": True,
        }

    return {
        "status": "complete",
        "seed": seed,
        "target_family": family,
        "victim_accuracy_gate": {"passed": True},
        "evaluation": {
            "groupdro_recurrent_ppo_stochastic": metrics(learned, 0.6),
            "random_action": metrics(random, 0.99),
            "bandit_action": metrics(bandit, 0.7),
            "score_greedy": metrics(score_greedy, 0.8),
        },
    }


class CIFARStudyTests(unittest.TestCase):
    def test_three_seed_promotion_requires_positive_paired_ci_against_every_control(self) -> None:
        passing = summarize_study(
            tuple(_run(seed, 0.40, 0.10, 0.15, 0.20) for seed in (1, 2, 3))
        )
        failing = summarize_study(
            (
                _run(1, 0.60, 0.10, 0.10, 0.10),
                _run(2, 0.60, 0.10, 0.10, 0.10),
                _run(3, 0.01, 0.10, 0.10, 0.10),
            )
        )
        self.assertTrue(passing["promotion_gate"]["passed"])
        self.assertFalse(failing["promotion_gate"]["passed"])
        self.assertIsNotNone(
            passing["aggregate"]["transformer"]["random_action"]["final_asr"]["ci95"]
        )
        self.assertGreater(
            passing["promotion_gate"]["fold_details"]["transformer"]["comparisons"]
            ["score_greedy"]["final_asr_delta"]["ci95"][0],
            0,
        )

    def test_single_seed_is_diagnostic_and_never_promoted(self) -> None:
        summary = summarize_study((_run(7, 0.5, 0.1, 0.2, 0.3),))
        self.assertFalse(summary["promotion_gate"]["passed"])
        self.assertIsNone(
            summary["aggregate"]["transformer"]["random_action"]["final_asr"]["ci95"]
        )

    def test_student_t_interval_is_conservative_for_three_seeds(self) -> None:
        summary = summarize_study(
            (
                _run(1, 0.4, 0.0, 0.1, 0.2),
                _run(2, 0.5, 0.1, 0.2, 0.3),
                _run(3, 0.6, 0.2, 0.3, 0.4),
            )
        )
        interval = summary["aggregate"]["transformer"]["random_action"]["final_asr"]["ci95"]
        self.assertEqual(interval[0], 0.0)
        self.assertGreater(interval[1], 0.3)

    def test_expected_grid_is_fail_closed_for_missing_or_duplicate_runs(self) -> None:
        runs = tuple(_run(seed, 0.4, 0.1, 0.1, 0.1) for seed in (1, 2, 3))
        partial = summarize_study(
            runs,
            expected_families=("classical_cnn", "transformer"),
            expected_seeds=(1, 2, 3),
        )
        self.assertFalse(partial["promotion_gate"]["passed"])
        self.assertFalse(partial["promotion_gate"]["grid_complete"])
        self.assertEqual(len(partial["promotion_gate"]["missing_runs"]), 3)
        with self.assertRaises(ValueError):
            summarize_study((*runs, runs[0]))

    def test_rejects_missing_methods_and_non_finite_metrics(self) -> None:
        missing = _run(1, 0.4, 0.1, 0.1, 0.1)
        del missing["evaluation"]["score_greedy"]
        with self.assertRaises(ValueError):
            summarize_study((missing,))
        invalid = _run(1, 0.4, 0.1, 0.1, 0.1)
        invalid["evaluation"]["random_action"]["asr_query_auc"] = None
        with self.assertRaises(ValueError):
            summarize_study((invalid,))

    def test_rejects_non_frozen_or_non_query_matched_summaries(self) -> None:
        non_frozen = _run(1, 0.4, 0.1, 0.1, 0.1)
        non_frozen["evaluation"]["random_action"]["frozen"] = False
        with self.assertRaises(ValueError):
            summarize_study((non_frozen,))

        mismatched_budget = _run(1, 0.4, 0.1, 0.1, 0.1)
        mismatched_budget["evaluation"]["random_action"]["query_budget"] = 10
        with self.assertRaises(ValueError):
            summarize_study((mismatched_budget,))

        inconsistent_asr = _run(1, 0.4, 0.1, 0.1, 0.1)
        inconsistent_asr["evaluation"]["random_action"]["successes"] = 9
        with self.assertRaises(ValueError):
            summarize_study((inconsistent_asr,))

    def test_minimum_practical_gain_and_every_seed_entropy_are_required(self) -> None:
        negligible = summarize_study(
            tuple(_run(seed, 0.101, 0.1, 0.1, 0.1) for seed in (1, 2, 3)),
            minimum_asr_gain=0.01,
            minimum_auc_gain=0.005,
        )
        self.assertFalse(negligible["promotion_gate"]["passed"])
        collapsed = tuple(_run(seed, 0.4, 0.1, 0.1, 0.1) for seed in (1, 2, 3))
        collapsed[0]["evaluation"]["groupdro_recurrent_ppo_stochastic"][
            "normalized_action_entropy"
        ] = 0.0
        collapsed_summary = summarize_study(collapsed)
        self.assertFalse(collapsed_summary["promotion_gate"]["passed"])

    def test_config_rejects_duplicate_seeds_and_unknown_families(self) -> None:
        base = {
            "schema_version": 1,
            "name": "fixture",
            "research_valid": False,
            "base_config": "base.json",
            "output_dir": "output",
            "seeds": [1, 1],
            "target_families": ["transformer"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(base))
            with self.assertRaises(ValueError):
                CIFARStudyConfig.from_json(path)
            path.write_text(json.dumps({**base, "seeds": [1], "target_families": ["unknown"]}))
            with self.assertRaises(ValueError):
                CIFARStudyConfig.from_json(path)
            path.write_text(
                json.dumps(
                    {
                        **base,
                        "name": "../../escape",
                        "seeds": [1],
                        "target_families": ["transformer"],
                    }
                )
            )
            with self.assertRaises(ValueError):
                CIFARStudyConfig.from_json(path)

    def test_fixture_study_executes_a_reverse_fold_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path = root / "base.json"
            base_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": "fixture-base",
                        "research_valid": False,
                        "dataset": "CIFAR-10",
                        "device": "cpu",
                        "download": False,
                        "data_root": str(root / "data"),
                        "output_dir": str(root / "unused"),
                        "seed": 7,
                        "victim_train_images": 50,
                        "policy_train_images": 50,
                        "source_validation_images": 50,
                        "outer_test_images": 20,
                        "victim_epochs": 1,
                        "policy_episodes": 2,
                        "policy_update_block": 1,
                        "policy_learning_rate": 0.001,
                        "policy_entropy_weight": 0.01,
                        "policy_update_epochs": 2,
                        "query_budget": 2,
                        "grid_size": 1,
                        "epsilon": 8 / 255,
                        "step_size": 2 / 255,
                        "batch_size": 25,
                        "num_workers": 0,
                        "hidden_dim": 8,
                        "victim_learning_rate": 0.001,
                        "target_family": "transformer",
                        "source_instances_per_family": 1,
                        "victim_profile": "pilot",
                        "reward_mode": "margin_delta",
                    }
                )
            )
            config = CIFARStudyConfig(
                schema_version=1,
                name="fixture-study",
                research_valid=False,
                base_config=str(base_path),
                output_dir=str(root / "study-output"),
                seeds=(7,),
                target_families=("classical_cnn", "modern_cnn"),
            )
            result = run_study_from_datasets(
                config,
                BalancedTensorDataset(per_class=20, seed=1),
                BalancedTensorDataset(per_class=5, seed=2),
                dataset_version="fixture",
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["runs"][0]["target_family"], "classical_cnn")
            self.assertEqual(
                result["runs"][0]["source_families"],
                ["modern_cnn", "transformer"],
            )
            self.assertIn("bandit_action", result["runs"][0]["evaluation"])
            self.assertIn("score_greedy", result["runs"][0]["evaluation"])
            self.assertTrue(
                all(
                    metrics["resumed"]
                    for family_metrics in result["runs"][1]["victim_instances"].values()
                    for metrics in family_metrics
                )
            )
            self.assertTrue(
                (root / "study-output" / "fixture-study" / "study_manifest.json").is_file()
            )
            self.assertEqual(result["promotion_gate"]["minimum_asr_gain"], 0.01)


if __name__ == "__main__":
    unittest.main()
