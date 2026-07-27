import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from rl_transfer.cifar_evaluation import evaluate_methods
from rl_transfer.config import AttackConfig
from rl_transfer.evaluation_audit import audit_evaluation
from rl_transfer.gpu_config import RTXPublicationConfig
from rl_transfer.gpu_reporting import (
    create_timestamped_export_directory,
    load_verified_runtime_freeze,
    load_verified_study_manifest,
    resolve_verified_result_rows,
)
from rl_transfer.gpu_study import confirmatory_transfer_gate
from rl_transfer.recurrent import RecurrentAttackPolicy
from rl_transfer.statistics import (
    exact_paired_sign_flip_pvalue,
    hierarchical_paired_bootstrap_interval,
)
from rl_transfer.verified_artifacts import write_verified_json


def _method_metrics(
    asr: float,
    auc: float,
    *,
    digest: str = "operator-digest",
    eligible_digest: str = "eligible-digest",
    entropy: float = 0.6,
) -> dict[str, object]:
    victim_metric = {
        "eligible": 50,
        "successes": round(asr * 50),
        "asr_at_budgets": {"0": 0.0, "50": asr},
        "asr_query_auc": auc,
        "eligible_sample_ids_sha256": eligible_digest,
    }
    return {
        "eligible": 100,
        "successes": round(asr * 100),
        "asr_at_budgets": {"0": 0.0, "50": asr},
        "asr_query_auc": auc,
        "normalized_action_entropy": entropy,
        "query_budget": 50,
        "max_total_target_calls": 50,
        "initialization_included": True,
        "eligible_sample_ids_sha256": eligible_digest,
        "policy_digest_before": "policy-digest",
        "policy_digest_after": "policy-digest",
        "operator_digest": digest,
        "frozen": True,
        "by_victim": {
            "victim-0": victim_metric,
            "victim-1": dict(victim_metric),
            "victim-2": dict(victim_metric),
        },
    }


class EvaluationIntegrityTests(unittest.TestCase):
    def _evaluated_fixture(
        self,
    ) -> tuple[
        list[object],
        dict[str, object],
        AttackConfig,
    ]:
        class MeanVictim(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(
                    torch.zeros(()),
                    requires_grad=False,
                )

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                means = images.mean(dim=(1, 2, 3)) + 0 * self.anchor
                return torch.stack((means, 1 - means), dim=1)

        attack = AttackConfig(
            epsilon=0.1,
            step_size=0.05,
            grid_size=1,
            max_queries=3,
            rollback_on_non_improvement=True,
        )
        policy = RecurrentAttackPolicy(
            attack.recurrent_observation_dim,
            attack.action_dim,
            hidden_dim=8,
            seed=7,
        )
        rows, _, evaluation = evaluate_methods(
            policy,
            (("victim-0", MeanVictim()),),
            ((torch.full((3, 4, 4), 0.7), 0),),
            (123,),
            attack,
            11,
            "classical_cnn",
            lambda _message: None,
        )
        return rows, evaluation, attack

    def test_raw_audit_requires_the_exact_preregistered_cohort(self) -> None:
        rows, evaluation, attack = self._evaluated_fixture()
        audit = audit_evaluation(
            rows,
            evaluation,
            attack,
            expected_sample_ids={
                "cifar10:classical_cnn:victim-0:123",
                "cifar10:classical_cnn:victim-1:123",
            },
            expected_victim_ids=("victim-0", "victim-1"),
        )
        self.assertFalse(audit["passed"])
        self.assertTrue(any("expected cohort" in error for error in audit["errors"]))

    def test_raw_audit_rejects_invalid_action_count_and_range(self) -> None:
        rows, evaluation, attack = self._evaluated_fixture()
        corrupted = [
            replace(rows[0], action_trace=(attack.action_dim,)),
            *rows[1:],
        ]
        audit = audit_evaluation(
            corrupted,
            evaluation,
            attack,
            expected_sample_ids={"cifar10:classical_cnn:victim-0:123"},
            expected_victim_ids=("victim-0",),
        )
        self.assertFalse(audit["passed"])
        self.assertTrue(any("action trace" in error for error in audit["errors"]))


class HierarchicalStatisticsTests(unittest.TestCase):
    def test_hierarchical_bootstrap_is_paired_deterministic_and_positive(self) -> None:
        cells = {
            17: {
                "victim-a": (0.1, 0.2, 0.3),
                "victim-b": (0.2, 0.3, 0.4),
            },
            29: {
                "victim-c": (0.1, 0.1, 0.2),
                "victim-d": (0.2, 0.2, 0.3),
            },
        }
        first = hierarchical_paired_bootstrap_interval(
            cells,
            samples=500,
            seed=7,
        )
        second = hierarchical_paired_bootstrap_interval(
            cells,
            samples=500,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertGreater(first[0], 0)
        self.assertGreaterEqual(first[1], first[0])

    def test_hierarchical_bootstrap_rejects_empty_cells(self) -> None:
        with self.assertRaises(ValueError):
            hierarchical_paired_bootstrap_interval({}, samples=100, seed=1)
        with self.assertRaises(ValueError):
            hierarchical_paired_bootstrap_interval(
                {1: {"victim": ()}},
                samples=100,
                seed=1,
            )

    def test_exact_sign_flip_uses_policy_replicates_not_victim_cells(
        self,
    ) -> None:
        differences = (0.04,) * 10
        pvalue = exact_paired_sign_flip_pvalue(differences)
        self.assertAlmostEqual(pvalue, 2 / (2**10))
        with self.assertRaises(ValueError):
            exact_paired_sign_flip_pvalue(())

    def test_authoritative_gate_aggregates_families_within_policy_seed(
        self,
    ) -> None:
        config = RTXPublicationConfig.from_json(
            Path("configs/rl_transfer/cifar10_rtx_publication.json")
        )
        runs = []
        for family in config.target_families:
            for seed in config.seeds:
                learned = _method_metrics(0.24, 0.12)
                learned["operator_digest"] = "shared"
                learned["policy_digest_before"] = f"policy:{family}:{seed}"
                learned["policy_digest_after"] = f"policy:{family}:{seed}"
                control = _method_metrics(0.16, 0.08)
                control["operator_digest"] = "shared"
                bc_only = _method_metrics(0.18, 0.09)
                bc_only["operator_digest"] = "shared"
                ppo_only = _method_metrics(0.17, 0.085)
                ppo_only["operator_digest"] = "shared"
                runs.append(
                    {
                        "status": "complete",
                        "target_family": family,
                        "seed": seed,
                        "victim_seed": config.victim_seed,
                        "victim_bank_digest": "a" * 64,
                        "victim_accuracy_gate": {"passed": True},
                        "evaluation_audit": {
                            "passed": True,
                            "expected_cohort_verified": True,
                        },
                        "policy": {
                            "checkpoint_sha256": hashlib.sha256(
                                f"{family}:{seed}".encode()
                            ).hexdigest(),
                        },
                        "evaluation": {
                            "gradient_bc_groupdro_ppo_stochastic": learned,
                            "score_greedy": control,
                            "gradient_bc_only_stochastic": bc_only,
                            "ppo_only_stochastic": ppo_only,
                        },
                    }
                )
        expected_checkpoints = {
            (f"{run['target_family']}/seed-{run['seed']}"): run["policy"][
                "checkpoint_sha256"
            ]
            for run in runs
        }
        gate = confirmatory_transfer_gate(
            runs,
            config,
            expected_victim_bank_digest="a" * 64,
            expected_policy_checkpoints=expected_checkpoints,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(len(gate["policy_seed_differences"]), len(config.seeds))
        self.assertLessEqual(gate["primary"]["exact_sign_flip_pvalue"], 0.05)
        failed = copy.deepcopy(runs)
        failed[0]["evaluation_audit"]["passed"] = False
        self.assertFalse(confirmatory_transfer_gate(failed, config)["passed"])
        incomplete = copy.deepcopy(runs)
        for metrics in incomplete[0]["evaluation"].values():
            del metrics["by_victim"]["victim-2"]
        self.assertFalse(confirmatory_transfer_gate(incomplete, config)["passed"])
        changed_bank = copy.deepcopy(runs)
        changed_bank[0]["victim_bank_digest"] = "b" * 64
        self.assertFalse(
            confirmatory_transfer_gate(
                changed_bank,
                config,
                expected_victim_bank_digest="a" * 64,
                expected_policy_checkpoints=expected_checkpoints,
            )["passed"]
        )


class RTXNotebookTests(unittest.TestCase):
    def test_reporting_rejects_tampered_manifest_and_result_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            study_dir = Path(directory)
            manifest_path = study_dir / "study_manifest.json"
            write_verified_json(
                manifest_path,
                {"status": "complete"},
            )
            self.assertEqual(
                load_verified_study_manifest(
                    manifest_path,
                    study_dir,
                )["status"],
                "complete",
            )
            manifest_path.write_text('{"status": "modified"}')
            with self.assertRaises(ValueError):
                load_verified_study_manifest(
                    manifest_path,
                    study_dir,
                )

            run_dir = study_dir / "runs" / "seed-17"
            run_dir.mkdir(parents=True)
            result_path = run_dir / "results.jsonl"
            result_path.write_text('{"success": true}\n')
            write_verified_json(
                run_dir / "target_evaluation.json",
                {
                    "results_sha256": hashlib.sha256(
                        result_path.read_bytes()
                    ).hexdigest()
                },
            )
            self.assertEqual(
                resolve_verified_result_rows(
                    run_dir,
                    study_dir / "runs",
                    study_dir,
                ),
                result_path.resolve(),
            )
            result_path.write_text('{"success": false}\n')
            with self.assertRaises(ValueError):
                resolve_verified_result_rows(
                    run_dir,
                    study_dir / "runs",
                    study_dir,
                )

    def test_reporting_rejects_export_symlink_and_freeze_mismatch(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as repository_directory,
            tempfile.TemporaryDirectory() as outside_directory,
        ):
            repository = Path(repository_directory)
            study_dir = repository / "study"
            study_dir.mkdir()
            report_root = study_dir / "paper_artifacts"
            report_root.symlink_to(
                Path(outside_directory),
                target_is_directory=True,
            )
            with self.assertRaises(ValueError):
                create_timestamped_export_directory(
                    study_dir,
                    report_root,
                    "20260727_120000",
                )
            runs_root = study_dir / "runs"
            runs_root.symlink_to(
                Path(outside_directory),
                target_is_directory=True,
            )
            with self.assertRaises(ValueError):
                resolve_verified_result_rows(
                    runs_root / "seed-17",
                    runs_root,
                    study_dir,
                )

            freeze_path = study_dir / "pip_freeze.txt"
            freeze_path.write_text("torch==2.7.1\n")
            study = {
                "runtime_environment": {
                    "pip_freeze_path": str(freeze_path.relative_to(repository)),
                    "pip_freeze_sha256": hashlib.sha256(
                        freeze_path.read_bytes()
                    ).hexdigest(),
                }
            }
            _, freeze = load_verified_runtime_freeze(
                study,
                study_dir,
                repository,
            )
            self.assertEqual(freeze, "torch==2.7.1\n")
            freeze_path.write_text("torch==0.0.0\n")
            with self.assertRaises(ValueError):
                load_verified_runtime_freeze(
                    study,
                    study_dir,
                    repository,
                )

    def test_notebook_is_thin_safe_and_disabled_by_default(self) -> None:
        path = Path("notebooks/cifar10_rtx_publication_study.ipynb")
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        sources = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        code_sources = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        for source in code_sources:
            compile(source, str(path), "exec")

        self.assertIn("RUN_FULL_STUDY = False", sources)
        self.assertIn("EXPORT_ARTIFACTS = False", sources)
        self.assertIn("RL_RUN_SOURCE_PHASE", sources)
        self.assertIn("PROTOCOL_TREE_CLEAN", sources)
        self.assertIn("DEVICE = 'cuda'", sources)
        self.assertIn("cifar10_rtx_publication.json", sources)
        self.assertIn("rl_transfer.gpu_study_cli", sources)
        self.assertIn("source_competence_gate", sources)
        self.assertIn("publication_candidate", sources)
        self.assertIn("confirmatory_gate", sources)
        self.assertIn(
            "gradient_bc_groupdro_ppo_stochastic",
            sources,
        )
        self.assertIn("Ten policy seeds", sources)
        self.assertIn("fixed victim bank", sources)
        self.assertIn("fixed custom CIFAR-10 victim bank", sources)
        self.assertIn(
            "result_path = resolve_verified_result_rows(",
            sources,
        )
        self.assertGreaterEqual(
            sources.count("load_verified_study_manifest(MANIFEST_PATH"),
            2,
        )
        self.assertNotIn(
            "json.loads(MANIFEST_PATH.read_text())",
            sources,
        )
        self.assertIn(
            "target_cache = load_verified_json(target_cache_path)",
            Path("rl_transfer/gpu_reporting.py").read_text(),
        )
        self.assertIn(
            "sha256_file(result_path) != expected_results_sha",
            Path("rl_transfer/gpu_reporting.py").read_text(),
        )
        self.assertIn(
            "REPORT_ROOT = resolve_descendant(",
            sources,
        )
        self.assertIn(
            "load_verified_runtime_freeze",
            sources,
        )
        self.assertIn("Jeetraj", sources)
        self.assertIn("Prashit", sources)
        for forbidden in (
            "/Users/",
            "/content/",
            "/kaggle/",
            "!pip",
            "%pip",
            "shell=True",
            "class RecurrentAttackPolicy",
            "optimizer.step",
            "loss.backward",
        ):
            self.assertNotIn(forbidden, sources)
        for superseded in (
            "final_publication_candidate",
            "hierarchical_gate_passed",
            "holm_adjusted_p",
            "victim_seed_offset",
        ):
            self.assertNotIn(superseded, sources)
        self.assertTrue(
            all(
                cell.get("execution_count") is None
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )
        self.assertTrue(
            all(
                not cell.get("outputs")
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
        )


if __name__ == "__main__":
    unittest.main()
