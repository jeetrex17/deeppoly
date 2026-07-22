import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rl_transfer.audit import AuditedVictim, QueryBudgetExceeded
from rl_transfer.actions import apply_action, dct_catalog, patch_catalog
from rl_transfer.cli import run_research_smoke, validate_full_config
from rl_transfer.population import FamilyRobustWeights, balanced_family_schedule
from rl_transfer.recurrent import PPOSequence, RecurrentAttackPolicy
from rl_transfer.config import AttackConfig
from rl_transfer.research_protocol import run_frozen_episode
from rl_transfer.registry import ExperimentSplit, VictimRegistry, VictimSpec
from rl_transfer.research_metrics import AttackOutcome, asr_query_auc, asr_at_budgets
from rl_transfer.results import ResearchResultRow
from rl_transfer.statistics import bootstrap_interval, paired_permutation_pvalue


class TwoClassVictim(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(1, 2, 3))
        return torch.stack((means, 1 - means), dim=1)


class NonFiniteVictim(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.tensor(((float("nan"), 0.0),), device=images.device)


class CrossVictimResearchTests(unittest.TestCase):
    def test_holdout_rejects_family_leakage(self) -> None:
        registry = VictimRegistry((
            VictimSpec("resnet18", "classical_cnn", "local", "identity", 0.8, "BSD-3-Clause"),
            VictimSpec("resnet50", "classical_cnn", "local", "identity", 0.81, "BSD-3-Clause"),
            VictimSpec("convnext", "modern_cnn", "local", "identity", 0.82, "BSD-3-Clause"),
            VictimSpec("vit_tiny", "transformer", "local", "identity", 0.75, "Apache-2.0"),
        ))
        registry.validate_split(ExperimentSplit(("resnet18",), ("convnext",), ("vit_tiny",)))
        with self.assertRaises(ValueError):
            registry.validate_split(ExperimentSplit((), ("convnext",), ("vit_tiny",)))
        with self.assertRaises(ValueError):
            registry.validate_split(ExperimentSplit(("resnet18",), ("resnet50",), ("vit_tiny",)))
        with self.assertRaises(ValueError):
            registry.validate_split(ExperimentSplit(("resnet18",), ("convnext",), ("resnet50",)))

    def test_audited_victim_counts_initialization_inside_budget(self) -> None:
        oracle = AuditedVictim(TwoClassVictim(), budget=2, feedback="scores", victim_id="toy")
        image = torch.full((1, 4, 4), 0.7)
        oracle.query(image, sample_id="x", purpose="initialization", step=0)
        oracle.query(image, sample_id="x", purpose="attack", step=1)
        with self.assertRaises(QueryBudgetExceeded):
            oracle.query(image, sample_id="x", purpose="attack", step=2)
        self.assertEqual(len(oracle.trace), 2)

    def test_feedback_none_redacts_scores_and_label(self) -> None:
        oracle = AuditedVictim(TwoClassVictim(), budget=1, feedback="none", victim_id="toy")
        response = oracle.query(torch.full((1, 4, 4), 0.7), "x", "evaluation", 0)
        self.assertIsNone(response.scores)
        self.assertIsNone(response.predicted_label)
        self.assertIsNone(oracle.trace[0].predicted_label)

    def test_audited_victim_rejects_non_finite_logits_and_counts_call(self) -> None:
        oracle = AuditedVictim(NonFiniteVictim(), budget=1, feedback="scores", victim_id="invalid")
        with self.assertRaises(ValueError):
            oracle.query(torch.zeros((1, 4, 4)), "x", "evaluation", 0)
        self.assertEqual(oracle.calls, 1)
        self.assertEqual(oracle.trace[0].error, "ValueError")

    def test_recurrent_hidden_state_is_ephemeral_and_policy_is_frozen(self) -> None:
        policy = RecurrentAttackPolicy(observation_dim=6, action_dim=4, hidden_dim=8, seed=3)
        before = policy.persistent_digest()
        hidden = policy.initial_state()
        _, first_hidden = policy.act(np.zeros(6, dtype=np.float32), hidden, deterministic=True)
        _, second_hidden = policy.act(np.ones(6, dtype=np.float32), first_hidden, deterministic=True)
        self.assertFalse(torch.equal(first_hidden, second_hidden))
        self.assertEqual(before, policy.persistent_digest())
        self.assertTrue(torch.equal(policy.initial_state(), hidden))

    def test_recurrent_ppo_unroll_updates_memory(self) -> None:
        policy = RecurrentAttackPolicy(observation_dim=3, action_dim=2, hidden_dim=4, seed=2)
        observations = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        with torch.no_grad():
            hidden = policy.initial_state(); old_logs = []
            for observation in observations:
                logits, _, hidden = policy(observation, hidden)
                old_logs.append(torch.log_softmax(logits, 0)[0])
        sequence = PPOSequence(observations, torch.zeros(3, dtype=torch.long), torch.stack(old_logs), torch.ones(3), torch.zeros(3))
        before = policy.memory.weight_hh.detach().clone()
        metrics = policy.ppo_update_sequences([(sequence, 1.0)])
        self.assertFalse(torch.equal(before, policy.memory.weight_hh))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_population_schedule_and_groupdro_are_deterministic(self) -> None:
        schedule = balanced_family_schedule(("cnn", "vit", "modern"), episodes=7, seed=5)
        self.assertEqual(schedule, balanced_family_schedule(("cnn", "vit", "modern"), 7, 5))
        counts = {family: schedule.count(family) for family in set(schedule)}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        weights = FamilyRobustWeights(("cnn", "vit"), eta=0.5).update({"cnn": 1.0, "vit": 3.0})
        self.assertGreater(weights["vit"], weights["cnn"])

    def test_asr_query_metrics_use_full_curve(self) -> None:
        outcomes = (
            AttackOutcome(True, 2),
            AttackOutcome(False, None),
            AttackOutcome(True, 5),
            AttackOutcome(True, 1),
            AttackOutcome(True, None),
        )
        rates = asr_at_budgets(outcomes, (0, 2, 5))
        self.assertEqual(rates, {0: 0.0, 2: 0.5, 5: 0.75})
        self.assertAlmostEqual(asr_query_auc(rates), 0.475)
        with self.assertRaises(ValueError):
            asr_at_budgets((AttackOutcome(False, None),), (0, 5))

    def test_patch_and_dct_actions_share_projection(self) -> None:
        original = torch.full((3, 8, 8), 0.95)
        for action in (patch_catalog(2)[0], dct_catalog(2)[-1]):
            adversarial = apply_action(original, original, action, epsilon=0.1, step_size=0.2, grid_size=2)
            self.assertLessEqual(float((adversarial - original).abs().max()), 0.100001)
            self.assertTrue(torch.all((0 <= adversarial) & (adversarial <= 1)))
        dct = dct_catalog(3)[-1]
        adversarial = apply_action(original, original, dct, epsilon=0.1, step_size=0.02, grid_size=2)
        self.assertGreater(float((adversarial - original).abs().sum()), 0.0)

    def test_rejects_policy_action_catalog_mismatch(self) -> None:
        policy = RecurrentAttackPolicy(8, action_dim=2, hidden_dim=8)
        with self.assertRaises(ValueError):
            run_frozen_episode(policy, TwoClassVictim(), torch.full((3, 4, 4), 0.7), 0, "x", "v", "cnn", AttackConfig(grid_size=1, max_queries=2))

    def test_result_schema_and_statistics(self) -> None:
        row = ResearchResultRow("x", "v", "cnn", "ppo", "T1", 1, 5, True, False, None, 5, 0.1, 1.0, "digest", (0, 1))
        self.assertEqual(row.total_target_calls, 5)
        with self.assertRaises(ValueError):
            ResearchResultRow("x", "v", "cnn", "ppo", "T1", 1, 5, True, False, 3, 5, 0.1, 1.0, "digest", ())
        self.assertEqual(bootstrap_interval((0.0, 1.0), samples=20, seed=2), bootstrap_interval((0.0, 1.0), samples=20, seed=2))
        self.assertLessEqual(paired_permutation_pvalue((1, 1, 1), (0, 0, 0), permutations=100, seed=1), 1.0)

    def test_notebook_is_thin_and_valid_json(self) -> None:
        notebook = Path("notebooks/rl_cross_victim_pilot.ipynb")
        payload = json.loads(notebook.read_text())
        code = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"] if cell["cell_type"] == "code")
        self.assertEqual(payload["nbformat"], 4)
        self.assertIn("from rl_transfer", code)
        self.assertNotIn("class DQNAgent", code)
        self.assertNotIn("!pip", code)
        self.assertNotIn("/content", code)

    def test_research_smoke_exercises_frozen_protocol_and_raw_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "research.json"
            result = run_research_smoke(output, seed=7)
            frozen = result["frozen_t1"]
            raw_results = Path(result["manifest"]["raw_results"])
            self.assertFalse(result["manifest"]["research_valid"])
            self.assertEqual(frozen["policy_digest_before"], frozen["policy_digest_after"])
            self.assertEqual(frozen["total_target_calls"], result["manifest"]["total_query_budget"])
            self.assertTrue(output.is_file())
            self.assertEqual(len(raw_results.read_text().splitlines()), 1)

    def test_full_config_matches_prespecified_research_grid(self) -> None:
        config = validate_full_config(Path("configs/rl_transfer/imagenet1k_lofo.json"))
        self.assertEqual(config["primary_threat_model"], "T1-frozen-score-based")
        self.assertEqual(config["seeds"], [1, 2, 3, 4, 5])
        with tempfile.TemporaryDirectory() as directory:
            for key, invalid_value in (
                ("inner_source_family_validation", False),
                ("research_valid", False),
                ("primary_threat_model", "T3-limited-adaptation"),
                ("epsilon_values", [8 / 255]),
                ("seeds", [1, 1, 1, 1, 1]),
                ("stage_images", 999),
                ("confirmation_images", 4999),
                ("robust_stress_models", 1),
                ("required_baselines", ["random"]),
                ("execute_in_ci", True),
            ):
                invalid = {**copy.deepcopy(config), key: invalid_value}
                path = Path(directory) / f"invalid-{key}.json"
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    validate_full_config(path)


if __name__ == "__main__":
    unittest.main()
