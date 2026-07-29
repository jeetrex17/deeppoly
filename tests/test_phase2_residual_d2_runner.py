from __future__ import annotations

import unittest

from rl_transfer.phase2_residual_d2_runner import (
    _family_safe_threshold,
    _metric,
    _plain_training,
)
from rl_transfer.residual_groupdro import GroupDROAudit, GroupDROFamilyAudit, GroupDROState


class D2ThresholdAdapterTests(unittest.TestCase):
    def test_plain_training_serializes_groupdro_audits(self) -> None:
        audit = GroupDROAudit(
            step=1,
            eta=0.1,
            objective=1.0,
            families=(
                GroupDROFamilyAudit("classical_cnn", 1, 1.0, 0.5, 0.5),
                GroupDROFamilyAudit("transformer", 1, 1.0, 0.5, 0.5),
            ),
        )
        serialized = _plain_training(
            {"groupdro_state": GroupDROState.uniform(("classical_cnn", "transformer")), "groupdro_audits": (audit,)}
        )

        self.assertEqual(serialized["groupdro_state"]["step"], 0)
        self.assertEqual(serialized["groupdro_audits"][0]["step"], 1)

    def test_metric_adapter_accepts_the_audited_legacy_evaluator_label(self) -> None:
        metric = _metric(
            {
                "methods": {
                    "score_greedy": {
                        "eligible": 10,
                        "successes": 1,
                        "asr_query_auc": 0.05,
                    },
                    "residual_ranker_bc": {
                        "eligible": 10,
                        "successes": 2,
                        "asr_query_auc": 0.06,
                    },
                }
            },
            seed=223,
            family="classical_cnn",
        )

        self.assertAlmostEqual(metric.asr_gain, 0.1)
        self.assertAlmostEqual(metric.query_auc_gain, 0.01)

    def test_family_safety_compares_candidate_to_always_fallback_baseline(self) -> None:
        fallback = {
            "threshold": 1.01,
            "selection_mode": "always_fallback",
            "by_source_family": {
                "classical_cnn": {"accuracy": 0.70},
                "transformer": {"accuracy": 0.75},
            },
        }
        selected = {
            "threshold": 0.2,
            "selection_mode": "confidence_gate",
            "by_source_family": {
                "classical_cnn": {"accuracy": 0.70},
                "transformer": {"accuracy": 0.76},
            },
        }
        threshold, safe, chosen = _family_safe_threshold(
            {"threshold": 0.2, "candidate_evaluations": [selected, fallback]}
        )

        self.assertEqual(threshold, 0.2)
        self.assertTrue(safe)
        self.assertEqual(chosen, selected)

    def test_family_regression_selects_the_verified_fallback(self) -> None:
        fallback = {
            "threshold": 1.01,
            "selection_mode": "always_fallback",
            "by_source_family": {
                "classical_cnn": {"accuracy": 0.70},
                "transformer": {"accuracy": 0.75},
            },
        }
        selected = {
            "threshold": 0.2,
            "selection_mode": "confidence_gate",
            "by_source_family": {
                "classical_cnn": {"accuracy": 0.69},
                "transformer": {"accuracy": 0.80},
            },
        }
        threshold, safe, chosen = _family_safe_threshold(
            {"threshold": 0.2, "candidate_evaluations": [selected, fallback]}
        )

        self.assertEqual(threshold, 1.01)
        self.assertFalse(safe)
        self.assertEqual(chosen, fallback)


if __name__ == "__main__":
    unittest.main()
