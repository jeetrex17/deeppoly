from __future__ import annotations

import copy
import unittest

from rl_transfer.phase2_residual_d1b_reporting import (
    residual_d1b_selection_decision,
)


SOURCE_FAMILIES = ("classical_cnn", "transformer")


def _competence() -> dict[str, object]:
    by_family = {
        family: {
            "accuracy_gain_vs_prior": 0.05,
            "soft_ce_improvement_vs_prior": 0.08,
            "residual_use_fraction": 0.20,
        }
        for family in SOURCE_FAMILIES
    }
    return {
        "target_mode": "all_soft",
        "target_calls": 0,
        "hidden_target_calls": 0,
        "accepted_steps": 100,
        "by_source_family": by_family,
        "equal_family_macro": {
            "accuracy_gain_vs_prior": 0.05,
            "soft_ce_improvement_vs_prior": 0.08,
            "residual_use_fraction": 0.20,
        },
        "worst_family": {
            "accuracy_gain_vs_prior": 0.05,
            "soft_ce_improvement_vs_prior": 0.08,
        },
    }


def _threshold() -> dict[str, object]:
    return {
        "selection_role": "d1b_threshold_selection_only",
        "threshold": 0.2,
        "accepted_steps": 100,
        "overrides_enabled": True,
        "target_calls": 0,
        "hidden_target_calls": 0,
    }


def _method(
    *,
    eligible: int,
    successes: int,
    auc: float,
    learned: int = 0,
    fallback: int = 0,
) -> dict[str, object]:
    return {
        "eligible": eligible,
        "successes": successes,
        "asr_query_auc": auc,
        "learned_override_decisions": learned,
        "score_fallback_decisions": fallback,
        "target_calls": 0,
        "hidden_target_calls": 0,
    }


def _conditions() -> dict[str, object]:
    return {
        family: {
            "audit": {"passed": True, "hidden_target_calls": 0},
            "target_calls": 0,
            "hidden_target_calls": 0,
            "methods": {
                "score_greedy": _method(
                    eligible=40,
                    successes=8,
                    auc=0.14,
                ),
                "residual_ranker_bc": _method(
                    eligible=40,
                    successes=9,
                    auc=0.15,
                    learned=20,
                    fallback=80,
                ),
                "residual_ranker_bc_ppo": _method(
                    eligible=40,
                    successes=10,
                    auc=0.16,
                    learned=25,
                    fallback=75,
                ),
            },
        }
        for family in SOURCE_FAMILIES
    }


class ResidualD1BReportingTests(unittest.TestCase):
    def test_selects_ppo_only_after_bc_reproduces_and_all_gates_pass(self) -> None:
        decision = residual_d1b_selection_decision(
            _competence(),
            _threshold(),
            _conditions(),
        )

        self.assertTrue(decision["passed"])
        self.assertTrue(decision["bc_reproduction_gate_passed"])
        self.assertTrue(decision["ppo_competence_gate_passed"])
        self.assertTrue(decision["ppo_vs_score_gate_passed"])
        self.assertTrue(decision["ppo_vs_bc_gate_passed"])
        self.assertEqual(decision["selected_method"], "residual_ranker_bc_ppo")
        self.assertEqual(decision["target_calls"], 0)
        self.assertFalse(decision["authorizes_hidden_target_evaluation"])
        self.assertEqual(
            set(decision["observed_point_estimate_gaps"]),
            set(SOURCE_FAMILIES),
        )

    def test_retains_frozen_bc_when_ppo_regresses_against_bc(self) -> None:
        conditions = _conditions()
        for family in SOURCE_FAMILIES:
            methods = conditions[family]["methods"]
            methods["residual_ranker_bc_ppo"]["successes"] = 9
            methods["residual_ranker_bc_ppo"]["asr_query_auc"] = 0.15
        conditions["transformer"]["methods"]["residual_ranker_bc_ppo"][
            "asr_query_auc"
        ] = 0.149

        decision = residual_d1b_selection_decision(
            _competence(),
            _threshold(),
            conditions,
        )

        self.assertTrue(decision["passed"])
        self.assertTrue(decision["ppo_vs_score_gate_passed"])
        self.assertFalse(decision["ppo_vs_bc_gate_passed"])
        self.assertEqual(decision["selected_method"], "residual_ranker_bc")
        self.assertEqual(
            decision["ppo_rejection_reason"],
            "ppo_observed_regression_against_frozen_bc",
        )

    def test_fails_when_frozen_bc_does_not_reproduce(self) -> None:
        conditions = _conditions()
        methods = conditions["transformer"]["methods"]
        methods["residual_ranker_bc"]["successes"] = 7

        decision = residual_d1b_selection_decision(
            _competence(),
            _threshold(),
            conditions,
        )

        self.assertFalse(decision["passed"])
        self.assertFalse(decision["bc_reproduction_gate_passed"])
        self.assertIsNone(decision["selected_method"])
        self.assertEqual(
            decision["failure_reason"],
            "frozen_bc_did_not_reproduce_on_reserved_d1b_cohort",
        )

    def test_requires_positive_worst_family_competence_and_material_use(
        self,
    ) -> None:
        competence = _competence()
        competence["by_source_family"]["transformer"]["accuracy_gain_vs_prior"] = 0.0
        competence["equal_family_macro"]["accuracy_gain_vs_prior"] = 0.025
        competence["worst_family"]["accuracy_gain_vs_prior"] = 0.0
        conditions = _conditions()
        for family in SOURCE_FAMILIES:
            methods = conditions[family]["methods"]
            methods["residual_ranker_bc_ppo"]["learned_override_decisions"] = 0
            methods["residual_ranker_bc_ppo"]["score_fallback_decisions"] = 100

        decision = residual_d1b_selection_decision(
            competence,
            _threshold(),
            conditions,
        )

        self.assertTrue(decision["passed"])
        self.assertFalse(decision["ppo_competence_gate_passed"])
        self.assertFalse(decision["ppo_deployment_gate_passed"])
        self.assertEqual(decision["selected_method"], "residual_ranker_bc")
        self.assertEqual(
            decision["ppo_rejection_reason"],
            "ppo_source_competence_or_deployment_gate_failed",
        )

    def test_rejects_wrong_roles_missing_families_and_target_evidence(self) -> None:
        bad_threshold = _threshold()
        bad_threshold["selection_role"] = "d1a_evaluation"
        with self.assertRaisesRegex(ValueError, "threshold|role"):
            residual_d1b_selection_decision(
                _competence(),
                bad_threshold,
                _conditions(),
            )

        missing = _conditions()
        del missing["transformer"]
        with self.assertRaisesRegex(ValueError, "famil"):
            residual_d1b_selection_decision(
                _competence(),
                _threshold(),
                missing,
            )

        for location in ("competence", "threshold", "conditions"):
            with self.subTest(location=location):
                competence = copy.deepcopy(_competence())
                threshold = copy.deepcopy(_threshold())
                conditions = copy.deepcopy(_conditions())
                if location == "competence":
                    competence["hidden_target_calls"] = 1
                elif location == "threshold":
                    threshold["target_calls"] = 1
                else:
                    conditions["classical_cnn"]["target_calls"] = 1
                with self.assertRaisesRegex(ValueError, "target|source"):
                    residual_d1b_selection_decision(
                        competence,
                        threshold,
                        conditions,
                    )

    def test_rejects_nonfinite_or_mismatched_method_cohorts(self) -> None:
        conditions = _conditions()
        conditions["classical_cnn"]["methods"]["residual_ranker_bc"]["eligible"] = 39
        with self.assertRaisesRegex(ValueError, "eligible|cohort"):
            residual_d1b_selection_decision(
                _competence(),
                _threshold(),
                conditions,
            )

        conditions = _conditions()
        conditions["transformer"]["methods"]["residual_ranker_bc_ppo"][
            "asr_query_auc"
        ] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite|metric"):
            residual_d1b_selection_decision(
                _competence(),
                _threshold(),
                conditions,
            )


if __name__ == "__main__":
    unittest.main()
