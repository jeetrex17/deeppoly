from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from rl_transfer.phase2_residual_d2 import (
    D2_BC_EPOCHS,
    D2_GROUPDRO_ETA,
    D2_POLICY_SEEDS,
    D2_SOURCE_FAMILIES,
    D2_SOURCE_FOLD_SEED,
    D2FamilyThresholdMetrics,
    D2PromotionDecision,
    D2SourceMetric,
    D2SourceRole,
    D2SourceRoles,
    D2ThresholdCandidate,
    ResidualD2Request,
    allocate_d2_roles,
    residual_d2_promotion_decision,
    select_family_safe_threshold,
)


class ResidualD2RequestTests(unittest.TestCase):
    def _request(self, **overrides: object) -> ResidualD2Request:
        values: dict[str, object] = {
            "source_manifest": Path("/research-test/source/manifest.json"),
            "source_root": Path("/research-test/source"),
            "output_dir": Path("/research-test/residual-d2"),
            "data_root": Path("/research-test/cifar10"),
        }
        return ResidualD2Request(**{**values, **overrides})

    def test_request_is_frozen_and_locked_to_preregistered_source_only_d2(
        self,
    ) -> None:
        request = self._request()

        self.assertEqual(request.source_fold_seed, D2_SOURCE_FOLD_SEED)
        self.assertEqual(request.policy_seeds, D2_POLICY_SEEDS)
        self.assertEqual(request.bc_epochs, D2_BC_EPOCHS)
        self.assertEqual(request.groupdro_eta, D2_GROUPDRO_ETA)
        self.assertEqual(request.device, "cuda")
        self.assertTrue(request.source_only)
        self.assertFalse(request.hidden_target_evaluation)
        self.assertEqual(request.hidden_target_calls, 0)
        with self.assertRaises(FrozenInstanceError):
            request.device = "cpu"  # type: ignore[misc]

    def test_request_rejects_any_protocol_change_or_target_access(self) -> None:
        invalid = (
            ("source_fold_seed", 18),
            ("policy_seeds", (223, 227)),
            ("policy_seeds", (223, 227, 231)),
            ("bc_epochs", 8),
            ("groupdro_eta", 0.2),
            ("device", "mps"),
            ("source_only", False),
            ("hidden_target_evaluation", True),
            ("hidden_target_calls", 1),
            ("download", True),
        )
        for field, value in invalid:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self._request(**{field: value})


class D2RoleAllocationTests(unittest.TestCase):
    def _roles(self) -> D2SourceRoles:
        return allocate_d2_roles(
            policy_train_candidates=tuple(range(1_400)),
            source_validation_candidates=tuple(range(1_100)),
            forbidden_policy_train_indices=tuple(range(600)),
            forbidden_source_validation_indices=tuple(range(700)),
        )

    def test_allocation_uses_only_explicit_untouched_candidates(self) -> None:
        roles = self._roles()

        self.assertEqual(roles.groupdro_training.sample_ids, tuple(range(600, 1_200)))
        self.assertEqual(
            roles.threshold_selection.sample_ids,
            tuple(range(700, 800)),
        )
        self.assertEqual(roles.competence_gate.sample_ids, tuple(range(800, 900)))
        self.assertEqual(roles.evaluation.sample_ids, tuple(range(900, 1_000)))
        self.assertEqual(roles.groupdro_training.split, "policy_train")
        self.assertTrue(
            all(
                role.split == "source_validation"
                for role in (
                    roles.threshold_selection,
                    roles.competence_gate,
                    roles.evaluation,
                )
            )
        )

    def test_allocation_is_deterministic_and_pairwise_disjoint(self) -> None:
        first = self._roles()
        second = self._roles()

        self.assertEqual(first, second)
        identities = [
            {(role.split, sample_id) for sample_id in role.sample_ids}
            for role in first.as_tuple
        ]
        self.assertFalse(
            any(
                left & right
                for offset, left in enumerate(identities)
                for right in identities[offset + 1 :]
            )
        )

    def test_allocation_rejects_duplicate_or_insufficient_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            allocate_d2_roles(
                policy_train_candidates=(0, 0, *range(1, 1_300)),
                source_validation_candidates=tuple(range(1_100)),
                forbidden_policy_train_indices=tuple(range(600)),
                forbidden_source_validation_indices=tuple(range(700)),
            )
        with self.assertRaisesRegex(ValueError, "600 untouched"):
            allocate_d2_roles(
                policy_train_candidates=tuple(range(1_199)),
                source_validation_candidates=tuple(range(1_100)),
                forbidden_policy_train_indices=tuple(range(600)),
                forbidden_source_validation_indices=tuple(range(700)),
            )

    def test_allocation_requires_the_exact_historical_prefix_exclusions(
        self,
    ) -> None:
        invalid_forbidden = (
            ((), tuple(range(700))),
            (tuple(range(600)), ()),
            (tuple(range(1, 601)), tuple(range(700))),
            (tuple(range(600)), tuple(range(1, 701))),
        )
        for forbidden_policy, forbidden_validation in invalid_forbidden:
            with self.subTest(
                policy=len(forbidden_policy),
                validation=len(forbidden_validation),
            ):
                with self.assertRaisesRegex(ValueError, "exclude exactly"):
                    allocate_d2_roles(
                        policy_train_candidates=tuple(range(1_400)),
                        source_validation_candidates=tuple(range(1_100)),
                        forbidden_policy_train_indices=forbidden_policy,
                        forbidden_source_validation_indices=forbidden_validation,
                    )

    def test_roles_reject_overlap_within_a_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "pairwise disjoint"):
            D2SourceRoles(
                groupdro_training=D2SourceRole(
                    "groupdro_training",
                    "policy_train",
                    tuple(range(600)),
                ),
                threshold_selection=D2SourceRole(
                    "threshold_selection",
                    "source_validation",
                    tuple(range(700, 800)),
                ),
                competence_gate=D2SourceRole(
                    "competence_gate",
                    "source_validation",
                    tuple(range(799, 899)),
                ),
                evaluation=D2SourceRole(
                    "evaluation",
                    "source_validation",
                    tuple(range(900, 1_000)),
                ),
            )


def _family_metrics(
    classical_accuracy: float,
    transformer_accuracy: float,
) -> tuple[D2FamilyThresholdMetrics, ...]:
    return (
        D2FamilyThresholdMetrics(
            family="classical_cnn",
            accuracy=classical_accuracy,
            prior_accuracy=0.70,
        ),
        D2FamilyThresholdMetrics(
            family="transformer",
            accuracy=transformer_accuracy,
            prior_accuracy=0.75,
        ),
    )


class D2ThresholdSelectionTests(unittest.TestCase):
    def _fallback(self) -> D2ThresholdCandidate:
        return D2ThresholdCandidate(
            threshold=1.01,
            family_metrics=_family_metrics(0.70, 0.75),
            residual_use_fraction=0.0,
            overrides_enabled=False,
            always_fallback=True,
        )

    def test_selector_rejects_macro_gain_that_regresses_one_family(self) -> None:
        unsafe_high_macro = D2ThresholdCandidate(
            threshold=0.1,
            family_metrics=_family_metrics(0.69, 0.90),
            residual_use_fraction=0.5,
            overrides_enabled=True,
        )
        safe = D2ThresholdCandidate(
            threshold=0.2,
            family_metrics=_family_metrics(0.71, 0.76),
            residual_use_fraction=0.3,
            overrides_enabled=True,
        )

        selection = select_family_safe_threshold(
            (unsafe_high_macro, safe, self._fallback())
        )

        self.assertEqual(selection.selected, safe)
        self.assertEqual(selection.safe_candidate_count, 2)
        self.assertTrue(selection.every_family_accuracy_non_regression)

    def test_selector_can_choose_exact_always_fallback(self) -> None:
        unsafe = D2ThresholdCandidate(
            threshold=0.1,
            family_metrics=_family_metrics(0.69, 0.74),
            residual_use_fraction=0.7,
            overrides_enabled=True,
        )

        selection = select_family_safe_threshold((unsafe, self._fallback()))

        self.assertTrue(selection.selected.always_fallback)
        self.assertFalse(selection.selected.overrides_enabled)
        self.assertEqual(selection.selected.residual_use_fraction, 0.0)

    def test_selector_requires_one_valid_always_fallback_candidate(self) -> None:
        ordinary = D2ThresholdCandidate(
            threshold=0.2,
            family_metrics=_family_metrics(0.71, 0.76),
            residual_use_fraction=0.3,
            overrides_enabled=True,
        )
        with self.assertRaisesRegex(ValueError, "always-fallback"):
            select_family_safe_threshold((ordinary,))
        with self.assertRaises(ValueError):
            replace(self._fallback(), overrides_enabled=True)


def _passing_metrics() -> tuple[D2SourceMetric, ...]:
    return tuple(
        D2SourceMetric(
            seed=seed,
            family=family,
            baseline_asr=0.10,
            learned_asr=0.11,
            baseline_query_auc=0.05,
            learned_query_auc=0.056,
        )
        for seed in D2_POLICY_SEEDS
        for family in D2_SOURCE_FAMILIES
    )


class D2PromotionDecisionTests(unittest.TestCase):
    def test_exact_three_seed_non_regression_can_authorize_only_source_ppo(
        self,
    ) -> None:
        decision = residual_d2_promotion_decision(
            _passing_metrics(),
            source_gates_passed=True,
            artifact_audits_passed=True,
        )

        self.assertTrue(decision.passed)
        self.assertTrue(decision.eligible_for_source_only_ppo)
        self.assertFalse(decision.authorizes_hidden_target_evaluation)
        self.assertFalse(decision.hidden_target_evaluation_performed)
        self.assertGreater(decision.mean_asr_gain, 0.0)
        self.assertGreater(decision.mean_query_auc_gain, 0.0)
        self.assertTrue(decision.mean_gain_gate_passed)
        self.assertTrue(decision.worst_family_mean_gain_gate_passed)

    def test_any_seed_family_asr_or_auc_regression_fails_closed(self) -> None:
        passing = _passing_metrics()
        regressions = (
            (0, replace(passing[0], learned_asr=0.099)),
            (-1, replace(passing[-1], learned_query_auc=0.049)),
        )
        for index, regressed in regressions:
            with self.subTest(regressed=regressed):
                metrics = tuple(
                    regressed if offset == index % len(passing) else item
                    for offset, item in enumerate(passing)
                )
                decision = residual_d2_promotion_decision(
                    metrics,
                    source_gates_passed=True,
                    artifact_audits_passed=True,
                )
                self.assertFalse(decision.passed)
                self.assertFalse(decision.eligible_for_source_only_ppo)
                self.assertFalse(decision.authorizes_hidden_target_evaluation)

    def test_non_regression_alone_does_not_pass_effect_size_gates(self) -> None:
        unchanged = tuple(
            replace(
                metric,
                learned_asr=metric.baseline_asr,
                learned_query_auc=metric.baseline_query_auc,
            )
            for metric in _passing_metrics()
        )

        decision = residual_d2_promotion_decision(
            unchanged,
            source_gates_passed=True,
            artifact_audits_passed=True,
        )

        self.assertTrue(decision.all_seed_family_non_regression)
        self.assertFalse(decision.mean_gain_gate_passed)
        self.assertFalse(decision.worst_family_mean_gain_gate_passed)
        self.assertFalse(decision.passed)

    def test_decision_rejects_missing_duplicate_or_extra_seed_family_cells(
        self,
    ) -> None:
        passing = _passing_metrics()
        invalid = (
            passing[:-1],
            (*passing[:-1], passing[0]),
            (*passing, passing[0]),
        )
        for metrics in invalid:
            with self.subTest(metrics=len(metrics)):
                with self.assertRaisesRegex(ValueError, "exactly"):
                    residual_d2_promotion_decision(
                        metrics,
                        source_gates_passed=True,
                        artifact_audits_passed=True,
                    )

    def test_metric_rejects_an_unregistered_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed"):
            D2SourceMetric(
                seed=233,
                family="classical_cnn",
                baseline_asr=0.1,
                learned_asr=0.1,
                baseline_query_auc=0.05,
                learned_query_auc=0.05,
            )

    def test_decision_schema_cannot_be_used_to_authorize_hidden_targets(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "target access"):
            D2PromotionDecision(
                passed=True,
                all_seed_family_non_regression=True,
                source_gates_passed=True,
                artifact_audits_passed=True,
                eligible_for_source_only_ppo=True,
                mean_asr_gain=0.01,
                mean_query_auc_gain=0.01,
                worst_family_mean_asr_gain=0.01,
                worst_family_mean_query_auc_gain=0.01,
                mean_gain_gate_passed=True,
                worst_family_mean_gain_gate_passed=True,
                authorizes_hidden_target_evaluation=True,
            )

    def test_decision_schema_recomputes_numeric_gain_gate_invariants(self) -> None:
        with self.assertRaisesRegex(ValueError, "mean-gain gate"):
            D2PromotionDecision(
                passed=True,
                all_seed_family_non_regression=True,
                source_gates_passed=True,
                artifact_audits_passed=True,
                eligible_for_source_only_ppo=True,
                mean_asr_gain=-1.0,
                mean_query_auc_gain=-1.0,
                worst_family_mean_asr_gain=-1.0,
                worst_family_mean_query_auc_gain=-1.0,
                mean_gain_gate_passed=True,
                worst_family_mean_gain_gate_passed=True,
            )


if __name__ == "__main__":
    unittest.main()
