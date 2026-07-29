from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import torch
from torch import nn

from rl_transfer.config import AttackConfig
from rl_transfer.imitation import BehaviorCloneStep
from rl_transfer.phase2_residual_d1 import (
    D1_SOURCE_FAMILIES,
    ResidualCacheBinding,
    ResidualD1Request,
    residual_d1_promotion_decision,
    select_residual_action,
    validate_d1_attack_contract,
    validate_residual_cache_binding,
)
from rl_transfer.phase2_residual_d1_runner import _decision
from rl_transfer.recurrent import RecurrentAttackPolicy
from rl_transfer.research_protocol import run_score_greedy_episode
from rl_transfer.residual_bc import fit_residual_ranker_bc
from rl_transfer.residual_ranker import (
    ResidualRankerPolicy,
    evaluate_residual_ranker_examples,
    run_residual_ranker_episode,
    score_greedy_action_order,
    select_confidence_threshold,
)


class ResidualD1RequestTests(unittest.TestCase):
    def _request(self, **overrides: object) -> ResidualD1Request:
        values: dict[str, object] = {
            "source_manifest": Path("/tmp/source/manifest.json"),
            "source_root": Path("/tmp/source"),
            "output_dir": Path("/tmp/residual-d1"),
            "data_root": Path("/tmp/cifar10"),
        }
        return ResidualD1Request(**{**values, **overrides})

    def test_request_is_immutable_and_locked_to_the_source_only_d1_rung(
        self,
    ) -> None:
        request = self._request()

        self.assertEqual(request.heldout_family, "modern_cnn")
        self.assertEqual(request.seed, 17)
        self.assertEqual(request.source_images, 50)
        self.assertEqual(request.deadline_seconds, 28_800.0)
        self.assertEqual(request.bc_episodes, 200)
        self.assertEqual(request.ppo_episodes, 0)
        self.assertEqual(request.device, "cuda")
        with self.assertRaises(FrozenInstanceError):
            request.seed = 29  # type: ignore[misc]

    def test_request_rejects_changes_to_preregistered_d1_limits(self) -> None:
        invalid_values = (
            ("heldout_family", "classical_cnn"),
            ("seed", 29),
            ("source_images", 51),
            ("deadline_seconds", 28_800.0001),
            ("bc_episodes", 199),
            ("ppo_episodes", -1),
            ("ppo_episodes", 1),
            ("ppo_episodes", 200),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "D1|locked|preregister|held.?out|seed|image|deadline|BC|PPO",
                ):
                    self._request(**{field: value})


class ResidualActionTests(unittest.TestCase):
    def test_low_confidence_returns_the_score_greedy_prior_exactly(self) -> None:
        self.assertEqual(
            select_residual_action(
                score_greedy_action=7,
                learned_action=31,
                residual_confidence=0.49,
                confidence_threshold=0.50,
            ),
            7,
        )

    def test_high_confidence_returns_the_learned_action(self) -> None:
        self.assertEqual(
            select_residual_action(
                score_greedy_action=7,
                learned_action=31,
                residual_confidence=0.51,
                confidence_threshold=0.50,
            ),
            31,
        )

    def test_prior_cursor_matches_score_greedy_at_every_step(self) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=4,
            hidden_dim=4,
            seed=3,
        )
        for parameter in backbone.parameters():
            parameter.data.zero_()
        policy = ResidualRankerPolicy(
            backbone,
            confidence_threshold=1.0,
            prior_temperature=24.0,
        )
        order = (2, 0, 3, 1)

        for proposal_index, expected in enumerate(order):
            with self.subTest(proposal_index=proposal_index):
                logits, _ = policy.combined_logits(
                    torch.zeros(2),
                    policy.initial_state(),
                    prior_order=order,
                    proposal_index=proposal_index,
                )
                self.assertEqual(int(logits.argmax()), expected)

    def test_threshold_selector_can_choose_exact_always_fallback(self) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=2,
            hidden_dim=4,
            seed=4,
        )
        for parameter in backbone.parameters():
            parameter.data.zero_()
        trajectory_id = "synthetic-threshold"
        order = score_greedy_action_order(
            action_dim=2,
            seed=17,
            sample_id=trajectory_id,
        )
        wrong_action = order[1]
        backbone.actor.bias.data[wrong_action] = 5.0
        step = BehaviorCloneStep(
            (0.0, 0.0),
            order[0],
            True,
            trajectory_id=trajectory_id,
            step_index=0,
            action_distribution=(
                1.0 if order[0] == 0 else 0.0,
                1.0 if order[0] == 1 else 0.0,
            ),
        )

        selected = select_confidence_threshold(
            backbone,
            (step,),
            seed=17,
            thresholds=(0.0,),
        )

        self.assertEqual(selected["accuracy"], 1.0)
        self.assertEqual(selected["prior_accuracy"], 1.0)
        self.assertEqual(selected["residual_use_fraction"], 0.0)
        self.assertEqual(selected["selection_mode"], "always_fallback")
        self.assertFalse(selected["overrides_enabled"])

        backbone.actor.bias.data[wrong_action] = 100.0
        policy = ResidualRankerPolicy(
            backbone,
            confidence_threshold=float(selected["threshold"]),
            overrides_enabled=bool(selected["overrides_enabled"]),
        )
        decision = policy.decide(
            torch.zeros(2).numpy(),
            policy.initial_state(),
            prior_order=order,
            proposal_index=0,
        )
        self.assertEqual(decision.action, order[0])
        self.assertFalse(decision.used_residual)

    def test_threshold_and_competence_propagate_deadline_checks(self) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=2,
            hidden_dim=4,
            seed=41,
        )
        trajectory_id = "bc-gradient-source:classical_cnn:deadline"
        step = BehaviorCloneStep(
            (0.0, 0.0),
            0,
            True,
            trajectory_id=trajectory_id,
            step_index=0,
            action_distribution=(1.0, 0.0),
        )
        policy = ResidualRankerPolicy(
            backbone,
            confidence_threshold=0.0,
        )

        def deadline() -> None:
            raise TimeoutError("shared D1 deadline")

        with self.assertRaisesRegex(TimeoutError, "shared D1 deadline"):
            select_confidence_threshold(
                backbone,
                (step,),
                seed=17,
                deadline_check=deadline,
            )
        with self.assertRaisesRegex(TimeoutError, "shared D1 deadline"):
            evaluate_residual_ranker_examples(
                policy,
                (step,),
                prior_seed=17,
                deadline_check=deadline,
            )
        with self.assertRaisesRegex(TypeError, "deadline"):
            select_confidence_threshold(
                backbone,
                (step,),
                seed=17,
                deadline_check=object(),  # type: ignore[arg-type]
            )

    def test_competence_uses_the_same_fallback_advantage_as_deployment(
        self,
    ) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=3,
            hidden_dim=4,
            seed=5,
        )
        for parameter in backbone.parameters():
            parameter.data.zero_()
        trajectory_id = "synthetic-competence"
        order = score_greedy_action_order(
            action_dim=3,
            seed=17,
            sample_id=trajectory_id,
        )
        learned_action = order[1]
        runner_up = order[2]
        backbone.actor.bias.data[learned_action] = 2.0
        backbone.actor.bias.data[runner_up] = 1.9
        policy = ResidualRankerPolicy(
            backbone,
            confidence_threshold=1.0,
        )
        distribution = tuple(
            1.0 if action == learned_action else 0.0 for action in range(3)
        )
        step = BehaviorCloneStep(
            (0.0, 0.0),
            learned_action,
            True,
            trajectory_id=trajectory_id,
            step_index=0,
            action_distribution=distribution,
        )

        metrics = evaluate_residual_ranker_examples(
            policy,
            (step,),
            prior_seed=17,
        )

        self.assertEqual(metrics["gated_top1_accuracy"], 1.0)
        self.assertEqual(metrics["residual_use_fraction"], 1.0)

    def test_disabled_residual_episode_exactly_replays_score_greedy(self) -> None:
        class ConstantVictim(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("fixed_logits", torch.tensor((2.0, 0.0)))

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.fixed_logits.repeat(value.shape[0], 1)

        attack = AttackConfig(
            epsilon=8 / 255,
            step_size=2 / 255,
            grid_size=4,
            max_queries=50,
            rollback_on_non_improvement=True,
            action_history_features=True,
            image_patch_features=True,
            image_patch_feature_mode="statistics",
        )
        backbone = RecurrentAttackPolicy(
            observation_dim=attack.recurrent_observation_dim,
            action_dim=attack.action_dim,
            hidden_dim=4,
            seed=9,
            actor_mode="action_conditioned",
            action_grid_size=attack.grid_size,
        )
        for parameter in backbone.parameters():
            parameter.data.zero_()
        policy = ResidualRankerPolicy(
            backbone,
            confidence_threshold=0.0,
            overrides_enabled=False,
        )
        victim = ConstantVictim()
        image = torch.full((3, 8, 8), 0.5)
        shared = {
            "victim": victim,
            "image": image,
            "label": 0,
            "sample_id": "synthetic-fallback",
            "victim_id": "synthetic-source-victim",
            "family": "classical_cnn",
            "config": attack,
        }

        score = run_score_greedy_episode(**shared, seed=17)
        residual = run_residual_ranker_episode(
            policy,
            **shared,
            score_prior_seed=17,
        )

        self.assertEqual(residual.actions, score.actions)
        self.assertEqual(residual.clean_correct, score.clean_correct)
        self.assertEqual(residual.success, score.success)
        self.assertEqual(residual.query_to_success, score.query_to_success)
        self.assertEqual(residual.total_target_calls, score.total_target_calls)
        self.assertEqual(residual.linf, score.linf)
        self.assertEqual(residual.l2, score.l2)
        self.assertEqual(
            residual.policy_digest_before,
            residual.policy_digest_after,
        )
        self.assertTrue(
            all(
                event["purpose"] == "residual-ranker-fallback"
                for event in residual.query_trace[1:]
            )
        )

    def test_residual_bc_reports_honest_training_metrics(self) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=2,
            hidden_dim=4,
            seed=11,
        )
        examples = (
            BehaviorCloneStep(
                (0.0, 0.0),
                0,
                True,
                trajectory_id="synthetic-train",
                step_index=0,
                action_distribution=(0.9, 0.1),
            ),
            BehaviorCloneStep(
                (0.5, -0.5),
                1,
                False,
                trajectory_id="synthetic-train",
                step_index=1,
                action_distribution=(0.2, 0.8),
            ),
            BehaviorCloneStep(
                (1.0, -1.0),
                1,
                True,
                trajectory_id="synthetic-train",
                step_index=2,
                action_distribution=(0.1, 0.9),
            ),
        )
        before = backbone.persistent_digest()

        metrics = fit_residual_ranker_bc(
            backbone,
            examples,
            epochs=1,
            seed=13,
            prior_seed=17,
            pairwise_weight=0.1,
        )

        final = metrics["final"]
        self.assertEqual(metrics["accepted_steps"], 2)
        self.assertEqual(metrics["trajectories"], 1)
        self.assertEqual(metrics["epochs"], 1)
        self.assertEqual(
            metrics["aggregation"],
            "equal_family_equal_trajectory",
        )
        self.assertEqual(
            metrics["source_family_diagnostics"]["unattributed"]["trajectories"],
            1,
        )
        self.assertEqual(len(metrics["history"]), 1)
        self.assertEqual(final, metrics["history"][-1])
        self.assertAlmostEqual(
            final["loss"],
            final["listwise_soft_cross_entropy"]
            + 0.1 * final["pairwise_logistic_loss"],
            places=6,
        )
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(value))
                for key, value in final.items()
                if key != "epoch"
            )
        )
        self.assertGreaterEqual(final["hybrid_top1_accuracy"], 0.0)
        self.assertLessEqual(final["hybrid_top1_accuracy"], 1.0)
        self.assertNotEqual(before, backbone.persistent_digest())

    def test_residual_bc_deadline_prevents_optimizer_mutation(self) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=2,
            hidden_dim=4,
            seed=23,
        )
        examples = (
            BehaviorCloneStep(
                (0.0, 0.0),
                0,
                True,
                trajectory_id="synthetic-deadline",
                step_index=0,
                action_distribution=(0.9, 0.1),
            ),
        )
        before = backbone.persistent_digest()
        checks = 0

        def deadline() -> None:
            nonlocal checks
            checks += 1
            if checks == 5:
                raise TimeoutError("synthetic deadline")

        with self.assertRaisesRegex(TimeoutError, "deadline"):
            fit_residual_ranker_bc(
                backbone,
                examples,
                epochs=1,
                seed=29,
                prior_seed=31,
                deadline_check=deadline,
            )

        self.assertEqual(checks, 5)
        self.assertEqual(backbone.persistent_digest(), before)

    def test_attack_episode_deadline_is_checked_before_each_query(self) -> None:
        class CountingVictim(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("fixed_logits", torch.tensor((2.0, 0.0)))
                self.calls = 0

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return self.fixed_logits.repeat(value.shape[0], 1)

        attack = AttackConfig(
            epsilon=8 / 255,
            step_size=2 / 255,
            grid_size=4,
            max_queries=50,
            rollback_on_non_improvement=True,
            action_history_features=True,
            image_patch_features=True,
            image_patch_feature_mode="statistics",
        )
        backbone = RecurrentAttackPolicy(
            observation_dim=attack.recurrent_observation_dim,
            action_dim=attack.action_dim,
            hidden_dim=4,
            seed=37,
            actor_mode="action_conditioned",
            action_grid_size=attack.grid_size,
        )
        policy = ResidualRankerPolicy(backbone, confidence_threshold=0.0)
        image = torch.full((3, 8, 8), 0.5)

        for method in ("score", "residual"):
            with self.subTest(method=method):
                victim = CountingVictim()
                checks = 0

                def deadline() -> None:
                    nonlocal checks
                    checks += 1
                    if checks == 2:
                        raise TimeoutError("synthetic query deadline")

                with self.assertRaisesRegex(TimeoutError, "deadline"):
                    if method == "score":
                        run_score_greedy_episode(
                            victim,
                            image,
                            0,
                            "deadline-sample",
                            "source-victim",
                            "classical_cnn",
                            attack,
                            seed=41,
                            deadline_check=deadline,
                        )
                    else:
                        run_residual_ranker_episode(
                            policy,
                            victim,
                            image,
                            0,
                            "deadline-sample",
                            "source-victim",
                            "classical_cnn",
                            attack,
                            score_prior_seed=41,
                            deadline_check=deadline,
                        )
                self.assertEqual(victim.calls, 1)

    def test_competence_reports_equal_trajectory_and_family_summaries(
        self,
    ) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=2,
            hidden_dim=4,
            seed=19,
        )
        for parameter in backbone.parameters():
            parameter.data.zero_()
        policy = ResidualRankerPolicy(
            backbone,
            confidence_threshold=0.0,
            overrides_enabled=False,
        )
        examples = []
        for offset, family in enumerate(D1_SOURCE_FAMILIES):
            trajectory_id = f"bc-gradient-source:{family}:synthetic-{offset}"
            prior = score_greedy_action_order(
                action_dim=2,
                seed=17,
                sample_id=trajectory_id,
            )[0]
            examples.append(
                BehaviorCloneStep(
                    (float(offset), 0.0),
                    prior if offset == 0 else 1 - prior,
                    True,
                    trajectory_id=trajectory_id,
                    step_index=0,
                    action_distribution=(
                        1.0 if (prior if offset == 0 else 1 - prior) == 0 else 0.0,
                        1.0 if (prior if offset == 0 else 1 - prior) == 1 else 0.0,
                    ),
                )
            )

        metrics = evaluate_residual_ranker_examples(
            policy,
            tuple(examples),
            prior_seed=17,
        )

        by_family = metrics["by_source_family"]
        self.assertEqual(set(by_family), set(D1_SOURCE_FAMILIES))
        self.assertEqual(len(metrics["by_trajectory"]), 2)
        self.assertEqual(metrics["aggregation"], "equal_trajectory_then_family")
        expected_macro = sum(
            by_family[family]["gated_top1_accuracy"] for family in D1_SOURCE_FAMILIES
        ) / len(D1_SOURCE_FAMILIES)
        self.assertAlmostEqual(
            metrics["equal_family_macro"]["gated_top1_accuracy"],
            expected_macro,
        )
        self.assertEqual(
            metrics["worst_family"]["accuracy_gain_vs_prior"],
            min(
                by_family[family]["accuracy_gain_vs_prior"]
                for family in D1_SOURCE_FAMILIES
            ),
        )

        with self.assertRaisesRegex(ValueError, "source famil"):
            evaluate_residual_ranker_examples(
                policy,
                tuple(examples[:1]),
                prior_seed=17,
                required_source_families=D1_SOURCE_FAMILIES,
            )


class ResidualCacheBindingTests(unittest.TestCase):
    def test_cache_binding_rejects_every_identity_mismatch(self) -> None:
        expected = ResidualCacheBinding(
            source_manifest_sha256="1" * 64,
            dataset_content_sha256="2" * 64,
            victim_cache_digest="3" * 64,
            request_sha256="4" * 64,
        )
        validate_residual_cache_binding(expected, expected)

        for field in (
            "source_manifest_sha256",
            "dataset_content_sha256",
            "victim_cache_digest",
            "request_sha256",
        ):
            with self.subTest(field=field):
                mismatched = replace(expected, **{field: "f" * 64})
                with self.assertRaisesRegex(
                    ValueError,
                    "cache|binding|identity|mismatch",
                ):
                    validate_residual_cache_binding(expected, mismatched)


class ResidualPromotionTests(unittest.TestCase):
    _PASSING = {
        "bc_validation_score": 0.61,
        "prior_validation_score": 0.60,
        "score_greedy_asr": 0.20,
        "score_greedy_auc": 0.10,
        "learned_asr": 0.21,
        "learned_auc": 0.11,
    }

    def test_promotion_requires_bc_improvement_over_the_oracle(self) -> None:
        passing = residual_d1_promotion_decision(**self._PASSING)
        self.assertTrue(passing["passed"])

        for bc_score in (0.60, 0.59):
            with self.subTest(bc_validation_score=bc_score):
                decision = residual_d1_promotion_decision(
                    **{
                        **self._PASSING,
                        "bc_validation_score": bc_score,
                    }
                )
                self.assertFalse(decision["passed"])
                self.assertFalse(decision["bc_improved_over_prior"])

    def test_promotion_fails_on_either_asr_or_auc_regression(self) -> None:
        regressions = (
            ("learned_asr", 0.19, "asr_observed_non_decrease"),
            ("learned_auc", 0.09, "auc_observed_non_decrease"),
        )
        for field, value, gate in regressions:
            with self.subTest(field=field):
                decision = residual_d1_promotion_decision(
                    **{**self._PASSING, field: value}
                )
                self.assertFalse(decision["passed"])
                self.assertFalse(decision[gate])

    def test_integrated_gate_accepts_the_selector_validation_role(self) -> None:
        competence = {
            "target_mode": "all_soft",
            "gated_top1_accuracy": 0.20,
            "prior_top1_accuracy": 0.10,
            "soft_cross_entropy": 1.0,
            "prior_soft_cross_entropy": 1.1,
            "residual_use_fraction": 0.10,
            "by_source_family": {
                family: {
                    "accuracy_gain_vs_prior": 0.10,
                    "soft_ce_improvement_vs_prior": 0.10,
                }
                for family in D1_SOURCE_FAMILIES
            },
            "equal_family_macro": {
                "gated_top1_accuracy": 0.20,
                "prior_top1_accuracy": 0.10,
                "soft_cross_entropy": 1.0,
                "prior_soft_cross_entropy": 1.1,
                "residual_use_fraction": 0.10,
                "accuracy_gain_vs_prior": 0.10,
                "soft_ce_improvement_vs_prior": 0.10,
            },
            "worst_family": {
                "accuracy_gain_vs_prior": 0.10,
                "soft_ce_improvement_vs_prior": 0.10,
            },
        }
        threshold = {
            "selection_role": "bc_validation_only",
            "threshold": 0.2,
        }
        conditions = {
            family: {
                "audit": {"passed": True},
                "methods": {
                    "score_greedy": {
                        "eligible": 10,
                        "successes": 1,
                        "asr_query_auc": 0.10,
                    },
                    "residual_ranker_bc": {
                        "eligible": 10,
                        "successes": 1,
                        "asr_query_auc": 0.10,
                        "learned_override_decisions": 1,
                        "score_fallback_decisions": 9,
                    },
                },
            }
            for family in D1_SOURCE_FAMILIES
        }

        decision = _decision(competence, threshold, conditions)

        self.assertTrue(decision["passed"])
        self.assertTrue(decision["threshold_selected_on_separate_role"])
        self.assertTrue(decision["worst_family_competence_gate_passed"])

        competence["worst_family"] = {
            "accuracy_gain_vs_prior": 0.0,
            "soft_ce_improvement_vs_prior": 0.10,
        }
        failed = _decision(competence, threshold, conditions)
        self.assertFalse(failed["passed"])


class ResidualAttackContractTests(unittest.TestCase):
    def _attack(self, **overrides: object) -> AttackConfig:
        values: dict[str, object] = {
            "epsilon": 8 / 255,
            "step_size": 2 / 255,
            "grid_size": 4,
            "max_queries": 50,
            "rollback_on_non_improvement": True,
            "action_history_features": True,
            "image_patch_features": True,
            "image_patch_feature_mode": "statistics",
        }
        return AttackConfig(**{**values, **overrides})

    def test_locked_attack_contract_accepts_only_the_preregistered_operator(
        self,
    ) -> None:
        validate_d1_attack_contract(
            self._attack(),
            D1_SOURCE_FAMILIES,
        )

        for field, value in (
            ("epsilon", 4 / 255),
            ("step_size", 1 / 255),
            ("max_queries", 49),
            ("grid_size", 5),
            ("rollback_on_non_improvement", False),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "D1|attack|operator"):
                    validate_d1_attack_contract(
                        self._attack(**{field: value}),
                        D1_SOURCE_FAMILIES,
                    )


if __name__ == "__main__":
    unittest.main()
