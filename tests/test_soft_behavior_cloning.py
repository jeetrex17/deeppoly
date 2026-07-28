import math
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from rl_transfer.cifar_config import MacPilotConfig
from rl_transfer.cifar_policy_training import (
    SOFT_GRADIENT_BC_ACTION_CONDITIONED_PPO_METHOD,
    _collect_and_fit_bc,
    main_policy_method_id,
)
from rl_transfer.config import AttackConfig
from rl_transfer.imitation import (
    BehaviorCloneStep,
    behavior_clone_policy,
    collect_gradient_demonstrations,
    evaluate_behavior_clone_policy,
)
from rl_transfer.recurrent import RecurrentAttackPolicy


class BehaviorCloneDistributionTests(unittest.TestCase):
    def test_distribution_is_validated_and_copied_immutably(self) -> None:
        probabilities = [0.1, 0.7, 0.2]
        step = BehaviorCloneStep(
            (0.0, 1.0),
            action=1,
            accepted=True,
            action_distribution=probabilities,
        )
        probabilities[1] = 0.0

        self.assertEqual(step.action_distribution, (0.1, 0.7, 0.2))
        self.assertIsInstance(step.action_distribution, tuple)

        invalid = (
            (),
            (0.5, 0.5),
            (0.2, -0.1, 0.9),
            (0.2, float("nan"), 0.8),
            (0.2, 0.2, 0.2),
            (0.8, 0.1, 0.1),
        )
        for distribution in invalid:
            with self.subTest(distribution=distribution):
                with self.assertRaises(ValueError):
                    BehaviorCloneStep(
                        (0.0, 1.0),
                        action=2,
                        accepted=True,
                        action_distribution=distribution,
                    )

    def test_soft_behavior_cloning_is_deterministic_and_improves_soft_ce(self) -> None:
        observation = np.asarray((0.2, -0.1, 0.4, 0.3), dtype=np.float32)
        distribution = (0.03, 0.92, 0.05)
        steps = tuple(
            BehaviorCloneStep(
                observation,
                action=1,
                accepted=True,
                trajectory_id=f"trajectory-{index}",
                action_distribution=distribution,
            )
            for index in range(12)
        )
        first = RecurrentAttackPolicy(4, 3, hidden_dim=8, seed=17)
        second = RecurrentAttackPolicy(4, 3, hidden_dim=8, seed=17)
        before = evaluate_behavior_clone_policy(first, steps)

        first_fit = behavior_clone_policy(first, steps, epochs=25, seed=23, batch_size=4)
        second_fit = behavior_clone_policy(second, steps, epochs=25, seed=23, batch_size=4)
        after = evaluate_behavior_clone_policy(first, steps)

        self.assertLess(after["soft_cross_entropy"], before["soft_cross_entropy"])
        self.assertLess(after["soft_kl"], before["soft_kl"])
        self.assertEqual(first.persistent_digest(), second.persistent_digest())
        self.assertEqual(first_fit, second_fit)
        self.assertEqual(after["top1_accuracy"], after["accuracy"])
        self.assertEqual(after["top5_accuracy"], 1.0)
        self.assertGreaterEqual(after["teacher_probability_regret"], 0.0)
        for key in (
            "soft_cross_entropy",
            "soft_kl",
            "teacher_entropy",
            "teacher_probability_regret",
            "uniform_soft_cross_entropy",
            "uniform_soft_kl",
            "validation_oracle_soft_cross_entropy",
            "validation_oracle_soft_kl",
            "uniform_top1_accuracy",
            "uniform_top5_accuracy",
            "validation_oracle_top1_accuracy",
            "validation_oracle_top5_accuracy",
        ):
            self.assertTrue(math.isfinite(float(after[key])), key)
        self.assertEqual(
            after["baseline_provenance"],
            "evaluated_labels_validation_oracle",
        )
        self.assertEqual(
            after["baseline_estimator"],
            "empirical_best_constant_no_smoothing",
        )
        self.assertTrue(after["deprecated_frequency_aliases_present"])
        self.assertEqual(
            after["deprecated_frequency_alias_semantics"],
            "validation_oracle_estimated_from_evaluated_labels",
        )
        self.assertNotIn("frequency_soft_cross_entropy", after)
        self.assertAlmostEqual(
            after["soft_cross_entropy"] - after["teacher_entropy"],
            after["soft_kl"],
            places=6,
        )
        self.assertAlmostEqual(
            after["validation_oracle_soft_cross_entropy"],
            after["teacher_entropy"],
            places=6,
        )
        self.assertAlmostEqual(
            after["uniform_soft_cross_entropy"],
            math.log(3),
            places=6,
        )
        self.assertAlmostEqual(after["uniform_top1_accuracy"], 1 / 3)
        self.assertEqual(after["uniform_top5_accuracy"], 1.0)

    def test_hard_label_path_remains_supported(self) -> None:
        policy = RecurrentAttackPolicy(3, 2, hidden_dim=4, seed=31)
        steps = (
            BehaviorCloneStep((0.0, 0.1, 0.2), 1, True),
            BehaviorCloneStep((0.2, 0.1, 0.0), 0, True),
            BehaviorCloneStep((0.1, 0.2, 0.0), 0, True),
        )

        fit = behavior_clone_policy(policy, steps, epochs=2, seed=37)
        evaluation = evaluate_behavior_clone_policy(policy, steps)

        self.assertIn("final_accuracy", fit)
        self.assertIn("accuracy", evaluation)
        self.assertEqual(
            set(fit),
            {
                "training_mode",
                "trajectories",
                "accepted_steps",
                "rejected_steps",
                "epochs",
                "uniform_accuracy",
                "uniform_nll",
                "final_loss",
                "final_accuracy",
                "history",
                "majority_accuracy",
                "frequency_nll",
                "training_empirical_top1_accuracy",
                "training_empirical_nll",
                "baseline_provenance",
                "baseline_estimator",
                "deprecated_frequency_aliases_present",
                "deprecated_frequency_alias_semantics",
            },
        )
        self.assertEqual(
            set(evaluation),
            {
                "training_mode",
                "trajectories",
                "accepted_steps",
                "nll",
                "accuracy",
                "uniform_nll",
                "uniform_accuracy",
                "majority_accuracy",
                "frequency_nll",
                "validation_oracle_top1_accuracy",
                "validation_oracle_nll",
                "baseline_provenance",
                "baseline_estimator",
                "deprecated_frequency_aliases_present",
                "deprecated_frequency_alias_semantics",
            },
        )
        self.assertEqual(
            set(fit["history"][0]),
            {"epoch", "loss", "accuracy"},
        )
        self.assertNotIn("final_top1_accuracy", fit)
        self.assertNotIn("top1_accuracy", evaluation)
        self.assertNotIn("final_soft_cross_entropy", fit)
        self.assertNotIn("soft_cross_entropy", evaluation)
        self.assertEqual(
            fit["baseline_provenance"],
            "training_labels_empirical_constant",
        )
        self.assertEqual(
            evaluation["baseline_provenance"],
            "evaluated_labels_validation_oracle",
        )
        self.assertEqual(
            evaluation["validation_oracle_nll"],
            evaluation["frequency_nll"],
        )
        expected_oracle_nll = -(
            (2 / 3) * math.log(2 / 3)
            + (1 / 3) * math.log(1 / 3)
        )
        self.assertAlmostEqual(
            evaluation["validation_oracle_nll"],
            expected_oracle_nll,
            places=7,
        )


class Phase2MethodIdentityTests(unittest.TestCase):
    def test_no_ablation_soft_action_conditioned_policy_has_stable_id(
        self,
    ) -> None:
        config = MacPilotConfig.from_json(
            Path("configs/rl_transfer/cifar10_rtx_phase2_base.json")
        )

        self.assertFalse(config.train_ablation_policies)
        self.assertEqual(
            main_policy_method_id(config),
            SOFT_GRADIENT_BC_ACTION_CONDITIONED_PPO_METHOD,
        )
        self.assertEqual(
            main_policy_method_id(
                replace(config, train_ablation_policies=True)
            ),
            SOFT_GRADIENT_BC_ACTION_CONDITIONED_PPO_METHOD,
        )
        self.assertEqual(
            main_policy_method_id(
                replace(
                    config,
                    behavior_cloning_soft_temperature=None,
                )
            ),
            "groupdro_recurrent_ppo",
        )

    def test_soft_bc_gate_uses_named_validation_oracle_not_deprecated_aliases(
        self,
    ) -> None:
        config = MacPilotConfig.from_json(
            Path("configs/rl_transfer/cifar10_rtx_phase2_base.json")
        )
        validation = {
            "target_mode": "soft",
            "baseline_provenance": "evaluated_labels_validation_oracle",
            "baseline_estimator": "empirical_best_constant_no_smoothing",
            "soft_cross_entropy": 4.0,
            "uniform_soft_cross_entropy": 4.10,
            "validation_oracle_soft_cross_entropy": 4.03,
            "top5_accuracy": 0.20,
            "validation_oracle_top5_accuracy": 0.17,
            "frequency_soft_cross_entropy": 0.0,
            "frequency_top5_accuracy": 1.0,
        }
        policy = RecurrentAttackPolicy(2, 2, hidden_dim=4, seed=47)

        with (
            mock.patch(
                "rl_transfer.cifar_policy_training."
                "collect_gradient_demonstrations",
                return_value=((), {}),
            ),
            mock.patch(
                "rl_transfer.cifar_policy_training.behavior_clone_policy",
                return_value={},
            ),
            mock.patch(
                "rl_transfer.cifar_policy_training."
                "evaluate_behavior_clone_policy",
                return_value=validation,
            ),
        ):
            diagnostics = _collect_and_fit_bc(
                policy,
                {},
                (),
                (),
                config.attack_config(),
                config,
            )

        self.assertTrue(diagnostics["gate"]["passed"])
        self.assertEqual(
            diagnostics["gate"]["baseline"],
            "evaluated_labels_validation_oracle",
        )
        self.assertNotIn(
            "minimum_top5_gain_over_frequency",
            diagnostics["gate"],
        )


class SoftGradientTeacherTests(unittest.TestCase):
    class MeanVictim(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            means = images.mean(dim=(1, 2, 3))
            return torch.stack((means, 1 - means), dim=1)

    class ScaledMeanVictim(nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.scale = scale

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            means = images.mean(dim=(1, 2, 3))
            return self.scale * torch.stack((means, 1 - means), dim=1)

    class FlatVictim(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            means = images.mean(dim=(1, 2, 3))
            return torch.stack((1.0 + 0.0 * means, 0.0 * means), dim=1)

    @staticmethod
    def _config() -> AttackConfig:
        return AttackConfig(
            epsilon=0.1,
            step_size=0.05,
            grid_size=1,
            max_queries=3,
            reward_mode="margin_delta",
            rollback_on_non_improvement=True,
            action_history_features=True,
            image_patch_features=True,
        )

    def test_gradient_teacher_emits_stable_normalized_soft_targets(self) -> None:
        image = torch.full((3, 4, 4), 0.7)
        original = image.clone()
        steps, metrics = collect_gradient_demonstrations(
            {"source": (("source-0", self.MeanVictim()),)},
            ((image, 0),),
            self._config(),
            episodes=1,
            decisions=1,
            seed=41,
            soft_temperature=0.01,
        )

        self.assertTrue(torch.equal(image, original))
        self.assertEqual(len(steps), 1)
        distribution = steps[0].action_distribution
        self.assertIsNotNone(distribution)
        assert distribution is not None
        self.assertEqual(len(distribution), self._config().action_dim)
        self.assertAlmostEqual(sum(distribution), 1.0, places=7)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in distribution))
        self.assertEqual(steps[0].action, int(np.argmax(distribution)))
        self.assertEqual(metrics["soft_temperature"], 0.01)
        self.assertGreater(metrics["soft_target_mean_entropy"], 0.0)
        self.assertGreaterEqual(
            metrics["soft_target_mean_expected_linearized_regret"],
            0.0,
        )

        repeated, repeated_metrics = collect_gradient_demonstrations(
            {"source": (("source-0", self.MeanVictim()),)},
            ((image, 0),),
            self._config(),
            episodes=1,
            decisions=1,
            seed=41,
            soft_temperature=0.01,
        )
        self.assertEqual(steps, repeated)
        self.assertEqual(metrics, repeated_metrics)

    def test_soft_targets_are_invariant_to_gradient_cost_scale(self) -> None:
        def collect(scale: float):
            return collect_gradient_demonstrations(
                {"source": (("source-0", self.ScaledMeanVictim(scale)),)},
                ((torch.full((3, 4, 4), 0.7), 0),),
                self._config(),
                episodes=1,
                decisions=1,
                seed=42,
                soft_temperature=0.5,
            )

        unit_steps, unit_metrics = collect(1.0)
        scaled_steps, scaled_metrics = collect(100.0)
        np.testing.assert_allclose(
            unit_steps[0].action_distribution,
            scaled_steps[0].action_distribution,
            rtol=1e-5,
            atol=1e-7,
        )
        self.assertGreater(
            max(unit_steps[0].action_distribution),
            1 / self._config().action_dim,
        )
        self.assertAlmostEqual(
            scaled_metrics["soft_target_mean_cost_scale"]
            / unit_metrics["soft_target_mean_cost_scale"],
            100.0,
            places=3,
        )
        self.assertAlmostEqual(
            unit_metrics[
                "soft_target_mean_expected_normalized_regret"
            ],
            scaled_metrics[
                "soft_target_mean_expected_normalized_regret"
            ],
            places=5,
        )
        self.assertEqual(
            unit_metrics["soft_target_cost_normalization"],
            "per_state_standard_deviation",
        )

    def test_zero_gradient_produces_a_finite_uniform_distribution(self) -> None:
        steps, metrics = collect_gradient_demonstrations(
            {"source": (("source-0", self.FlatVictim()),)},
            ((torch.full((3, 4, 4), 0.7), 0),),
            self._config(),
            episodes=1,
            decisions=1,
            seed=42,
            soft_temperature=0.5,
        )
        distribution = np.asarray(steps[0].action_distribution)
        np.testing.assert_allclose(
            distribution,
            np.full(self._config().action_dim, 1 / self._config().action_dim),
        )
        self.assertTrue(np.isfinite(distribution).all())
        self.assertAlmostEqual(float(distribution.sum()), 1.0)
        self.assertEqual(
            metrics["soft_target_min_cost_scale"],
            0.0,
        )
        self.assertEqual(metrics["soft_target_count"], 0)
        self.assertEqual(metrics["soft_target_generated_count"], 1)
        self.assertEqual(
            metrics["soft_target_metric_scope"],
            "accepted_behavior_clone_steps",
        )

    def test_none_temperature_preserves_hard_teacher_and_invalid_values_fail_closed(
        self,
    ) -> None:
        arguments = (
            {"source": (("source-0", self.MeanVictim()),)},
            ((torch.full((3, 4, 4), 0.7), 0),),
            self._config(),
        )
        hard_steps, hard_metrics = collect_gradient_demonstrations(
            *arguments,
            episodes=1,
            decisions=1,
            seed=43,
        )
        explicit_none_steps, explicit_none_metrics = collect_gradient_demonstrations(
            *arguments,
            episodes=1,
            decisions=1,
            seed=43,
            soft_temperature=None,
        )
        self.assertEqual(hard_steps, explicit_none_steps)
        self.assertEqual(hard_metrics, explicit_none_metrics)
        self.assertIsNone(hard_steps[0].action_distribution)
        self.assertNotIn("soft_temperature", hard_metrics)

        for temperature in (0.0, -1.0, float("inf"), float("nan"), True):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    collect_gradient_demonstrations(
                        {"source": (("source-0", self.MeanVictim()),)},
                        ((torch.full((3, 4, 4), 0.7), 0),),
                        self._config(),
                        episodes=1,
                        decisions=1,
                        seed=43,
                        soft_temperature=temperature,
                    )


if __name__ == "__main__":
    unittest.main()
