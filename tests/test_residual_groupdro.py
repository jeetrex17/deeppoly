from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import unittest
from unittest.mock import patch

import torch

from rl_transfer.imitation import BehaviorCloneStep
from rl_transfer.recurrent import RecurrentAttackPolicy
from rl_transfer.residual_groupdro import (
    GroupDROAudit,
    GroupDROState,
    fit_groupdro_residual_ranker_bc,
    reduce_groupdro_family_losses,
)


SOURCE_FAMILIES = ("classical_cnn", "transformer")


def _trajectory(
    family: str,
    suffix: str,
    distribution: tuple[float, float],
    *,
    accepted_steps: int = 1,
) -> tuple[BehaviorCloneStep, ...]:
    action = int(distribution[1] > distribution[0])
    trajectory_id = f"bc-gradient-source:{family}:{suffix}"
    return tuple(
        BehaviorCloneStep(
            (float(index), float(index + 1)),
            action,
            True,
            trajectory_id=trajectory_id,
            step_index=index,
            action_distribution=distribution,
        )
        for index in range(accepted_steps)
    )


class GroupDROStateTests(unittest.TestCase):
    def test_uniform_state_is_validated_normalized_and_immutable(self) -> None:
        state = GroupDROState.uniform(SOURCE_FAMILIES)

        self.assertEqual(state.families, SOURCE_FAMILIES)
        self.assertEqual(state.weights, (0.5, 0.5))
        self.assertEqual(state.step, 0)
        with self.assertRaises(FrozenInstanceError):
            state.step = 1  # type: ignore[misc]

    def test_state_rejects_invalid_family_and_weight_contracts(self) -> None:
        invalid = (
            (("", "transformer"), (0.5, 0.5)),
            (("classical_cnn", "classical_cnn"), (0.5, 0.5)),
            (SOURCE_FAMILIES, (1.0,)),
            (SOURCE_FAMILIES, (0.0, 1.0)),
            (SOURCE_FAMILIES, (math.nan, 0.5)),
            (SOURCE_FAMILIES, (0.4, 0.5)),
        )
        for families, weights in invalid:
            with self.subTest(families=families, weights=weights):
                with self.assertRaises((TypeError, ValueError)):
                    GroupDROState(families=families, weights=weights)


class GroupDROReductionTests(unittest.TestCase):
    def test_reduction_averages_trajectories_before_family_reweighting(self) -> None:
        state = GroupDROState.uniform(SOURCE_FAMILIES)
        classical_first = torch.tensor(1.0, requires_grad=True)
        classical_second = torch.tensor(3.0, requires_grad=True)
        transformer = torch.tensor(4.0, requires_grad=True)

        objective, next_state, audit = reduce_groupdro_family_losses(
            {
                "classical_cnn": (classical_first, classical_second),
                "transformer": (transformer,),
            },
            state,
            eta=math.log(2.0),
            required_source_families=SOURCE_FAMILIES,
        )

        expected_transformer_weight = 4.0 / 5.0
        self.assertAlmostEqual(next_state.weights[1], expected_transformer_weight)
        self.assertAlmostEqual(
            float(objective.detach()),
            (1.0 - expected_transformer_weight) * 2.0
            + expected_transformer_weight * 4.0,
        )
        objective.backward()
        self.assertAlmostEqual(float(classical_first.grad), 0.1)
        self.assertAlmostEqual(float(classical_second.grad), 0.1)
        self.assertAlmostEqual(float(transformer.grad), 0.8)
        self.assertIsInstance(audit, GroupDROAudit)
        self.assertEqual(
            tuple(item.trajectory_count for item in audit.families),
            (2, 1),
        )
        with self.assertRaises(FrozenInstanceError):
            audit.step = 99  # type: ignore[misc]

    def test_reduction_uses_stable_detached_exponentiated_weight_update(self) -> None:
        state = GroupDROState.uniform(SOURCE_FAMILIES)
        classical = torch.tensor(100_000.0, requires_grad=True)
        transformer = torch.tensor(99_999.0, requires_grad=True)

        objective, next_state, _ = reduce_groupdro_family_losses(
            {
                "classical_cnn": (classical,),
                "transformer": (transformer,),
            },
            state,
            eta=1_000.0,
            required_source_families=SOURCE_FAMILIES,
        )

        self.assertTrue(math.isfinite(float(objective.detach())))
        self.assertTrue(all(math.isfinite(weight) for weight in next_state.weights))
        self.assertAlmostEqual(sum(next_state.weights), 1.0)
        self.assertFalse(objective.isnan().item())
        objective.backward()
        self.assertIsNotNone(classical.grad)
        self.assertIsNotNone(transformer.grad)

    def test_reduction_fails_closed_for_missing_or_nonfinite_locked_family(
        self,
    ) -> None:
        state = GroupDROState.uniform(SOURCE_FAMILIES)
        invalid_losses = (
            {"classical_cnn": (torch.tensor(1.0),)},
            {
                "classical_cnn": (torch.tensor(1.0),),
                "transformer": (),
            },
            {
                "classical_cnn": (torch.tensor(1.0),),
                "transformer": (torch.tensor(float("nan")),),
            },
        )
        for losses in invalid_losses:
            with self.subTest(losses=losses):
                with self.assertRaises((TypeError, ValueError)):
                    reduce_groupdro_family_losses(
                        losses,
                        state,
                        eta=0.1,
                        required_source_families=SOURCE_FAMILIES,
                    )


class GroupDROResidualBehaviorCloningTests(unittest.TestCase):
    def test_fit_requires_every_locked_family_and_returns_immutable_audits(
        self,
    ) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=2,
            hidden_dim=4,
            seed=7,
        )

        with self.assertRaisesRegex(ValueError, "locked source family"):
            fit_groupdro_residual_ranker_bc(
                backbone,
                _trajectory("classical_cnn", "only", (0.9, 0.1)),
                epochs=1,
                seed=11,
                prior_seed=13,
                required_source_families=SOURCE_FAMILIES,
            )

        result = fit_groupdro_residual_ranker_bc(
            backbone,
            (
                *_trajectory(
                    "classical_cnn",
                    "long",
                    (0.9, 0.1),
                    accepted_steps=2,
                ),
                *_trajectory("classical_cnn", "short", (0.8, 0.2)),
                *_trajectory("transformer", "one", (0.2, 0.8)),
            ),
            epochs=2,
            seed=17,
            prior_seed=19,
            required_source_families=SOURCE_FAMILIES,
            groupdro_eta=0.2,
        )

        self.assertEqual(result["aggregation"], "equal_trajectory_then_groupdro")
        self.assertEqual(result["groupdro_state"].step, 2)
        self.assertEqual(len(result["groupdro_audits"]), 2)
        self.assertTrue(
            all(
                isinstance(audit, GroupDROAudit)
                for audit in result["groupdro_audits"]
            )
        )
        self.assertEqual(
            result["source_family_diagnostics"]["classical_cnn"]["trajectories"],
            2,
        )

    def test_fit_checks_deadline_immediately_before_optimizer_step(self) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=2,
            hidden_dim=4,
            seed=23,
        )
        examples = (
            *_trajectory("classical_cnn", "one", (0.9, 0.1)),
            *_trajectory("transformer", "one", (0.1, 0.9)),
        )
        events: list[str] = []
        optimizer = backbone.optimizer
        original_step = optimizer.step

        def deadline_check() -> None:
            events.append("deadline")

        def record_step(*args: object, **kwargs: object) -> object:
            events.append("optimizer.step")
            return original_step(*args, **kwargs)

        with patch.object(optimizer, "step", side_effect=record_step):
            fit_groupdro_residual_ranker_bc(
                backbone,
                examples,
                epochs=1,
                seed=29,
                prior_seed=31,
                required_source_families=SOURCE_FAMILIES,
                deadline_check=deadline_check,
            )

        optimizer_index = events.index("optimizer.step")
        self.assertEqual(events[optimizer_index - 1], "deadline")


if __name__ == "__main__":
    unittest.main()
