from __future__ import annotations

import unittest

from rl_transfer.imitation import BehaviorCloneStep
from rl_transfer.recurrent import RecurrentAttackPolicy
from rl_transfer.residual_bc import fit_residual_ranker_bc


class ResidualBehaviorCloningDeadlineTests(unittest.TestCase):
    def test_one_epoch_checks_deadline_after_optimizer_mutation(self) -> None:
        backbone = RecurrentAttackPolicy(
            observation_dim=2,
            action_dim=2,
            hidden_dim=4,
            seed=43,
        )
        examples = (
            BehaviorCloneStep(
                (0.0, 0.0),
                0,
                True,
                trajectory_id="synthetic-deadline-boundary",
                step_index=0,
                action_distribution=(0.9, 0.1),
            ),
        )
        checks = 0

        def deadline() -> None:
            nonlocal checks
            checks += 1

        fit_residual_ranker_bc(
            backbone,
            examples,
            epochs=1,
            seed=47,
            prior_seed=53,
            deadline_check=deadline,
        )

        self.assertEqual(checks, 6)


if __name__ == "__main__":
    unittest.main()
