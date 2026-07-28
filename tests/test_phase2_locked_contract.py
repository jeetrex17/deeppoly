import unittest
from dataclasses import replace
from pathlib import Path

from rl_transfer.cifar_config import MacPilotConfig
from rl_transfer.phase2_config import Phase2ScreenConfig
from rl_transfer.phase2_screen import validate_phase2_base_contract


SCREEN_PATH = Path(
    "configs/rl_transfer/cifar10_rtx_phase2_screen.json"
)
BASE_PATH = Path(
    "configs/rl_transfer/cifar10_rtx_phase2_base.json"
)


class Phase2LockedContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.screen = Phase2ScreenConfig.from_json(SCREEN_PATH)
        self.base = MacPilotConfig.from_json(BASE_PATH)

    def test_contract_rejects_expansive_or_target_capable_cells(self) -> None:
        invalid = (
            replace(self.base, train_ablation_policies=True),
            replace(self.base, policy_episodes=2001),
            replace(self.base, source_evaluation_images=200),
            replace(self.base, device="cpu"),
            replace(
                self.base,
                source_holdout_instances_per_family=0,
            ),
            replace(self.base, policy_evaluation_temperature=0.5),
        )
        for base in invalid:
            with self.subTest(base=base), self.assertRaises(ValueError):
                validate_phase2_base_contract(self.screen, base)

    def test_every_preregistered_base_deviation_is_rejected(self) -> None:
        deviations = {
            "victim_train_images": 39990,
            "victim_epochs": 49,
            "victim_learning_rate": 0.002,
            "source_instances_per_family": 1,
            "source_holdout_instances_per_family": 2,
            "target_instances_per_family": 2,
            "reward_mode": "legacy",
            "margin_reward_scale": 4.0,
            "terminal_success_bonus": 1.0,
            "query_penalty": 0.02,
            "query_budget": 49,
            "epsilon": 4 / 255,
            "step_size": 1 / 255,
            "behavior_cloning_episodes": 599,
            "behavior_cloning_validation_episodes": 90,
            "behavior_cloning_epochs": 24,
            "behavior_cloning_batch_size": 256,
            "behavior_cloning_soft_temperature": 0.75,
            "policy_episodes": 599,
            "policy_update_block": 40,
            "policy_learning_rate": 0.0002,
            "policy_entropy_weight": 0.0002,
            "policy_update_epochs": 5,
            "source_evaluation_images": 90,
            "policy_actor_mode": "flat",
            "image_patch_feature_mode": "means",
            "policy_evaluation_temperature": 0.75,
            "train_ablation_policies": True,
        }
        for field, value in deviations.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_phase2_base_contract(
                    self.screen,
                    replace(self.base, **{field: value}),
                )

    def test_every_preregistered_screen_deviation_is_rejected(self) -> None:
        deviations = {
            "seeds": (29,),
            "split_seed": 20260726,
            "victim_seed": 999999,
            "max_wall_clock_minutes": 61,
            "estimated_minutes_per_cell": 13,
            "minimum_mean_bc_accuracy_gain": 0.02,
            "minimum_mean_bc_nll_improvement": 0.03,
            "minimum_mean_score_asr_gain": 0.02,
            "minimum_mean_score_auc_gain": 0.01,
            "minimum_positive_condition_fraction": 0.75,
        }
        for field, value in deviations.items():
            varied = replace(self.screen, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_phase2_base_contract(varied, self.base)


if __name__ == "__main__":
    unittest.main()
