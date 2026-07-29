from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from rl_transfer.config import AttackConfig
from rl_transfer.population import balanced_family_schedule
from rl_transfer.recurrent import PPOConfig, RecurrentAttackPolicy
from rl_transfer.residual_ppo import (
    RESIDUAL_PPO_BLOCK_EPISODES,
    _ResidualPPOSequence,
    _ppo_update_combined,
    train_residual_ranker_ppo,
)
from rl_transfer.residual_ranker import (
    ResidualRankerPolicy,
    score_greedy_action_order,
)


SOURCE_FAMILIES = ("classical_cnn", "transformer")


class AnyChangeSourceVictim(nn.Module):
    """Classify the clean fixture as zero and any perturbation as one."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += images.shape[0]
        distance = (images - 0.5).abs().flatten(1).sum(dim=1)
        changed_logit = distance * 100.0
        return torch.stack((1.0 - changed_logit, changed_logit), dim=1)


class ConstantSourceVictim(nn.Module):
    """Keep the synthetic source example correctly classified."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += images.shape[0]
        return torch.tensor(
            ((2.0, 0.0),),
            dtype=images.dtype,
            device=images.device,
        ).expand(images.shape[0], -1)


class RecordingResidualPolicy(ResidualRankerPolicy):
    def __init__(self, backbone: RecurrentAttackPolicy) -> None:
        super().__init__(
            backbone,
            confidence_threshold=0.0,
            prior_temperature=4.0,
        )
        self.proposal_indices: list[int] = []

    def combined_logits(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        *,
        prior_order: tuple[int, ...],
        proposal_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.proposal_indices.append(proposal_index)
        return super().combined_logits(
            observation,
            hidden,
            prior_order=prior_order,
            proposal_index=proposal_index,
        )


class ResidualPPOTests(unittest.TestCase):
    def _attack(self, **overrides: object) -> AttackConfig:
        values: dict[str, object] = {
            "grid_size": 1,
            "max_queries": 50,
            "rollback_on_non_improvement": True,
            "reward_mode": "margin_delta",
        }
        return AttackConfig(**{**values, **overrides})

    def _policy(
        self,
        *,
        seed: int = 7,
        update_epochs: int = 1,
    ) -> RecordingResidualPolicy:
        backbone = RecurrentAttackPolicy(
            observation_dim=8,
            action_dim=6,
            hidden_dim=4,
            seed=seed,
            config=PPOConfig(
                learning_rate=1e-3,
                clip_ratio=0.11,
                value_weight=0.3,
                entropy_weight=0.02,
                gradient_clip_norm=0.4,
                update_epochs=update_epochs,
            ),
        )
        return RecordingResidualPolicy(backbone)

    @staticmethod
    def _samples(count: int = 4) -> tuple[tuple[torch.Tensor, int], ...]:
        return tuple((torch.full((3, 1, 1), 0.5), 0) for _ in range(count))

    def test_refinement_uses_combined_logits_and_source_only_audit_metrics(
        self,
    ) -> None:
        policy = self._policy(update_epochs=2)
        constant = ConstantSourceVictim()
        changed = AnyChangeSourceVictim()
        victims = {
            "classical_cnn": (("classical-source", constant),),
            "transformer": (("transformer-source", changed),),
        }
        deadline_calls: list[int] = []
        before = policy.persistent_digest()

        with patch(
            "rl_transfer.residual_ppo.score_greedy_action_order",
            wraps=score_greedy_action_order,
        ) as prior_order:
            metrics = train_residual_ranker_ppo(
                policy,
                victims,
                self._samples(),
                self._attack(),
                episodes=2,
                seed=13,
                prior_seed=29,
                deadline_check=lambda: deadline_calls.append(1),
            )

        self.assertNotEqual(before, policy.persistent_digest())
        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["trained_episodes"], 2)
        self.assertEqual(metrics["eligible_episodes"], 2)
        self.assertEqual(metrics["successful_episodes"], 1)
        self.assertEqual(metrics["source_calls"], 52)
        self.assertEqual(
            metrics["source_calls_by_family"],
            {"classical_cnn": 50, "transformer": 2},
        )
        self.assertEqual(
            metrics["source_calls_by_victim"],
            {"classical-source": 50, "transformer-source": 2},
        )
        self.assertEqual(metrics["hidden_target_calls"], 0)
        self.assertEqual(set(metrics["schedule"]), set(SOURCE_FAMILIES))
        self.assertEqual(prior_order.call_count, 2)
        self.assertEqual(
            len({call.kwargs["sample_id"] for call in prior_order.call_args_list}),
            2,
        )
        self.assertGreaterEqual(len(deadline_calls), 2)
        self.assertEqual(metrics["success"]["count"], 2)
        self.assertEqual(metrics["margin_reduction"]["count"], 2)
        self.assertEqual(metrics["episode_return"]["count"], 2)
        self.assertEqual(metrics["ppo"]["update_epochs"], 2)
        self.assertEqual(metrics["ppo"]["clip_ratio"], 0.11)
        self.assertEqual(metrics["ppo"]["value_weight"], 0.3)
        self.assertEqual(metrics["ppo"]["entropy_weight"], 0.02)
        self.assertEqual(
            metrics["ppo"]["objective"],
            "clipped_prior_plus_residual_actor_critic",
        )
        self.assertTrue(set(range(49)).issubset(policy.proposal_indices))
        self.assertEqual(constant.calls, 50)
        self.assertEqual(changed.calls, 2)

    def test_resumable_blocks_continue_schedule_samples_and_instances(
        self,
    ) -> None:
        seed = 19
        policy = self._policy(seed=11)
        victims = {
            family: (
                (f"{family}-source-0", AnyChangeSourceVictim()),
                (f"{family}-source-1", AnyChangeSourceVictim()),
            )
            for family in SOURCE_FAMILIES
        }

        first = train_residual_ranker_ppo(
            policy,
            victims,
            self._samples(),
            self._attack(),
            episodes=2,
            seed=seed,
            prior_seed=31,
        )
        second = train_residual_ranker_ppo(
            policy,
            victims,
            self._samples(),
            self._attack(),
            episodes=2,
            seed=seed,
            prior_seed=31,
            episode_offset=2,
            initial_family_weights=first["family_weights"],
            initial_instance_offsets=first["instance_offsets"],
        )

        full_schedule = balanced_family_schedule(SOURCE_FAMILIES, 4, seed)
        self.assertEqual(first["schedule"], full_schedule[:2])
        self.assertEqual(second["schedule"], full_schedule[2:])
        self.assertEqual(first["sample_indices"], [0, 1])
        self.assertEqual(second["sample_indices"], [2, 3])
        self.assertEqual(first["instance_offsets"], dict.fromkeys(SOURCE_FAMILIES, 1))
        self.assertEqual(second["instance_offsets"], dict.fromkeys(SOURCE_FAMILIES, 2))
        for family in SOURCE_FAMILIES:
            self.assertEqual(
                first["source_calls_by_victim"][f"{family}-source-0"],
                2,
            )
            self.assertEqual(
                first["source_calls_by_victim"][f"{family}-source-1"],
                0,
            )
            self.assertEqual(
                second["source_calls_by_victim"][f"{family}-source-0"],
                0,
            )
            self.assertEqual(
                second["source_calls_by_victim"][f"{family}-source-1"],
                2,
            )
        self.assertAlmostEqual(sum(second["family_weights"].values()), 1.0)
        self.assertEqual(second["hidden_target_calls"], 0)

    def test_locked_operator_source_families_and_block_bounds_fail_closed(
        self,
    ) -> None:
        policy = self._policy()
        valid_victims = {
            family: ((f"{family}-source", AnyChangeSourceVictim()),)
            for family in SOURCE_FAMILIES
        }
        invalid_calls = (
            {
                "source_victims": valid_victims,
                "config": self._attack(max_queries=49),
                "episodes": 1,
            },
            {
                "source_victims": valid_victims,
                "config": self._attack(rollback_on_non_improvement=False),
                "episodes": 1,
            },
            {
                "source_victims": valid_victims,
                "config": self._attack(),
                "episodes": RESIDUAL_PPO_BLOCK_EPISODES + 1,
            },
            {
                "source_victims": {
                    "classical_cnn": valid_victims["classical_cnn"],
                    "modern_cnn": (("heldout", AnyChangeSourceVictim()),),
                },
                "config": self._attack(),
                "episodes": 1,
            },
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    ValueError,
                    "50|rollback|block|source|family|held.?out",
                ):
                    train_residual_ranker_ppo(
                        policy,
                        call["source_victims"],
                        self._samples(),
                        call["config"],
                        episodes=call["episodes"],
                        seed=13,
                        prior_seed=29,
                    )

        with self.assertRaisesRegex(ValueError, "200|bound|episode"):
            train_residual_ranker_ppo(
                policy,
                valid_victims,
                self._samples(),
                self._attack(),
                episodes=2,
                seed=13,
                prior_seed=29,
                episode_offset=199,
            )

    def test_deadline_is_checked_before_each_episode_queries_a_source(
        self,
    ) -> None:
        policy = self._policy()
        victims = {
            family: ((f"{family}-source", AnyChangeSourceVictim()),)
            for family in SOURCE_FAMILIES
        }

        with self.assertRaisesRegex(TimeoutError, "fixture deadline"):
            train_residual_ranker_ppo(
                policy,
                victims,
                self._samples(),
                self._attack(),
                episodes=2,
                seed=13,
                prior_seed=29,
                deadline_check=lambda: (_ for _ in ()).throw(
                    TimeoutError("fixture deadline")
                ),
            )

        self.assertEqual(
            sum(
                victim.calls
                for family_victims in victims.values()
                for _, victim in family_victims
            ),
            0,
        )

    def test_mid_episode_deadline_stops_queries_before_any_ppo_update(
        self,
    ) -> None:
        policy = self._policy()
        victims = {
            family: ((f"{family}-source", ConstantSourceVictim()),)
            for family in SOURCE_FAMILIES
        }
        deadline_calls = 0
        before = policy.persistent_digest()

        def deadline_check() -> None:
            nonlocal deadline_calls
            deadline_calls += 1
            if deadline_calls == 3:
                raise TimeoutError("mid-episode deadline")

        with self.assertRaisesRegex(TimeoutError, "mid-episode deadline"):
            train_residual_ranker_ppo(
                policy,
                victims,
                self._samples(),
                self._attack(),
                episodes=2,
                seed=13,
                prior_seed=29,
                deadline_check=deadline_check,
            )

        self.assertEqual(
            sum(
                victim.calls
                for family_victims in victims.values()
                for _, victim in family_victims
            ),
            2,
        )
        self.assertEqual(policy.persistent_digest(), before)

    def test_deadline_brackets_optimizer_step_and_pre_step_failure_blocks_it(
        self,
    ) -> None:
        def sequence(policy: ResidualRankerPolicy) -> _ResidualPPOSequence:
            return _ResidualPPOSequence(
                observations=torch.zeros((2, policy.backbone.observation_dim)),
                actions=torch.tensor((0, 1)),
                old_log_probabilities=torch.zeros(2),
                advantages=torch.tensor((1.0, -1.0)),
                returns=torch.tensor((1.0, 0.0)),
                prior_order=tuple(range(policy.action_dim)),
            )

        policy = self._policy(update_epochs=1)
        optimizer = policy.backbone.optimizer
        normal_events: list[str] = []
        original_zero_grad = optimizer.zero_grad
        original_step = optimizer.step

        def record_zero_grad(*args: object, **kwargs: object) -> None:
            normal_events.append("zero_grad")
            original_zero_grad(*args, **kwargs)

        def record_step(*args: object, **kwargs: object) -> object:
            normal_events.append("optimizer.step")
            return original_step(*args, **kwargs)

        with (
            patch.object(optimizer, "zero_grad", side_effect=record_zero_grad),
            patch.object(optimizer, "step", side_effect=record_step),
        ):
            _ppo_update_combined(
                policy,
                ((sequence(policy), 1.0),),
                deadline_check=lambda: normal_events.append("deadline"),
            )

        self.assertEqual(
            normal_events,
            [
                "deadline",
                "zero_grad",
                "deadline",
                "optimizer.step",
                "deadline",
            ],
        )

        blocked_policy = self._policy(update_epochs=1)
        blocked_optimizer = blocked_policy.backbone.optimizer
        blocked_events: list[str] = []
        before = blocked_policy.persistent_digest()
        blocked_zero_grad = blocked_optimizer.zero_grad

        def record_blocked_zero_grad(*args: object, **kwargs: object) -> None:
            blocked_events.append("zero_grad")
            blocked_zero_grad(*args, **kwargs)

        def fail_at_pre_step_boundary() -> None:
            blocked_events.append("deadline")
            if blocked_events[-2:] == ["zero_grad", "deadline"]:
                raise TimeoutError("pre-step deadline")

        with (
            patch.object(
                blocked_optimizer,
                "zero_grad",
                side_effect=record_blocked_zero_grad,
            ),
            patch.object(
                blocked_optimizer,
                "step",
                wraps=blocked_optimizer.step,
            ) as blocked_step,
        ):
            with self.assertRaisesRegex(TimeoutError, "pre-step deadline"):
                _ppo_update_combined(
                    blocked_policy,
                    ((sequence(blocked_policy), 1.0),),
                    deadline_check=fail_at_pre_step_boundary,
                )

        blocked_step.assert_not_called()
        self.assertEqual(
            blocked_events,
            ["deadline", "zero_grad", "deadline"],
        )
        self.assertEqual(blocked_policy.persistent_digest(), before)


if __name__ == "__main__":
    unittest.main()
