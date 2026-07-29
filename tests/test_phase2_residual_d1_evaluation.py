from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rl_transfer.config import AttackConfig
from rl_transfer.phase2_residual_d1_evaluation import (
    evaluate_residual_d1,
    evaluate_residual_policy_cohort,
)
from rl_transfer.recurrent import RecurrentAttackPolicy
from rl_transfer.research_protocol import FrozenEpisodeResult
from rl_transfer.residual_ranker import ResidualRankerPolicy


def _attack(**overrides: object) -> AttackConfig:
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


def _policy(attack: AttackConfig, seed: int) -> ResidualRankerPolicy:
    backbone = RecurrentAttackPolicy(
        attack.recurrent_observation_dim,
        attack.action_dim,
        hidden_dim=4,
        seed=seed,
        actor_mode="action_conditioned",
        action_grid_size=attack.grid_size,
    )
    return ResidualRankerPolicy(
        backbone,
        confidence_threshold=0.0,
        prior_temperature=24.0,
    )


def _event(
    *,
    call_index: int,
    purpose: str,
    sample_id: str,
    victim_id: str,
) -> dict[str, object]:
    return {
        "call_index": call_index,
        "error": None,
        "feedback": "scores",
        "predicted_label": 0,
        "purpose": purpose,
        "sample_id": sample_id,
        "step": call_index - 1,
        "victim_id": victim_id,
    }


def _episode(
    *,
    sample_id: str,
    victim_id: str,
    family: str,
    digest: str,
    purpose: str,
    total_calls: int = 2,
) -> FrozenEpisodeResult:
    events = (
        _event(
            call_index=1,
            purpose="initialization",
            sample_id=sample_id,
            victim_id=victim_id,
        ),
        *(
            _event(
                call_index=call_index,
                purpose=purpose,
                sample_id=sample_id,
                victim_id=victim_id,
            )
            for call_index in range(2, total_calls + 1)
        ),
    )
    return FrozenEpisodeResult(
        sample_id=sample_id,
        victim_id=victim_id,
        family=family,
        clean_correct=True,
        success=False,
        query_to_success=None,
        total_target_calls=total_calls,
        linf=0.0,
        l2=0.0,
        actions=tuple(0 for _ in range(total_calls - 1)),
        policy_digest_before=digest,
        policy_digest_after=digest,
        query_trace=events,
    )


class ResidualD1EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.attack = _attack()
        self.samples = (
            (torch.zeros(3, 4, 4), 0),
            (torch.ones(3, 4, 4), 0),
        )
        self.indices = (7, 11)
        self.victims = (
            ("classical-source-0", nn.Identity()),
            ("classical-source-1", nn.Identity()),
        )
        self.bc = _policy(self.attack, 3)
        self.ppo = _policy(self.attack, 5)

    def _patched_episodes(
        self,
        *,
        over_budget_method: str | None = None,
    ):
        score_calls: list[tuple[str, int]] = []
        residual_calls: dict[str, list[tuple[str, int]]] = defaultdict(list)
        method_by_policy = {
            id(self.bc): "residual_ranker_bc",
            id(self.ppo): "residual_ranker_ppo",
        }

        def score_episode(
            victim,
            image,
            label,
            sample_id,
            victim_id,
            family,
            attack,
            seed,
            *,
            deadline_check,
        ) -> FrozenEpisodeResult:
            del victim, image, label, attack
            deadline_check()
            score_calls.append((sample_id, seed))
            return _episode(
                sample_id=sample_id,
                victim_id=victim_id,
                family=family,
                digest="s" * 64,
                purpose="score-greedy",
            )

        def residual_episode(
            policy,
            victim,
            image,
            label,
            sample_id,
            victim_id,
            family,
            attack,
            *,
            score_prior_seed,
            deadline_check,
        ) -> FrozenEpisodeResult:
            del victim, image, label, attack
            deadline_check()
            method = method_by_policy[id(policy)]
            residual_calls[method].append((sample_id, score_prior_seed))
            return _episode(
                sample_id=sample_id,
                victim_id=victim_id,
                family=family,
                digest=policy.persistent_digest(),
                purpose=(
                    "residual-ranker-learned"
                    if method == "residual_ranker_ppo"
                    else "residual-ranker-fallback"
                ),
                total_calls=(51 if method == over_budget_method else 2),
            )

        return score_calls, residual_calls, score_episode, residual_episode

    def test_three_methods_share_the_exact_cohort_and_full_trace_multiset(
        self,
    ) -> None:
        (
            score_calls,
            residual_calls,
            score_episode,
            residual_episode,
        ) = self._patched_episodes()

        with (
            patch(
                "rl_transfer.phase2_residual_d1_evaluation.run_score_greedy_episode",
                side_effect=score_episode,
            ),
            patch(
                "rl_transfer.phase2_residual_d1_evaluation.run_residual_ranker_episode",
                side_effect=residual_episode,
            ),
        ):
            condition, rows, traces = evaluate_residual_policy_cohort(
                policies={
                    "residual_ranker_bc": self.bc,
                    "residual_ranker_ppo": self.ppo,
                },
                victims=self.victims,
                samples=self.samples,
                indices=self.indices,
                attack=self.attack,
                family="classical_cnn",
                seed=29,
                heldout_family="modern_cnn",
                source_slice="seen_family_new_instance",
                deadline_check=lambda: None,
                progress=lambda _: None,
            )

        expected_cohort_size = len(self.victims) * len(self.samples)
        self.assertEqual(len(score_calls), expected_cohort_size)
        self.assertEqual(
            set(residual_calls),
            {"residual_ranker_bc", "residual_ranker_ppo"},
        )
        self.assertTrue(
            all(
                len(method_calls) == expected_cohort_size
                for method_calls in residual_calls.values()
            )
        )
        self.assertEqual(
            {sample_id for sample_id, _ in score_calls},
            {
                sample_id
                for method_calls in residual_calls.values()
                for sample_id, _ in method_calls
            },
        )
        self.assertEqual(
            {
                seed
                for method_calls in residual_calls.values()
                for _, seed in method_calls
            },
            {29},
        )

        self.assertEqual(
            set(condition["methods"]),
            {
                "score_greedy",
                "residual_ranker_bc",
                "residual_ranker_ppo",
            },
        )
        self.assertEqual(len(rows), expected_cohort_size * 3)
        self.assertEqual(len(traces), len(rows))
        row_identities = Counter(
            (row.method, row.sample_id, row.victim_id, row.victim_family)
            for row in rows
        )
        trace_identities = Counter(
            (
                trace["method"],
                trace["sample_id"],
                trace["victim_id"],
                trace["victim_family"],
            )
            for trace in traces
        )
        self.assertEqual(trace_identities, row_identities)
        self.assertTrue(
            all(
                len(trace["query_trace"]) == trace["total_target_calls"]
                for trace in traces
            )
        )
        self.assertTrue(
            all(
                isinstance(trace["actions"], list)
                and isinstance(trace["query_trace"], list)
                for trace in traces
            )
        )
        self.assertTrue(
            all(
                trace["family"] == trace["victim_family"]
                and trace["source_slice"] == "seen_family_new_instance"
                and trace["heldout_family"] == "modern_cnn"
                and trace["target_calls"] == 0
                and trace["hidden_target_calls"] == 0
                for trace in traces
            )
        )
        audit = condition["audit"]
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["raw_cohort_aligned"])
        self.assertTrue(audit["expected_cohort_verified"])
        self.assertTrue(audit["trace_result_identity_multiset_matched"])
        self.assertEqual(audit["method_count"], 3)
        self.assertEqual(
            len(set(audit["method_cohort_sha256"].values())),
            1,
        )
        self.assertEqual(condition["hidden_target_calls"], 0)
        self.assertGreaterEqual(condition["elapsed_seconds"], 0.0)
        self.assertEqual(
            condition["source_model_calls"],
            expected_cohort_size * 2 * 3,
        )
        self.assertTrue(
            all(
                summary["source_model_calls"] == expected_cohort_size * 2
                for summary in condition["methods"].values()
            )
        )

    def test_d1a_wrapper_keeps_the_bc_method_name_and_source_fields(
        self,
    ) -> None:
        (
            _,
            _,
            score_episode,
            residual_episode,
        ) = self._patched_episodes()
        context = SimpleNamespace(
            config=SimpleNamespace(attack_config=lambda: self.attack),
            source_families=("classical_cnn", "transformer"),
            evaluation_victims={
                "classical_cnn": (self.victims[0],),
                "transformer": (("transformer-source-0", nn.Identity()),),
            },
            evaluation_samples=self.samples[:1],
            evaluation_indices=self.indices[:1],
        )
        request = SimpleNamespace(seed=17, heldout_family="modern_cnn")

        with (
            patch(
                "rl_transfer.phase2_residual_d1_evaluation.run_score_greedy_episode",
                side_effect=score_episode,
            ),
            patch(
                "rl_transfer.phase2_residual_d1_evaluation.run_residual_ranker_episode",
                side_effect=residual_episode,
            ),
        ):
            conditions, rows, traces = evaluate_residual_d1(
                request,
                context,
                self.bc,
                deadline_check=lambda: None,
                progress=lambda _: None,
            )

        self.assertEqual(
            set(conditions),
            {"classical_cnn", "transformer"},
        )
        self.assertEqual(
            {row.method for row in rows},
            {"score_greedy", "residual_ranker_bc"},
        )
        self.assertEqual(len(traces), len(rows))
        self.assertTrue(
            all(
                trace["victim_family"]
                in {
                    "classical_cnn",
                    "transformer",
                }
                and trace["hidden_target_calls"] == 0
                for trace in traces
            )
        )

    def test_rejects_unsafe_method_names_and_unlocked_attacks(self) -> None:
        invalid_cases = (
            (
                {"../../residual-ppo": self.ppo},
                self.attack,
                "method|identifier|name",
            ),
            (
                {"residual_ranker_ppo": self.ppo},
                _attack(max_queries=49),
                "D1|attack|operator|query",
            ),
        )
        for policies, attack, message in invalid_cases:
            with self.subTest(policies=tuple(policies), attack=attack):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_residual_policy_cohort(
                        policies=policies,
                        victims=self.victims,
                        samples=self.samples,
                        indices=self.indices,
                        attack=attack,
                        family="classical_cnn",
                        seed=29,
                        heldout_family="modern_cnn",
                        source_slice="seen_family_new_instance",
                        deadline_check=lambda: None,
                        progress=lambda _: None,
                    )

    def test_rejects_a_result_that_exceeds_the_locked_query_budget(
        self,
    ) -> None:
        (
            _,
            _,
            score_episode,
            residual_episode,
        ) = self._patched_episodes(
            over_budget_method="residual_ranker_ppo",
        )

        with (
            patch(
                "rl_transfer.phase2_residual_d1_evaluation.run_score_greedy_episode",
                side_effect=score_episode,
            ),
            patch(
                "rl_transfer.phase2_residual_d1_evaluation.run_residual_ranker_episode",
                side_effect=residual_episode,
            ),
            self.assertRaisesRegex(ValueError, "query|budget|calls"),
        ):
            evaluate_residual_policy_cohort(
                policies={"residual_ranker_ppo": self.ppo},
                victims=self.victims[:1],
                samples=self.samples[:1],
                indices=self.indices[:1],
                attack=self.attack,
                family="classical_cnn",
                seed=29,
                heldout_family="modern_cnn",
                source_slice="seen_family_new_instance",
                deadline_check=lambda: None,
                progress=lambda _: None,
            )


if __name__ == "__main__":
    unittest.main()
