from __future__ import annotations

import unittest

from rl_transfer.phase2_residual_d1_reporting import (
    paired_source_statistics,
)
from rl_transfer.results import ResearchResultRow


def _row(
    *,
    sample: int,
    method: str,
    success_query: int | None,
) -> ResearchResultRow:
    return ResearchResultRow(
        sample_id=f"sample-{sample}",
        victim_id="source-victim-0",
        victim_family="classical_cnn",
        method=method,
        threat_model="T1",
        seed=17,
        query_budget=50,
        clean_correct=True,
        success=success_query is not None,
        query_to_success=success_query,
        total_target_calls=success_query or 50,
        linf=0.0,
        l2=0.0,
        policy_digest=("a" if method == "score_greedy" else "b") * 64,
        action_trace=tuple(0 for _ in range((success_query or 50) - 1)),
    )


class ResidualD1ReportingTests(unittest.TestCase):
    def test_paired_statistics_are_deterministic_and_condition_on_images(
        self,
    ) -> None:
        control_queries = (None, None, 50, None)
        learned_queries = (10, None, 40, None)
        rows = tuple(
            row
            for sample, (control, learned) in enumerate(
                zip(control_queries, learned_queries)
            )
            for row in (
                _row(
                    sample=sample,
                    method="score_greedy",
                    success_query=control,
                ),
                _row(
                    sample=sample,
                    method="residual_ranker_bc",
                    success_query=learned,
                ),
            )
        )

        first = paired_source_statistics(
            rows,
            learned_method="residual_ranker_bc",
            bootstrap_samples=200,
            seed=31,
        )
        second = paired_source_statistics(
            rows,
            learned_method="residual_ranker_bc",
            bootstrap_samples=200,
            seed=31,
        )

        self.assertEqual(first, second)
        family = first["by_family"]["classical_cnn"]
        self.assertEqual(family["paired_eligible"], 4)
        self.assertEqual(family["asr_difference"], 0.25)
        self.assertGreater(family["query_auc_difference"], 0.0)
        self.assertEqual(
            family["inference_scope"],
            "fixed-victim paired image bootstrap",
        )
        self.assertFalse(first["supports_seed_level_inference"])

    def test_paired_statistics_reject_unmatched_cohorts(self) -> None:
        rows = (
            _row(
                sample=0,
                method="score_greedy",
                success_query=None,
            ),
            _row(
                sample=1,
                method="residual_ranker_bc",
                success_query=10,
            ),
        )

        with self.assertRaisesRegex(ValueError, "paired|cohort|match"):
            paired_source_statistics(
                rows,
                learned_method="residual_ranker_bc",
                bootstrap_samples=100,
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
