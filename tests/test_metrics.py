import unittest

from rl_transfer.metrics import EpisodeResult, aggregate_results


class MetricsTests(unittest.TestCase):
    def test_uses_only_clean_correct_samples_as_asr_denominator(self) -> None:
        results = (
            EpisodeResult("a", True, True, 2, 0.05, 1.0, 0.3, 1),
            EpisodeResult("b", True, False, 4, 0.05, 1.5, 0.1, 2),
            EpisodeResult("c", False, False, 0, 0.0, 0.0, 0.0, 0),
        )

        metrics = aggregate_results(results)

        self.assertEqual(metrics.total_samples, 3)
        self.assertEqual(metrics.eligible_samples, 2)
        self.assertEqual(metrics.attack_success_rate, 0.5)
        self.assertEqual(metrics.mean_queries, 3.0)
        self.assertEqual(metrics.mean_successful_queries, 2.0)
        self.assertEqual(metrics.max_linf, 0.05)

    def test_marks_asr_unavailable_without_eligible_samples(self) -> None:
        metrics = aggregate_results(
            (EpisodeResult("wrong", False, False, 0, 0.0, 0.0, 0.0, 0),)
        )

        self.assertIsNone(metrics.attack_success_rate)
        self.assertIsNone(metrics.mean_queries)
        self.assertIsNone(metrics.mean_successful_queries)


if __name__ == "__main__":
    unittest.main()
