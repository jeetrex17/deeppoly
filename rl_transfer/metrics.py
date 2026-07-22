from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EpisodeResult:
    sample_id: str
    clean_correct: bool
    success: bool
    queries: int
    linf: float
    l2: float
    confidence_drop: float
    patches: int


@dataclass(frozen=True)
class Metrics:
    total_samples: int
    eligible_samples: int
    attack_success_rate: float | None
    mean_queries: float | None
    mean_successful_queries: float | None
    max_linf: float


def aggregate_results(results: Iterable[EpisodeResult]) -> Metrics:
    rows = tuple(results)
    eligible = tuple(row for row in rows if row.clean_correct)
    successful = tuple(row for row in eligible if row.success)
    return Metrics(len(rows), len(eligible), len(successful) / len(eligible) if eligible else None,
                   sum(row.queries for row in eligible) / len(eligible) if eligible else None,
                   sum(row.queries for row in successful) / len(successful) if successful else None,
                   max((row.linf for row in eligible), default=0.0))
