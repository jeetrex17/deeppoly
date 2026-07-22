from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class AttackOutcome:
    clean_correct: bool
    query_to_success: int | None

    def __post_init__(self) -> None:
        if self.query_to_success is not None and self.query_to_success < 0:
            raise ValueError("query_to_success cannot be negative")


def asr_at_budgets(outcomes: Sequence[AttackOutcome], budgets: Sequence[int]) -> dict[int, float]:
    eligible = tuple(outcome for outcome in outcomes if outcome.clean_correct)
    if not eligible:
        raise ValueError("at least one clean-correct outcome is required")
    return {
        int(budget): sum(
            outcome.query_to_success is not None and outcome.query_to_success <= budget
            for outcome in eligible
        ) / len(eligible)
        for budget in budgets
    }


def asr_query_auc(rates: Mapping[int, float]) -> float:
    points = sorted((int(budget), float(rate)) for budget, rate in rates.items())
    if len(points) < 2 or points[-1][0] <= points[0][0]:
        raise ValueError("AUC requires at least two distinct budgets")
    area = sum((right_budget - left_budget) * (left_rate + right_rate) / 2 for (left_budget, left_rate), (right_budget, right_rate) in zip(points, points[1:]))
    return area / (points[-1][0] - points[0][0])


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(indexed)
    running = 0.0
    for rank, (original_index, value) in enumerate(indexed):
        running = max(running, min(1.0, (len(indexed) - rank) * value))
        adjusted[original_index] = running
    return tuple(adjusted)
