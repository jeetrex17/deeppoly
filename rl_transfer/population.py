from dataclasses import dataclass
import math
import random
from typing import Mapping


def balanced_family_schedule(families: tuple[str, ...], episodes: int, seed: int) -> tuple[str, ...]:
    if not families or episodes < 0:
        raise ValueError("families are required and episodes cannot be negative")
    rng = random.Random(seed)
    schedule: list[str] = []
    while len(schedule) < episodes:
        cycle = list(families)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    return tuple(schedule[:episodes])


@dataclass(frozen=True)
class FamilyRobustWeights:
    families: tuple[str, ...]
    eta: float = 0.1
    values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.families or self.eta <= 0:
            raise ValueError("families and positive eta are required")
        if self.values is None:
            object.__setattr__(self, "values", tuple(1 / len(self.families) for _ in self.families))

    def update(self, losses: Mapping[str, float]) -> dict[str, float]:
        if set(losses) != set(self.families):
            raise ValueError("losses must contain exactly the configured families")
        raw = [weight * math.exp(self.eta * float(losses[family])) for family, weight in zip(self.families, self.values)]
        total = sum(raw)
        return {family: value / total for family, value in zip(self.families, raw)}
