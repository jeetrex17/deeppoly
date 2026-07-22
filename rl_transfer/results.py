from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ResearchResultRow:
    sample_id: str
    victim_id: str
    victim_family: str
    method: str
    threat_model: str
    seed: int
    query_budget: int
    clean_correct: bool
    success: bool
    query_to_success: int | None
    total_target_calls: int
    linf: float
    l2: float
    policy_digest: str
    action_trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.threat_model not in {"T0", "T1", "T2", "T3"}:
            raise ValueError("unknown threat model")
        if self.query_budget < 0 or self.total_target_calls < 0:
            raise ValueError("query budgets and calls cannot be negative")
        if self.total_target_calls > self.query_budget:
            raise ValueError("target calls exceeded budget")
        if self.success and self.query_to_success is None:
            raise ValueError("successful result needs query_to_success")
        if not self.success and self.query_to_success is not None:
            raise ValueError("failed result cannot have query_to_success")
        if self.query_to_success is not None and not 1 <= self.query_to_success <= self.total_target_calls:
            raise ValueError("query_to_success must be within recorded target calls")


def write_jsonl(path: Path, rows: Iterable[ResearchResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> tuple[ResearchResultRow, ...]:
    return tuple(ResearchResultRow(**{**item, "action_trace": tuple(item["action_trace"])}) for item in (json.loads(line) for line in path.read_text().splitlines() if line))
