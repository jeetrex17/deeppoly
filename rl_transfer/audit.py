from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn


Feedback = Literal["none", "scores", "label"]


class QueryBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryRecord:
    victim_id: str
    sample_id: str
    purpose: str
    step: int
    call_index: int
    feedback: Feedback
    predicted_label: int | None
    error: str | None = None


@dataclass(frozen=True)
class QueryResponse:
    predicted_label: int | None
    scores: torch.Tensor | None


class AuditedVictim:
    """The only target access surface for research protocols."""

    def __init__(self, victim: nn.Module, budget: int, feedback: Feedback, victim_id: str) -> None:
        if budget < 0:
            raise ValueError("budget cannot be negative")
        if feedback not in {"none", "scores", "label"}:
            raise ValueError("unsupported feedback mode")
        self.victim = victim.eval()
        self.budget = budget
        self.feedback = feedback
        self.victim_id = victim_id
        self.trace: list[QueryRecord] = []

    @property
    def calls(self) -> int:
        return len(self.trace)

    def query(self, image: torch.Tensor, sample_id: str, purpose: str, step: int) -> QueryResponse:
        if self.calls >= self.budget:
            raise QueryBudgetExceeded(f"query budget {self.budget} exhausted")
        call_index = self.calls + 1
        try:
            with torch.inference_mode():
                logits = self.victim(image.unsqueeze(0))[0]
                if logits.ndim != 1 or logits.numel() < 2:
                    raise ValueError("victim must return at least two class logits")
                if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
                    raise ValueError("victim logits must be finite floating-point values")
                probabilities = logits.softmax(dim=0).detach().cpu()
            predicted = int(probabilities.argmax())
        except Exception as error:
            self.trace.append(QueryRecord(self.victim_id, sample_id, purpose, step, call_index, self.feedback, None, type(error).__name__))
            raise
        exposed_label = predicted if self.feedback in {"scores", "label"} else None
        self.trace.append(QueryRecord(self.victim_id, sample_id, purpose, step, call_index, self.feedback, exposed_label))
        scores = probabilities.clone() if self.feedback == "scores" else None
        return QueryResponse(exposed_label, scores)

    def trace_dicts(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.trace]
