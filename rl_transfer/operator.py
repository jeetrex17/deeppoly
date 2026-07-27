"""Shared proposal acceptance contract for learned and control attacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch

from .actions import patch_catalog
from .config import AttackConfig


@dataclass(frozen=True)
class AttackTransition:
    image: torch.Tensor
    accepted: bool


@dataclass(frozen=True)
class AttackOperatorContract:
    epsilon: float
    step_size: float
    grid_size: int
    rollback_on_non_improvement: bool
    channels: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be finite and in (0, 1]")
        if not math.isfinite(self.step_size) or not 0 < self.step_size <= self.epsilon:
            raise ValueError("step_size must be finite and in (0, epsilon]")
        if (
            not isinstance(self.grid_size, int)
            or isinstance(self.grid_size, bool)
            or self.grid_size < 1
        ):
            raise ValueError("grid_size must be a positive integer")
        if not isinstance(self.rollback_on_non_improvement, bool):
            raise ValueError("rollback_on_non_improvement must be boolean")
        if (
            not isinstance(self.channels, int)
            or isinstance(self.channels, bool)
            or self.channels < 1
        ):
            raise ValueError("channels must be a positive integer")

    @classmethod
    def from_config(
        cls,
        config: AttackConfig,
        *,
        channels: int = 3,
    ) -> "AttackOperatorContract":
        return cls(
            epsilon=config.epsilon,
            step_size=config.step_size,
            grid_size=config.grid_size,
            rollback_on_non_improvement=config.rollback_on_non_improvement,
            channels=channels,
        )

    def as_dict(self) -> dict[str, object]:
        catalog = patch_catalog(self.grid_size, self.channels)
        catalog_payload = [
            {
                "basis": action.basis,
                "channel": action.channel,
                "index": action.index,
                "sign": action.sign,
                "step_scale": action.step_scale,
                "basis_size": action.basis_size,
            }
            for action in catalog
        ]
        catalog_digest = hashlib.sha256(
            json.dumps(
                catalog_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            **asdict(self),
            "action_dim": len(catalog),
            "catalog_sha256": catalog_digest,
        }

    def digest(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def choose_attack_transition(
    current: torch.Tensor,
    proposal: torch.Tensor,
    *,
    current_margin: float,
    proposal_margin: float,
    success: bool,
    rollback_on_non_improvement: bool,
) -> AttackTransition:
    """Return a fresh accepted state without mutating either input tensor."""

    if current.shape != proposal.shape:
        raise ValueError("current and proposal tensors must have identical shapes")
    if not torch.isfinite(current).all() or not torch.isfinite(proposal).all():
        raise ValueError("attack states must contain finite values")
    if not math.isfinite(current_margin) or not math.isfinite(proposal_margin):
        raise ValueError("confidence margins must be finite")
    if not isinstance(success, bool) or not isinstance(
        rollback_on_non_improvement,
        bool,
    ):
        raise ValueError("success and rollback flags must be boolean")
    accepted = bool(
        success
        or not rollback_on_non_improvement
        or proposal_margin < current_margin
    )
    return AttackTransition(
        image=(proposal if accepted else current).detach().clone(),
        accepted=accepted,
    )
