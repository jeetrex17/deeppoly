"""Worst-family behavior cloning for the source-only residual ranker."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn

from .imitation import BehaviorCloneStep
from .recurrent import RecurrentAttackPolicy
from .residual_diagnostics import (
    equal_family_scalar_mean,
    trajectory_source_family,
)
from .residual_ranker import (
    ResidualRankerPolicy,
    _soft_target,
    _trajectory_groups,
    score_greedy_action_order,
)


def _validated_families(families: Sequence[str]) -> tuple[str, ...]:
    if isinstance(families, (str, bytes)) or not isinstance(families, Sequence):
        raise TypeError("GroupDRO families must be a sequence")
    values = tuple(families)
    if (
        not values
        or any(not isinstance(family, str) or not family for family in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("GroupDRO families must be unique non-empty strings")
    return values


@dataclass(frozen=True)
class GroupDROState:
    """Detached family weights carried between optimizer updates."""

    families: tuple[str, ...]
    weights: tuple[float, ...]
    step: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.families, tuple) or not isinstance(self.weights, tuple):
            raise TypeError("GroupDRO state collections must be immutable tuples")
        families = _validated_families(self.families)
        if (
            len(self.weights) != len(families)
            or any(
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0
                for weight in self.weights
            )
            or not math.isclose(
                sum(float(weight) for weight in self.weights),
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "GroupDRO weights must be finite, positive, and sum to one"
            )
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ValueError("GroupDRO step must be a non-negative integer")

    @classmethod
    def uniform(cls, families: Sequence[str]) -> GroupDROState:
        locked = _validated_families(families)
        weight = 1.0 / len(locked)
        return cls(locked, tuple(weight for _ in locked))

    def as_dict(self) -> dict[str, object]:
        return {
            "families": list(self.families),
            "weights": list(self.weights),
            "step": self.step,
        }


@dataclass(frozen=True)
class GroupDROFamilyAudit:
    """One family's immutable contribution to a GroupDRO update."""

    family: str
    trajectory_count: int
    mean_loss: float
    weight_before: float
    weight_after: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.family, str)
            or not self.family
            or isinstance(self.trajectory_count, bool)
            or not isinstance(self.trajectory_count, int)
            or self.trajectory_count < 1
            or any(
                not math.isfinite(value) or value <= 0
                for value in (self.weight_before, self.weight_after)
            )
            or not math.isfinite(self.mean_loss)
        ):
            raise ValueError("GroupDRO family audit is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "trajectory_count": self.trajectory_count,
            "mean_loss": self.mean_loss,
            "weight_before": self.weight_before,
            "weight_after": self.weight_after,
        }


@dataclass(frozen=True)
class GroupDROAudit:
    """Immutable evidence for one detached family-weight update."""

    step: int
    eta: float
    objective: float
    families: tuple[GroupDROFamilyAudit, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 1
            or isinstance(self.eta, bool)
            or not isinstance(self.eta, (int, float))
            or not math.isfinite(self.eta)
            or self.eta <= 0
            or not math.isfinite(self.objective)
            or not isinstance(self.families, tuple)
            or not self.families
            or any(
                not isinstance(item, GroupDROFamilyAudit) for item in self.families
            )
        ):
            raise ValueError("GroupDRO audit is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "eta": self.eta,
            "objective": self.objective,
            "families": [item.as_dict() for item in self.families],
        }


def _validate_reduction_contract(
    family_trajectory_losses: Mapping[str, Sequence[torch.Tensor]],
    state: GroupDROState,
    *,
    eta: float,
    required_source_families: Sequence[str],
) -> tuple[str, ...]:
    if not isinstance(state, GroupDROState):
        raise TypeError("GroupDRO reduction requires a GroupDROState")
    if not isinstance(family_trajectory_losses, Mapping):
        raise TypeError("GroupDRO losses must be mapped by source family")
    locked = _validated_families(required_source_families)
    if state.families != locked or set(family_trajectory_losses) != set(locked):
        raise ValueError("GroupDRO reduction lacks an exact locked source family set")
    if (
        isinstance(eta, bool)
        or not isinstance(eta, (int, float))
        or not math.isfinite(float(eta))
        or float(eta) <= 0
    ):
        raise ValueError("GroupDRO eta must be finite and positive")
    return locked


def _family_trajectory_means(
    family_trajectory_losses: Mapping[str, Sequence[torch.Tensor]],
    families: Sequence[str],
) -> tuple[torch.Tensor, ...]:
    reference: torch.Tensor | None = None
    family_means: list[torch.Tensor] = []
    for family in families:
        losses = family_trajectory_losses[family]
        if not isinstance(losses, Sequence) or not losses:
            raise ValueError("each locked source family requires trajectory losses")
        validated: list[torch.Tensor] = []
        for loss in losses:
            if (
                not isinstance(loss, torch.Tensor)
                or loss.ndim != 0
                or not bool(torch.isfinite(loss.detach()))
            ):
                raise ValueError("GroupDRO trajectory losses must be finite scalars")
            if reference is not None and (
                loss.device != reference.device or loss.dtype != reference.dtype
            ):
                raise ValueError(
                    "GroupDRO trajectory losses must share device and dtype"
                )
            reference = loss if reference is None else reference
            validated.append(loss)
        family_means.append(
            torch.stack(tuple(loss.to(dtype=torch.float64) for loss in validated)).mean()
        )
    return tuple(family_means)


def _stable_exponentiated_weights(
    state: GroupDROState,
    family_means: Sequence[torch.Tensor],
    *,
    eta: float,
) -> tuple[float, ...]:
    log_unnormalized = tuple(
        math.log(weight) + float(eta) * float(loss.detach())
        for weight, loss in zip(state.weights, family_means)
    )
    if any(not math.isfinite(value) for value in log_unnormalized):
        raise ValueError("GroupDRO weight update produced a non-finite log weight")
    maximum = max(log_unnormalized)
    minimum_log_weight = math.log(math.ulp(0.0))
    shifted = tuple(
        math.exp(max(value - maximum, minimum_log_weight))
        for value in log_unnormalized
    )
    normalizer = sum(shifted)
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("GroupDRO weight normalization is invalid")
    normalized = tuple(value / normalizer for value in shifted)
    if any(not math.isfinite(value) or value <= 0 for value in normalized):
        raise ValueError("GroupDRO weight update collapsed a locked source family")
    return normalized


def reduce_groupdro_family_losses(
    family_trajectory_losses: Mapping[str, Sequence[torch.Tensor]],
    state: GroupDROState,
    *,
    eta: float,
    required_source_families: Sequence[str],
) -> tuple[torch.Tensor, GroupDROState, GroupDROAudit]:
    """Reduce trajectory losses with a detached, stable GroupDRO update."""

    locked = _validate_reduction_contract(
        family_trajectory_losses,
        state,
        eta=eta,
        required_source_families=required_source_families,
    )
    family_means = _family_trajectory_means(family_trajectory_losses, locked)
    weights_after = _stable_exponentiated_weights(
        state,
        family_means,
        eta=float(eta),
    )
    detached_weights = family_means[0].new_tensor(weights_after)
    objective = (
        torch.stack(family_means).to(dtype=torch.float64) * detached_weights
    ).sum()
    if not bool(torch.isfinite(objective.detach())):
        raise ValueError("GroupDRO objective must be finite")
    next_state = GroupDROState(
        families=state.families,
        weights=weights_after,
        step=state.step + 1,
    )
    audit = GroupDROAudit(
        step=next_state.step,
        eta=float(eta),
        objective=float(objective.detach()),
        families=tuple(
            GroupDROFamilyAudit(
                family=family,
                trajectory_count=len(family_trajectory_losses[family]),
                mean_loss=float(mean.detach()),
                weight_before=state.weights[index],
                weight_after=next_state.weights[index],
            )
            for index, (family, mean) in enumerate(zip(locked, family_means))
        ),
    )
    return objective, next_state, audit


def _source_family_diagnostics(
    trajectories: Sequence[tuple[str, tuple[BehaviorCloneStep, ...]]],
) -> dict[str, dict[str, int]]:
    diagnostics: dict[str, dict[str, int]] = {}
    for trajectory_id, trajectory in trajectories:
        family = trajectory_source_family(trajectory_id)
        current = diagnostics.get(
            family,
            {"trajectories": 0, "accepted_steps": 0},
        )
        diagnostics = {
            **diagnostics,
            family: {
                "trajectories": current["trajectories"] + 1,
                "accepted_steps": current["accepted_steps"]
                + sum(step.accepted for step in trajectory),
            },
        }
    return diagnostics


def _validate_fit_contract(
    *,
    epochs: int,
    pairwise_weight: float,
    groupdro_eta: float,
    deadline_check: Callable[[], None] | None,
) -> None:
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("GroupDRO residual BC requires positive epochs")
    numeric = (pairwise_weight, groupdro_eta)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric
    ):
        raise ValueError("GroupDRO residual BC controls must be finite")
    if pairwise_weight < 0 or groupdro_eta <= 0:
        raise ValueError("GroupDRO residual BC controls are outside their valid range")
    if deadline_check is not None and not callable(deadline_check):
        raise TypeError("GroupDRO residual BC deadline check must be callable")


def _weighted_family_component(
    terms: Sequence[tuple[str, torch.Tensor]],
    state: GroupDROState,
) -> float:
    grouped: dict[str, list[torch.Tensor]] = {
        family: [] for family in state.families
    }
    for family, term in terms:
        grouped[family].append(term)
    means = tuple(
        torch.stack(tuple(grouped[family])).mean() for family in state.families
    )
    return sum(
        weight * float(mean.detach())
        for weight, mean in zip(state.weights, means)
    )


def fit_groupdro_residual_ranker_bc(
    backbone: RecurrentAttackPolicy,
    examples: Iterable[BehaviorCloneStep],
    *,
    epochs: int,
    seed: int,
    prior_seed: int,
    required_source_families: Sequence[str],
    prior_temperature: float = 24.0,
    pairwise_weight: float = 0.1,
    groupdro_eta: float = 0.1,
    initial_groupdro_state: GroupDROState | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Fit residual logits against the current worst source-family objective."""

    if not isinstance(backbone, RecurrentAttackPolicy):
        raise TypeError("GroupDRO residual BC requires a recurrent backbone")
    if backbone.action_dim < 2:
        raise ValueError("GroupDRO residual BC requires at least two actions")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (seed, prior_seed)
    ):
        raise ValueError("GroupDRO residual BC seeds must be non-negative integers")
    _validate_fit_contract(
        epochs=epochs,
        pairwise_weight=pairwise_weight,
        groupdro_eta=groupdro_eta,
        deadline_check=deadline_check,
    )
    locked = _validated_families(required_source_families)
    steps = tuple(examples)
    trajectories = _trajectory_groups(steps)
    if not trajectories:
        raise ValueError("GroupDRO residual BC requires examples")
    if any(
        len(step.observation) != backbone.observation_dim
        or step.action >= backbone.action_dim
        or step.action_distribution is None
        or len(step.action_distribution) != backbone.action_dim
        for step in steps
    ):
        raise ValueError(
            "GroupDRO residual BC examples do not match backbone dimensions"
        )
    diagnostics = _source_family_diagnostics(trajectories)
    if set(diagnostics) != set(locked) or any(
        diagnostics.get(family, {}).get("accepted_steps", 0) < 1
        for family in locked
    ):
        raise ValueError("GroupDRO residual BC lacks an accepted locked source family")
    state = (
        GroupDROState.uniform(locked)
        if initial_groupdro_state is None
        else initial_groupdro_state
    )
    if not isinstance(state, GroupDROState) or state.families != locked:
        raise ValueError("initial GroupDRO state does not match locked source families")

    torch.manual_seed(seed)
    policy = ResidualRankerPolicy(
        backbone,
        confidence_threshold=0.0,
        prior_temperature=prior_temperature,
    )
    device = next(policy.parameters()).device
    history: tuple[dict[str, object], ...] = ()
    audits: tuple[GroupDROAudit, ...] = ()
    accepted_count = sum(step.accepted for step in steps)
    for epoch in range(epochs):
        if deadline_check is not None:
            deadline_check()
        total_terms: dict[str, list[torch.Tensor]] = {
            family: [] for family in locked
        }
        listwise_terms: list[tuple[str, torch.Tensor]] = []
        pairwise_terms: list[tuple[str, torch.Tensor]] = []
        trajectory_accuracies: list[tuple[str, float]] = []
        for trajectory_id, trajectory in trajectories:
            if deadline_check is not None:
                deadline_check()
            family = trajectory_source_family(trajectory_id)
            hidden = policy.initial_state()
            order = score_greedy_action_order(
                action_dim=policy.action_dim,
                seed=prior_seed,
                sample_id=trajectory_id,
            )
            trajectory_listwise: list[torch.Tensor] = []
            trajectory_pairwise: list[torch.Tensor] = []
            trajectory_correct = 0
            for step in trajectory:
                if deadline_check is not None:
                    deadline_check()
                observation = torch.as_tensor(
                    step.observation,
                    dtype=torch.float32,
                    device=device,
                )
                combined, hidden = policy.combined_logits(
                    observation,
                    hidden,
                    prior_order=order,
                    proposal_index=step.step_index,
                )
                if not step.accepted:
                    continue
                target = _soft_target(
                    step,
                    action_dim=policy.action_dim,
                    device=device,
                )
                listwise = -(target * combined.log_softmax(-1)).sum()
                negative_mask = torch.ones(
                    policy.action_dim,
                    dtype=torch.bool,
                    device=device,
                )
                negative_mask[step.action] = False
                hard_negatives = combined[negative_mask].topk(
                    min(5, policy.action_dim - 1)
                ).values
                pairwise = nn.functional.softplus(
                    hard_negatives - combined[step.action]
                ).mean()
                trajectory_listwise.append(listwise)
                trajectory_pairwise.append(pairwise)
                trajectory_correct += int(combined.argmax().item() == step.action)
            if trajectory_listwise:
                listwise_mean = torch.stack(tuple(trajectory_listwise)).mean()
                pairwise_mean = torch.stack(tuple(trajectory_pairwise)).mean()
                total_terms[family].append(
                    listwise_mean + float(pairwise_weight) * pairwise_mean
                )
                listwise_terms.append((family, listwise_mean))
                pairwise_terms.append((family, pairwise_mean))
                trajectory_accuracies.append(
                    (
                        family,
                        trajectory_correct / len(trajectory_listwise),
                    )
                )
        objective, next_state, reduction_audit = reduce_groupdro_family_losses(
            {family: tuple(total_terms[family]) for family in locked},
            state,
            eta=groupdro_eta,
            required_source_families=locked,
        )
        if deadline_check is not None:
            deadline_check()
        backbone.optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(
            backbone.parameters(),
            backbone.config.gradient_clip_norm,
        )
        if deadline_check is not None:
            deadline_check()
        backbone.optimizer.step()
        state = next_state
        audits = (*audits, reduction_audit)
        if deadline_check is not None:
            deadline_check()
        history = (
            *history,
            {
                "epoch": epoch + 1,
                "loss": float(objective.detach()),
                "listwise_soft_cross_entropy": _weighted_family_component(
                    listwise_terms,
                    state,
                ),
                "pairwise_logistic_loss": _weighted_family_component(
                    pairwise_terms,
                    state,
                ),
                "hybrid_top1_accuracy": equal_family_scalar_mean(
                    trajectory_accuracies
                ),
                "groupdro_state": state.as_dict(),
                "groupdro_audit": reduction_audit.as_dict(),
            },
        )
    return {
        "objective": (
            "prior_plus_residual_groupdro_listwise_soft_ce_pairwise_logistic"
        ),
        "target_mode": "all_soft",
        "aggregation": "equal_trajectory_then_groupdro",
        "trajectories": len(trajectories),
        "accepted_steps": accepted_count,
        "source_family_diagnostics": diagnostics,
        "epochs": epochs,
        "pairwise_weight": float(pairwise_weight),
        "groupdro_eta": float(groupdro_eta),
        "groupdro_state": state,
        "groupdro_audits": audits,
        "history": history,
        "final": history[-1],
    }


__all__ = (
    "GroupDROAudit",
    "GroupDROFamilyAudit",
    "GroupDROState",
    "fit_groupdro_residual_ranker_bc",
    "reduce_groupdro_family_losses",
)
