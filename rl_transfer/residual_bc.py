"""Source-balanced behavior cloning for the D1 residual ranker."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import math

import torch
from torch import nn

from .imitation import BehaviorCloneStep
from .recurrent import RecurrentAttackPolicy
from .residual_diagnostics import (
    equal_family_scalar_mean,
    equal_family_tensor_mean,
    trajectory_source_family,
)
from .residual_ranker import (
    ResidualRankerPolicy,
    _soft_target,
    _trajectory_groups,
    score_greedy_action_order,
)


def fit_residual_ranker_bc(
    backbone: RecurrentAttackPolicy,
    examples: Iterable[BehaviorCloneStep],
    *,
    epochs: int,
    seed: int,
    prior_seed: int,
    prior_temperature: float = 24.0,
    pairwise_weight: float = 0.1,
    deadline_check: Callable[[], None] | None = None,
    required_source_families: Sequence[str] | None = None,
) -> dict[str, object]:
    """Fit prior-plus-residual logits with equal-family trajectory weighting."""

    steps = tuple(examples)
    trajectories = _trajectory_groups(steps)
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 1
        or not trajectories
    ):
        raise ValueError("residual BC requires examples and positive epochs")
    if (
        isinstance(pairwise_weight, bool)
        or not isinstance(pairwise_weight, (int, float))
        or not math.isfinite(float(pairwise_weight))
        or pairwise_weight < 0
    ):
        raise ValueError("pairwise weight must be finite and non-negative")
    if deadline_check is not None and not callable(deadline_check):
        raise TypeError("residual BC deadline check must be callable")
    accepted_count = sum(step.accepted for step in steps)
    if accepted_count < 1:
        raise ValueError("residual BC requires accepted source-teacher steps")

    torch.manual_seed(seed)
    policy = ResidualRankerPolicy(
        backbone,
        confidence_threshold=0.0,
        prior_temperature=prior_temperature,
    )
    device = next(policy.parameters()).device
    source_family_diagnostics: dict[str, dict[str, int]] = {}
    for trajectory_id, trajectory in trajectories:
        family = trajectory_source_family(trajectory_id)
        current = source_family_diagnostics.get(
            family,
            {"trajectories": 0, "accepted_steps": 0},
        )
        source_family_diagnostics = {
            **source_family_diagnostics,
            family: {
                "trajectories": current["trajectories"] + 1,
                "accepted_steps": current["accepted_steps"]
                + sum(step.accepted for step in trajectory),
            },
        }
    required = (
        tuple(required_source_families) if required_source_families is not None else ()
    )
    if required and (
        set(source_family_diagnostics) != set(required)
        or any(
            source_family_diagnostics[family]["accepted_steps"] < 1
            for family in required
        )
    ):
        raise ValueError("residual BC lacks an accepted locked source family")

    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        if deadline_check is not None:
            deadline_check()
        listwise_terms: list[tuple[str, torch.Tensor]] = []
        pairwise_terms: list[tuple[str, torch.Tensor]] = []
        trajectory_accuracies: list[tuple[str, float]] = []
        for trajectory_id, trajectory in trajectories:
            if deadline_check is not None:
                deadline_check()
            hidden = policy.initial_state()
            family = trajectory_source_family(trajectory_id)
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
                trajectory_listwise.append(-(target * combined.log_softmax(-1)).sum())
                teacher_action = step.action
                negative_mask = torch.ones(
                    policy.action_dim,
                    dtype=torch.bool,
                    device=device,
                )
                negative_mask[teacher_action] = False
                hard_negatives = (
                    combined[negative_mask].topk(min(5, policy.action_dim - 1)).values
                )
                trajectory_pairwise.append(
                    nn.functional.softplus(
                        hard_negatives - combined[teacher_action]
                    ).mean()
                )
                trajectory_correct += int(combined.argmax().item() == teacher_action)
            if trajectory_listwise:
                listwise_terms.append((family, torch.stack(trajectory_listwise).mean()))
                pairwise_terms.append((family, torch.stack(trajectory_pairwise).mean()))
                trajectory_accuracies.append(
                    (
                        family,
                        trajectory_correct / len(trajectory_listwise),
                    )
                )
        listwise_loss = equal_family_tensor_mean(listwise_terms)
        pairwise_loss = equal_family_tensor_mean(pairwise_terms)
        loss = listwise_loss + float(pairwise_weight) * pairwise_loss
        if deadline_check is not None:
            deadline_check()
        backbone.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            backbone.parameters(),
            backbone.config.gradient_clip_norm,
        )
        if deadline_check is not None:
            deadline_check()
        backbone.optimizer.step()
        if deadline_check is not None:
            deadline_check()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.detach()),
                "listwise_soft_cross_entropy": float(listwise_loss.detach()),
                "pairwise_logistic_loss": float(pairwise_loss.detach()),
                "hybrid_top1_accuracy": equal_family_scalar_mean(trajectory_accuracies),
            }
        )
    return {
        "objective": "prior_plus_residual_listwise_soft_ce_pairwise_logistic",
        "target_mode": "all_soft",
        "aggregation": "equal_family_equal_trajectory",
        "trajectories": len(trajectories),
        "accepted_steps": accepted_count,
        "source_family_diagnostics": source_family_diagnostics,
        "epochs": epochs,
        "pairwise_weight": float(pairwise_weight),
        "history": history,
        "final": history[-1],
    }


__all__ = ("fit_residual_ranker_bc",)
