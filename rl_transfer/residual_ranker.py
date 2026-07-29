"""Confidence-gated residual ranker for the source-only D1 diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .actions import apply_action, patch_catalog
from .audit import AuditedVictim
from .config import AttackConfig
from .features import configured_patch_image_features
from .imitation import BehaviorCloneStep
from .operator import choose_attack_transition
from .phase2_residual_d1 import select_residual_action
from .recurrent import RecurrentAttackPolicy
from .residual_diagnostics import (
    summarize_competence_trajectories,
    summarize_threshold_choices,
)
from .research_protocol import (
    FrozenEpisodeResult,
    calibration_resistant_observation,
)
from .rewards import recurrent_attack_reward, score_margin


@dataclass(frozen=True)
class ResidualDecision:
    action: int
    learned_action: int
    score_greedy_action: int
    confidence: float
    used_residual: bool
    hidden: torch.Tensor


def score_greedy_action_order(
    *,
    action_dim: int,
    seed: int,
    sample_id: str,
) -> tuple[int, ...]:
    """Reproduce the proposal order used by ``run_score_greedy_episode``."""

    if (
        isinstance(action_dim, bool)
        or not isinstance(action_dim, int)
        or action_dim < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not isinstance(sample_id, str)
        or not sample_id
    ):
        raise ValueError("invalid score-greedy order controls")
    order = list(range(action_dim))
    episode_seed = seed ^ int.from_bytes(
        hashlib.sha256(sample_id.encode("utf-8")).digest()[:8],
        "big",
    )
    # Reproducible experiment scheduling is intentional; this RNG is not
    # used for secrets or any security-sensitive decision.
    random.Random(episode_seed).shuffle(order)  # noqa: S311
    return tuple(order)


class ResidualRankerPolicy(nn.Module):
    """A frozen recurrent scorer added to a deterministic proposal prior."""

    def __init__(
        self,
        backbone: RecurrentAttackPolicy,
        *,
        confidence_threshold: float,
        prior_temperature: float = 24.0,
        overrides_enabled: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, RecurrentAttackPolicy):
            raise TypeError("residual ranker requires a recurrent backbone")
        numeric = (confidence_threshold, prior_temperature)
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in numeric
            )
            or prior_temperature <= 0
        ):
            raise ValueError("residual ranker controls must be finite and valid")
        if not isinstance(overrides_enabled, bool):
            raise ValueError("residual override control must be boolean")
        self.backbone = backbone
        self.confidence_threshold = float(confidence_threshold)
        self.prior_temperature = float(prior_temperature)
        self.overrides_enabled = overrides_enabled

    @property
    def action_dim(self) -> int:
        return self.backbone.action_dim

    def initial_state(self) -> torch.Tensor:
        return self.backbone.initial_state()

    def persistent_digest(self) -> str:
        payload = {
            "schema": "residual-ranker-d1-v1",
            "backbone": self.backbone.persistent_digest(),
            "confidence_threshold": self.confidence_threshold,
            "prior_temperature": self.prior_temperature,
            "overrides_enabled": self.overrides_enabled,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def combined_logits(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        *,
        prior_order: Sequence[int],
        proposal_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        order = tuple(prior_order)
        if (
            len(order) != self.action_dim
            or set(order) != set(range(self.action_dim))
            or isinstance(proposal_index, bool)
            or not isinstance(proposal_index, int)
            or proposal_index < 0
        ):
            raise ValueError("residual logits require a valid prior cursor")
        residual_logits, _, next_hidden = self.backbone(observation, hidden)
        cursor = proposal_index % self.action_dim
        active_order = order[cursor:] + order[:cursor]
        ranks = torch.empty(
            self.action_dim,
            dtype=residual_logits.dtype,
            device=residual_logits.device,
        )
        ranks[
            torch.as_tensor(
                active_order,
                dtype=torch.long,
                device=residual_logits.device,
            )
        ] = torch.arange(
            self.action_dim,
            dtype=residual_logits.dtype,
            device=residual_logits.device,
        )
        return residual_logits - ranks / self.prior_temperature, next_hidden

    def decide(
        self,
        observation: np.ndarray,
        hidden: torch.Tensor,
        *,
        prior_order: Sequence[int],
        proposal_index: int,
    ) -> ResidualDecision:
        order = tuple(prior_order)
        if (
            len(order) != self.action_dim
            or set(order) != set(range(self.action_dim))
            or isinstance(proposal_index, bool)
            or not isinstance(proposal_index, int)
            or proposal_index < 0
        ):
            raise ValueError("residual decision requires a complete prior order")
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=hidden.device,
        )
        with torch.inference_mode():
            combined, next_hidden = self.combined_logits(
                observation_tensor,
                hidden,
                prior_order=order,
                proposal_index=proposal_index,
            )
        score_action = order[proposal_index % self.action_dim]
        learned_action = int(combined.argmax())
        confidence = float(
            (combined[learned_action] - combined[score_action]).clamp_min(0)
        )
        action = (
            select_residual_action(
                score_greedy_action=score_action,
                learned_action=learned_action,
                residual_confidence=confidence,
                confidence_threshold=self.confidence_threshold,
            )
            if self.overrides_enabled
            else score_action
        )
        return ResidualDecision(
            action=action,
            learned_action=learned_action,
            score_greedy_action=score_action,
            confidence=confidence,
            used_residual=action != score_action,
            hidden=next_hidden.detach(),
        )


def _trajectory_groups(
    examples: Sequence[BehaviorCloneStep],
) -> tuple[tuple[str, tuple[BehaviorCloneStep, ...]], ...]:
    grouped: dict[str, list[BehaviorCloneStep]] = {}
    for step in examples:
        if not isinstance(step, BehaviorCloneStep):
            raise TypeError("residual BC examples must be BehaviorCloneStep values")
        grouped.setdefault(step.trajectory_id, []).append(step)
    return tuple(
        (
            trajectory_id,
            tuple(sorted(steps, key=lambda item: item.step_index)),
        )
        for trajectory_id, steps in sorted(grouped.items())
    )


def _soft_target(
    step: BehaviorCloneStep,
    *,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if step.action_distribution is None:
        raise ValueError("D1 residual BC requires all-soft teacher targets")
    target = torch.as_tensor(
        step.action_distribution,
        dtype=torch.float32,
        device=device,
    )
    if (
        target.shape != (action_dim,)
        or not torch.isfinite(target).all()
        or bool((target < 0).any())
        or not torch.isclose(target.sum(), torch.tensor(1.0, device=device))
    ):
        raise ValueError("D1 residual BC received an invalid soft target")
    return target


def _prior_logits(
    policy: ResidualRankerPolicy,
    prior_order: Sequence[int],
    *,
    proposal_index: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    order = tuple(prior_order)
    cursor = proposal_index % policy.action_dim
    active_order = order[cursor:] + order[:cursor]
    ranks = torch.empty(policy.action_dim, dtype=dtype, device=device)
    ranks[torch.as_tensor(active_order, dtype=torch.long, device=device)] = (
        torch.arange(policy.action_dim, dtype=dtype, device=device)
    )
    return -ranks / policy.prior_temperature


def evaluate_residual_ranker_examples(
    policy: ResidualRankerPolicy,
    examples: Iterable[BehaviorCloneStep],
    *,
    prior_seed: int,
    required_source_families: Sequence[str] | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Evaluate the deployed confidence gate on a sealed BC data role."""

    if deadline_check is not None and not callable(deadline_check):
        raise TypeError("residual competence deadline check must be callable")
    trajectories = _trajectory_groups(tuple(examples))
    device = next(policy.parameters()).device
    accepted = 0
    hybrid_correct = 0
    gated_correct = 0
    prior_correct = 0
    residual_uses = 0
    top5_correct = 0
    soft_ce: list[float] = []
    prior_soft_ce: list[float] = []
    trajectory_records: list[dict[str, object]] = []
    with torch.inference_mode():
        for trajectory_id, trajectory in trajectories:
            if deadline_check is not None:
                deadline_check()
            hidden = policy.initial_state()
            order = score_greedy_action_order(
                action_dim=policy.action_dim,
                seed=prior_seed,
                sample_id=trajectory_id,
            )
            trajectory_accepted = 0
            trajectory_hybrid_correct = 0
            trajectory_gated_correct = 0
            trajectory_prior_correct = 0
            trajectory_residual_uses = 0
            trajectory_top5_correct = 0
            trajectory_soft_ce = 0.0
            trajectory_prior_soft_ce = 0.0
            for step in trajectory:
                if deadline_check is not None:
                    deadline_check()
                observation = torch.as_tensor(
                    step.observation,
                    dtype=torch.float32,
                    device=device,
                )
                combined, next_hidden = policy.combined_logits(
                    observation,
                    hidden,
                    prior_order=order,
                    proposal_index=step.step_index,
                )
                hidden = next_hidden
                if not step.accepted:
                    continue
                accepted += 1
                trajectory_accepted += 1
                target = _soft_target(
                    step,
                    action_dim=policy.action_dim,
                    device=device,
                )
                teacher_action = step.action
                prior_action = order[step.step_index % policy.action_dim]
                hybrid_action = int(combined.argmax())
                confidence = float(
                    (combined[hybrid_action] - combined[prior_action]).clamp_min(0)
                )
                gated_action = (
                    select_residual_action(
                        score_greedy_action=prior_action,
                        learned_action=hybrid_action,
                        residual_confidence=confidence,
                        confidence_threshold=policy.confidence_threshold,
                    )
                    if policy.overrides_enabled
                    else prior_action
                )
                hybrid_correct += int(hybrid_action == teacher_action)
                gated_correct += int(gated_action == teacher_action)
                prior_correct += int(prior_action == teacher_action)
                residual_uses += int(gated_action != prior_action)
                trajectory_hybrid_correct += int(hybrid_action == teacher_action)
                trajectory_gated_correct += int(gated_action == teacher_action)
                trajectory_prior_correct += int(prior_action == teacher_action)
                trajectory_residual_uses += int(gated_action != prior_action)
                top5_correct += int(
                    teacher_action
                    in combined.topk(min(5, policy.action_dim)).indices.tolist()
                )
                trajectory_top5_correct += int(
                    teacher_action
                    in combined.topk(min(5, policy.action_dim)).indices.tolist()
                )
                step_soft_ce = float(-(target * combined.log_softmax(-1)).sum())
                soft_ce.append(step_soft_ce)
                trajectory_soft_ce += step_soft_ce
                prior = _prior_logits(
                    policy,
                    order,
                    proposal_index=step.step_index,
                    dtype=combined.dtype,
                    device=device,
                )
                step_prior_soft_ce = float(-(target * prior.log_softmax(-1)).sum())
                prior_soft_ce.append(step_prior_soft_ce)
                trajectory_prior_soft_ce += step_prior_soft_ce
            if trajectory_accepted:
                trajectory_records.append(
                    {
                        "trajectory_id": trajectory_id,
                        "accepted_steps": trajectory_accepted,
                        "hybrid_correct": trajectory_hybrid_correct,
                        "gated_correct": trajectory_gated_correct,
                        "prior_correct": trajectory_prior_correct,
                        "top5_correct": trajectory_top5_correct,
                        "soft_ce_total": trajectory_soft_ce,
                        "prior_soft_ce_total": trajectory_prior_soft_ce,
                        "residual_uses": trajectory_residual_uses,
                    }
                )
    if accepted < 1:
        raise ValueError("residual competence evaluation has no accepted steps")
    balanced = summarize_competence_trajectories(trajectory_records)
    required = (
        set(required_source_families) if required_source_families is not None else set()
    )
    if required and set(balanced["by_source_family"]) != required:
        raise ValueError("residual competence lacks a locked source family")
    return {
        "target_mode": "all_soft",
        "trajectories": len(trajectories),
        "accepted_steps": accepted,
        "hybrid_top1_accuracy": hybrid_correct / accepted,
        "gated_top1_accuracy": gated_correct / accepted,
        "prior_top1_accuracy": prior_correct / accepted,
        "hybrid_top5_accuracy": top5_correct / accepted,
        "soft_cross_entropy": sum(soft_ce) / accepted,
        "prior_soft_cross_entropy": sum(prior_soft_ce) / accepted,
        "residual_use_fraction": residual_uses / accepted,
        "confidence_threshold": policy.confidence_threshold,
        "overrides_enabled": policy.overrides_enabled,
        **balanced,
    }


def select_confidence_threshold(
    backbone: RecurrentAttackPolicy,
    validation_steps: Iterable[BehaviorCloneStep],
    *,
    seed: int,
    prior_temperature: float = 24.0,
    thresholds: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    required_source_families: Sequence[str] | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Select a conservative fallback threshold on BC validation data only."""

    if deadline_check is not None and not callable(deadline_check):
        raise TypeError("confidence threshold deadline check must be callable")
    examples = tuple(validation_steps)
    if not examples:
        raise ValueError("confidence selection requires validation examples")
    candidates = tuple(float(value) for value in thresholds)
    if not candidates or any(
        not math.isfinite(value) or value < 0 for value in candidates
    ):
        raise ValueError("confidence thresholds must be finite and non-negative")

    grouped: dict[str, list[BehaviorCloneStep]] = {}
    for step in examples:
        grouped.setdefault(step.trajectory_id, []).append(step)
    policy = ResidualRankerPolicy(
        backbone,
        confidence_threshold=0.0,
        prior_temperature=prior_temperature,
    )
    observations: list[tuple[str, int, int, float, int]] = []
    for trajectory_id, trajectory in sorted(grouped.items()):
        if deadline_check is not None:
            deadline_check()
        hidden = policy.initial_state()
        order = score_greedy_action_order(
            action_dim=policy.action_dim,
            seed=seed,
            sample_id=trajectory_id,
        )
        for step in sorted(trajectory, key=lambda item: item.step_index):
            if deadline_check is not None:
                deadline_check()
            decision = policy.decide(
                np.asarray(step.observation, dtype=np.float32),
                hidden,
                prior_order=order,
                proposal_index=step.step_index,
            )
            hidden = decision.hidden
            if step.accepted:
                observations.append(
                    (
                        trajectory_id,
                        decision.score_greedy_action,
                        decision.learned_action,
                        decision.confidence,
                        step.action,
                    )
                )
    if not observations:
        raise ValueError("confidence selection has no accepted validation steps")

    baseline_summary = summarize_threshold_choices(
        observations,
        tuple(observation[1] for observation in observations),
    )
    learned_summary = summarize_threshold_choices(
        observations,
        tuple(observation[2] for observation in observations),
    )
    required = (
        set(required_source_families) if required_source_families is not None else set()
    )
    if required and set(baseline_summary["by_source_family"]) != required:
        raise ValueError("confidence selection lacks a locked source family")
    fallback_threshold = math.nextafter(
        max(observation[3] for observation in observations),
        math.inf,
    )
    evaluated_thresholds = tuple(dict.fromkeys((*candidates, fallback_threshold)))
    evaluations: list[dict[str, object]] = []
    for threshold in evaluated_thresholds:
        if deadline_check is not None:
            deadline_check()
        selected = tuple(
            learned if confidence >= threshold else baseline
            for _, baseline, learned, confidence, _ in observations
        )
        metrics = summarize_threshold_choices(observations, selected)
        evaluations.append(
            {
                "threshold": threshold,
                **metrics,
                "selection_mode": (
                    "always_fallback"
                    if metrics["pooled_residual_use_fraction"] == 0
                    else "confidence_gate"
                ),
            }
        )
    selected = max(
        evaluations,
        key=lambda item: (
            float(item["accuracy"]),
            -float(item["residual_use_fraction"]),
            float(item["threshold"]),
        ),
    )
    return {
        "selection_role": "bc_validation_only",
        "aggregation": "equal_trajectory_then_family",
        "threshold": selected["threshold"],
        "accuracy": selected["accuracy"],
        "prior_accuracy": baseline_summary["accuracy"],
        "learned_accuracy": learned_summary["accuracy"],
        "residual_use_fraction": selected["residual_use_fraction"],
        "by_source_family": selected["by_source_family"],
        "pooled_accuracy": selected["pooled_accuracy"],
        "pooled_residual_use_fraction": selected["pooled_residual_use_fraction"],
        "selection_mode": selected["selection_mode"],
        "overrides_enabled": selected["selection_mode"] != "always_fallback",
        "accepted_steps": len(observations),
        "candidate_evaluations": evaluations,
    }


def run_residual_ranker_episode(
    policy: ResidualRankerPolicy,
    victim: nn.Module,
    image: torch.Tensor,
    label: int,
    sample_id: str,
    victim_id: str,
    family: str,
    config: AttackConfig,
    *,
    score_prior_seed: int,
    deadline_check: Callable[[], None] | None = None,
) -> FrozenEpisodeResult:
    """Run the confidence-gated ranker with matched score-query accounting."""

    if config.max_queries < 1:
        raise ValueError("D1 requires at least one source query")
    if deadline_check is not None and not callable(deadline_check):
        raise TypeError("residual-ranker deadline check must be callable")
    if deadline_check is not None:
        deadline_check()
    device = next(policy.parameters()).device
    before = policy.persistent_digest()
    oracle = AuditedVictim(victim, config.max_queries, "scores", victim_id)
    original = image.detach().clone().float().clamp(0, 1).to(device)
    accepted = original.clone()
    response = oracle.query(accepted, sample_id, "initialization", 0)
    clean_scores = response.scores.clone()
    if response.predicted_label != label:
        return FrozenEpisodeResult(
            sample_id,
            victim_id,
            family,
            False,
            False,
            None,
            oracle.calls,
            0.0,
            0.0,
            (),
            before,
            policy.persistent_digest(),
            tuple(oracle.trace_dicts()),
        )

    catalog = patch_catalog(config.grid_size, image.shape[0])
    if len(catalog) != config.action_dim or policy.action_dim != config.action_dim:
        raise ValueError("D1 policy and action catalog dimensions do not match")
    order = score_greedy_action_order(
        action_dim=config.action_dim,
        seed=score_prior_seed,
        sample_id=sample_id,
    )
    hidden = policy.initial_state()
    previous_action: int | None = None
    previous_reward = 0.0
    action_counts = np.zeros(config.action_dim, dtype=np.float32)
    action_values = np.zeros(config.action_dim, dtype=np.float32)
    actions: list[int] = []
    success = False
    success_query: int | None = None
    while oracle.calls < config.max_queries and not success:
        if deadline_check is not None:
            deadline_check()
        observation = calibration_resistant_observation(
            response.scores,
            label,
            clean_scores,
            (config.max_queries - oracle.calls) / config.max_queries,
            previous_action,
            config.action_dim,
            previous_reward,
            oracle.calls / config.max_queries,
            action_counts if config.action_history_features else None,
            action_values if config.action_history_features else None,
            configured_patch_image_features(original, accepted, config),
        )
        decision = policy.decide(
            observation,
            hidden,
            prior_order=order,
            proposal_index=len(actions),
        )
        hidden = decision.hidden
        proposal = apply_action(
            accepted,
            original,
            catalog[decision.action],
            config.epsilon,
            config.step_size,
            config.grid_size,
        )
        candidate = oracle.query(
            proposal,
            sample_id,
            (
                "residual-ranker-learned"
                if decision.used_residual
                else "residual-ranker-fallback"
            ),
            len(actions) + 1,
        )
        actions.append(decision.action)
        success = candidate.predicted_label != label
        previous_reward = recurrent_attack_reward(
            response.scores,
            candidate.scores,
            label,
            success,
            config,
        )
        current_margin = score_margin(response.scores, label)
        proposal_margin = score_margin(candidate.scores, label)
        transition = choose_attack_transition(
            accepted,
            proposal,
            current_margin=current_margin,
            proposal_margin=proposal_margin,
            success=success,
            rollback_on_non_improvement=True,
        )
        action_counts[decision.action] += 1.0
        action_values[decision.action] += (
            previous_reward - action_values[decision.action]
        ) / action_counts[decision.action]
        previous_action = decision.action
        if transition.accepted:
            accepted = transition.image
            response = candidate
        if success:
            success_query = oracle.calls

    delta = accepted - original
    after = policy.persistent_digest()
    if before != after:
        raise RuntimeError("D1 frozen deployment mutated persistent policy state")
    return FrozenEpisodeResult(
        sample_id,
        victim_id,
        family,
        True,
        success,
        success_query,
        oracle.calls,
        float(delta.abs().max()),
        float(delta.flatten().norm()),
        tuple(actions),
        before,
        after,
        tuple(oracle.trace_dicts()),
    )


__all__ = (
    "ResidualDecision",
    "ResidualRankerPolicy",
    "evaluate_residual_ranker_examples",
    "run_residual_ranker_episode",
    "score_greedy_action_order",
    "select_confidence_threshold",
)
