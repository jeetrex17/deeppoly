"""Reward functions shared by the classic and recurrent attack environments."""

import math

import torch

from .config import AttackConfig


def confidence_margin(
    true_probability: float,
    rival_probability: float,
) -> float:
    """Return p(true) - max(p(other)); lower values are better for the attacker."""
    margin = float(true_probability) - float(rival_probability)
    if not math.isfinite(margin):
        raise ValueError("confidence probabilities must be finite")
    return margin


def score_margin(scores: torch.Tensor, label: int) -> float:
    """Compute the true-label confidence margin from a class score vector."""
    if scores.ndim != 1 or scores.numel() < 2:
        raise ValueError("scores must be a one-dimensional vector with at least two classes")
    if not 0 <= label < scores.numel():
        raise ValueError("label is outside the score vector")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    rival = torch.cat((scores[:label], scores[label + 1 :])).max()
    return confidence_margin(float(scores[label]), float(rival))


def dense_margin_reward(
    previous_margin: float,
    current_margin: float,
    success: bool,
    config: AttackConfig,
) -> float:
    """Reward one-step margin reduction, then add success bonus and query cost.

    Using a one-step difference makes the shaping term telescope over an episode:
    repeatedly receiving reward requires continued progress rather than merely
    remaining below the clean confidence. The per-action query cost also makes an
    earlier successful attack preferable to an otherwise identical later one.
    """
    previous_margin = float(previous_margin)
    current_margin = float(current_margin)
    if not math.isfinite(previous_margin) or not math.isfinite(current_margin):
        raise ValueError("confidence margins must be finite")
    reduction = previous_margin - current_margin
    return float(
        config.margin_reward_scale * reduction
        + (config.terminal_success_bonus if success else 0.0)
        - config.query_penalty
    )


def recurrent_attack_reward(
    previous_scores: torch.Tensor,
    current_scores: torch.Tensor,
    label: int,
    success: bool,
    config: AttackConfig,
) -> float:
    """Compute a PPO attack reward, retaining the original rule on request."""
    if config.reward_mode == "legacy":
        return (
            config.terminal_success_bonus
            if success
            else float(previous_scores[label] - current_scores[label]) - config.query_penalty
        )
    return dense_margin_reward(
        score_margin(previous_scores, label),
        score_margin(current_scores, label),
        success,
        config,
    )


def patch_environment_reward(
    previous_true_probability: float,
    previous_rival_probability: float,
    current_true_probability: float,
    current_rival_probability: float,
    success: bool,
    step: int,
    config: AttackConfig,
) -> float:
    """Compute a classic patch-environment reward with a legacy compatibility mode."""
    if config.reward_mode == "legacy":
        return (
            config.terminal_success_bonus - 0.2 * step
            if success
            else -config.query_penalty - current_true_probability
        )
    return dense_margin_reward(
        confidence_margin(previous_true_probability, previous_rival_probability),
        confidence_margin(current_true_probability, current_rival_probability),
        success,
        config,
    )
