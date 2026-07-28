from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    gradient_clip_norm: float = 0.5
    update_epochs: int = 4

    def __post_init__(self) -> None:
        scalar_values = (
            self.learning_rate,
            self.clip_ratio,
            self.value_weight,
            self.entropy_weight,
            self.gradient_clip_norm,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("PPO configuration values must be finite")
        if self.learning_rate <= 0 or not 0 < self.clip_ratio <= 1:
            raise ValueError("PPO learning rate and clip ratio must be positive")
        if self.value_weight < 0 or self.entropy_weight < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("PPO loss weights and gradient clip must be valid")
        if not 1 <= self.update_epochs <= 32:
            raise ValueError("PPO update epochs must be between 1 and 32")


@dataclass(frozen=True)
class PPOBatch:
    observations: torch.Tensor
    hidden_states: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass(frozen=True)
class PPOSequence:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ActionConditionedActor(nn.Module):
    """Shared scorer over patch geometry, RGB channel, and update sign."""

    feature_dim = 6

    def __init__(self, hidden_dim: int, grid_size: int) -> None:
        super().__init__()
        if (
            not isinstance(grid_size, int)
            or isinstance(grid_size, bool)
            or grid_size < 1
        ):
            raise ValueError("action-conditioned actor requires a positive grid size")
        features: list[tuple[float, ...]] = []
        denominator = max(1, grid_size - 1)
        for patch_index in range(grid_size * grid_size):
            row, column = divmod(patch_index, grid_size)
            row_position = -1.0 + 2.0 * row / denominator
            column_position = -1.0 + 2.0 * column / denominator
            for channel in range(3):
                channel_features = tuple(
                    1.0 if index == channel else 0.0
                    for index in range(3)
                )
                for sign in (-1, 1):
                    features.append(
                        (
                            row_position,
                            column_position,
                            *channel_features,
                            float(sign),
                        )
                    )
        self.register_buffer(
            "action_features",
            torch.tensor(features, dtype=torch.float32),
            persistent=True,
        )
        self.state_projection = nn.Linear(hidden_dim, hidden_dim)
        self.action_projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_bias = nn.Linear(self.feature_dim, 1, bias=False)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query = self.state_projection(hidden)
        action_embeddings = self.action_projection(self.action_features)
        logits = query @ action_embeddings.transpose(0, 1) / self.scale
        return logits + self.action_bias(self.action_features).squeeze(-1)


class RecurrentAttackPolicy(nn.Module):
    """GRU actor-critic whose hidden state adapts without changing parameters."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        seed: int = 0,
        config: PPOConfig | None = None,
        *,
        actor_mode: str = "flat",
        action_grid_size: int | None = None,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim, hidden_dim) < 1:
            raise ValueError("network dimensions must be positive")
        if actor_mode not in {"flat", "action_conditioned"}:
            raise ValueError("actor_mode must be 'flat' or 'action_conditioned'")
        if actor_mode == "action_conditioned":
            if (
                not isinstance(action_grid_size, int)
                or isinstance(action_grid_size, bool)
                or action_grid_size < 1
                or action_dim != action_grid_size * action_grid_size * 3 * 2
            ):
                raise ValueError(
                    "action-conditioned actor dimensions must match the patch catalog"
                )
        elif action_grid_size is not None:
            raise ValueError("flat actor does not accept action_grid_size")
        torch.manual_seed(seed)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.actor_mode = actor_mode
        self.action_grid_size = action_grid_size
        self.config = config or PPOConfig()
        self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh())
        self.memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor = (
            nn.Linear(hidden_dim, action_dim)
            if actor_mode == "flat"
            else ActionConditionedActor(hidden_dim, action_grid_size)
        )
        self.critic = nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def initial_state(self, batch_size: int | None = None) -> torch.Tensor:
        shape = (self.hidden_dim,) if batch_size is None else (batch_size, self.hidden_dim)
        return torch.zeros(shape, dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observation)
        next_hidden = self.memory(encoded, hidden)
        return self.actor(next_hidden), self.critic(next_hidden).squeeze(-1), next_hidden

    def act(
        self,
        observation: np.ndarray,
        hidden: torch.Tensor,
        deterministic: bool = False,
        random_draw: float | None = None,
    ) -> tuple[int, torch.Tensor]:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=hidden.device)
        with torch.inference_mode():
            logits, _, next_hidden = self(observation_tensor, hidden)
            if deterministic:
                action = logits.argmax(-1)
            elif random_draw is None:
                action = torch.distributions.Categorical(logits=logits).sample()
            else:
                if not 0.0 <= random_draw < 1.0:
                    raise ValueError("random_draw must be in [0, 1)")
                probabilities = logits.softmax(-1).detach().cpu().numpy()
                action_index = int(
                    np.searchsorted(
                        np.cumsum(probabilities),
                        random_draw,
                        side="right",
                    )
                )
                action = torch.tensor(
                    min(action_index, self.action_dim - 1),
                    device=hidden.device,
                )
        return int(action), next_hidden.detach()

    def persistent_digest(self) -> str:
        hasher = hashlib.sha256()
        def update(value) -> None:
            if isinstance(value, torch.Tensor):
                hasher.update(str(value.dtype).encode("utf-8"))
                hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(value, dict):
                for key in sorted(value, key=str):
                    hasher.update(str(key).encode("utf-8"))
                    update(value[key])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    update(item)
            else:
                hasher.update(repr(value).encode("utf-8"))
        update(self.state_dict())
        update(self.optimizer.state_dict())
        hasher.update(json.dumps(asdict(self.config), sort_keys=True).encode("utf-8"))
        hasher.update(self.actor_mode.encode("utf-8"))
        hasher.update(repr(self.action_grid_size).encode("utf-8"))
        return hasher.hexdigest()

    def ppo_update(self, batch: PPOBatch) -> dict[str, float]:
        logits, values, _ = self(batch.observations, batch.hidden_states)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(batch.actions)
        ratios = (log_probabilities - batch.old_log_probabilities).exp()
        unclipped = ratios * batch.advantages
        clipped = ratios.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch.advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = nn.functional.mse_loss(values, batch.returns)
        entropy = distribution.entropy().mean()
        loss = policy_loss + self.config.value_weight * value_loss - self.config.entropy_weight * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        return {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach())}

    def ppo_update_sequences(self, weighted_sequences: list[tuple[PPOSequence, float]]) -> dict[str, float]:
        if not weighted_sequences:
            raise ValueError("at least one sequence is required")
        weights = torch.tensor(tuple(weight for _, weight in weighted_sequences))
        if not torch.isfinite(weights).all() or bool((weights < 0).any()) or float(weights.sum()) <= 0:
            raise ValueError("sequence weights must be finite, non-negative, and non-zero")
        advantage_values = torch.cat(
            tuple(sequence.advantages.detach().flatten() for sequence, _ in weighted_sequences)
        )
        advantage_mean = advantage_values.mean()
        advantage_scale = advantage_values.std(unbiased=False).clamp_min(1e-8)
        normalized_advantages = tuple(
            (sequence.advantages.detach() - advantage_mean) / advantage_scale
            for sequence, _ in weighted_sequences
        )
        final_metrics: dict[str, float] = {}
        for _ in range(self.config.update_epochs):
            policy_terms, value_terms, entropy_terms = [], [], []
            for (sequence, weight), advantages in zip(weighted_sequences, normalized_advantages):
                hidden = self.initial_state()
                logits_steps, value_steps = [], []
                for observation in sequence.observations:
                    logits, value, hidden = self(observation, hidden)
                    logits_steps.append(logits)
                    value_steps.append(value)
                logits = torch.stack(logits_steps)
                values = torch.stack(value_steps)
                distribution = torch.distributions.Categorical(logits=logits)
                log_probabilities = distribution.log_prob(sequence.actions)
                ratios = (log_probabilities - sequence.old_log_probabilities.detach()).exp()
                unclipped = ratios * advantages
                clipped = ratios.clamp(
                    1 - self.config.clip_ratio,
                    1 + self.config.clip_ratio,
                ) * advantages
                policy_terms.append(-torch.minimum(unclipped, clipped).mean() * weight)
                value_terms.append(nn.functional.mse_loss(values, sequence.returns) * weight)
                entropy_terms.append(distribution.entropy().mean() * weight)
            policy_loss = torch.stack(policy_terms).sum()
            value_loss = torch.stack(value_terms).sum()
            entropy = torch.stack(entropy_terms).sum()
            loss = (
                policy_loss
                + self.config.value_weight * value_loss
                - self.config.entropy_weight * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            final_metrics = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "update_epochs": float(self.config.update_epochs),
                "advantage_mean": float(advantage_mean.detach()),
                "advantage_std": float(advantage_scale.detach()),
            }
        return final_metrics
