from dataclasses import asdict, dataclass
import hashlib
import json

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


class RecurrentAttackPolicy(nn.Module):
    """GRU actor-critic whose hidden state adapts without changing parameters."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128, seed: int = 0, config: PPOConfig | None = None) -> None:
        super().__init__()
        if min(observation_dim, action_dim, hidden_dim) < 1:
            raise ValueError("network dimensions must be positive")
        torch.manual_seed(seed)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.config = config or PPOConfig()
        self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh())
        self.memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def initial_state(self, batch_size: int | None = None) -> torch.Tensor:
        shape = (self.hidden_dim,) if batch_size is None else (batch_size, self.hidden_dim)
        return torch.zeros(shape, dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observation)
        next_hidden = self.memory(encoded, hidden)
        return self.actor(next_hidden), self.critic(next_hidden).squeeze(-1), next_hidden

    def act(self, observation: np.ndarray, hidden: torch.Tensor, deterministic: bool = False) -> tuple[int, torch.Tensor]:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=hidden.device)
        with torch.inference_mode():
            logits, _, next_hidden = self(observation_tensor, hidden)
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else distribution.sample()
        return int(action), next_hidden.detach()

    def persistent_digest(self) -> str:
        hasher = hashlib.sha256()
        def update(value) -> None:
            if isinstance(value, torch.Tensor):
                hasher.update(str(value.dtype).encode("utf-8"))
                hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(value, dict):
                for key in sorted(value, key=str):
                    hasher.update(str(key).encode("utf-8")); update(value[key])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    update(item)
            else:
                hasher.update(repr(value).encode("utf-8"))
        update(self.state_dict())
        update(self.optimizer.state_dict())
        hasher.update(json.dumps(asdict(self.config), sort_keys=True).encode("utf-8"))
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
        policy_terms, value_terms, entropy_terms = [], [], []
        for sequence, weight in weighted_sequences:
            hidden = self.initial_state()
            logits_steps, value_steps = [], []
            for observation in sequence.observations:
                logits, value, hidden = self(observation, hidden)
                logits_steps.append(logits); value_steps.append(value)
            logits = torch.stack(logits_steps)
            values = torch.stack(value_steps)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probabilities = distribution.log_prob(sequence.actions)
            ratios = (log_probabilities - sequence.old_log_probabilities).exp()
            unclipped = ratios * sequence.advantages
            clipped = ratios.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * sequence.advantages
            policy_terms.append(-torch.minimum(unclipped, clipped).mean() * weight)
            value_terms.append(nn.functional.mse_loss(values, sequence.returns) * weight)
            entropy_terms.append(distribution.entropy().mean() * weight)
        policy_loss = torch.stack(policy_terms).sum()
        value_loss = torch.stack(value_terms).sum()
        entropy = torch.stack(entropy_terms).sum()
        loss = policy_loss + self.config.value_weight * value_loss - self.config.entropy_weight * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        return {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach())}
