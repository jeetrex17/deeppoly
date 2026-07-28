"""Inference-only controls for frozen Phase 2 recurrent policies."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch
from torch import nn

from .recurrent import PPOConfig, RecurrentAttackPolicy


class FrozenTemperaturePolicy(RecurrentAttackPolicy):
    """Apply temperature scaling without changing a checkpoint or its digest.

    Subclassing :class:`RecurrentAttackPolicy` is intentional. The frozen
    evaluation protocol recognizes recurrent policies by that type and supplies
    its episode-seeded ``random_draw`` to ``act``. The wrapped checkpoint is
    registered as the only submodule, so device discovery and ``parameters()``
    remain compatible with the existing evaluation code.
    """

    def __init__(
        self,
        checkpoint: RecurrentAttackPolicy,
        temperature: float = 1.0,
    ) -> None:
        if not isinstance(checkpoint, RecurrentAttackPolicy):
            raise TypeError("checkpoint must be a RecurrentAttackPolicy")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, Real)
        ):
            raise TypeError("temperature must be a real number")
        numeric_temperature = float(temperature)
        if not math.isfinite(numeric_temperature) or numeric_temperature <= 0:
            raise ValueError("temperature must be positive and finite")

        nn.Module.__init__(self)
        self._checkpoint = checkpoint
        self._temperature = numeric_temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def observation_dim(self) -> int:
        return self._checkpoint.observation_dim

    @property
    def action_dim(self) -> int:
        return self._checkpoint.action_dim

    @property
    def hidden_dim(self) -> int:
        return self._checkpoint.hidden_dim

    @property
    def actor_mode(self) -> str:
        return self._checkpoint.actor_mode

    @property
    def action_grid_size(self) -> int | None:
        return self._checkpoint.action_grid_size

    @property
    def config(self) -> PPOConfig:
        return self._checkpoint.config

    def initial_state(self, batch_size: int | None = None) -> torch.Tensor:
        return self._checkpoint.initial_state(batch_size=batch_size)

    def forward(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._checkpoint(observation, hidden)

    def persistent_digest(self) -> str:
        """Return the checkpoint digest, excluding the inference-only control."""

        return self._checkpoint.persistent_digest()

    def ppo_update(self, batch) -> dict[str, float]:
        del batch
        raise RuntimeError("frozen temperature policies cannot be trained")

    def ppo_update_sequences(self, weighted_sequences) -> dict[str, float]:
        del weighted_sequences
        raise RuntimeError("frozen temperature policies cannot be trained")

    def action_probabilities(
        self,
        observation: np.ndarray,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Return the temperature-scaled categorical probabilities."""

        observation_tensor = self._validated_inputs(observation, hidden)
        with torch.inference_mode():
            logits, _, _ = self(observation_tensor, hidden)
            probabilities = torch.softmax(logits / self.temperature, dim=-1)
        return probabilities.detach()

    def act(
        self,
        observation: np.ndarray,
        hidden: torch.Tensor,
        deterministic: bool = False,
        random_draw: float | None = None,
    ) -> tuple[int, torch.Tensor]:
        """Select an action using a caller-provided seeded draw when supplied."""

        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be boolean")
        numeric_draw = self._validated_random_draw(random_draw)
        observation_tensor = self._validated_inputs(observation, hidden)
        with torch.inference_mode():
            logits, _, next_hidden = self(observation_tensor, hidden)
            if deterministic:
                action = logits.argmax(dim=-1)
            elif numeric_draw is None:
                action = torch.distributions.Categorical(
                    logits=logits / self.temperature
                ).sample()
            else:
                probabilities = (
                    torch.softmax(logits / self.temperature, dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                action_index = int(
                    np.searchsorted(
                        np.cumsum(probabilities),
                        numeric_draw,
                        side="right",
                    )
                )
                action = torch.tensor(
                    min(action_index, self.action_dim - 1),
                    device=hidden.device,
                )
        return int(action), next_hidden.detach()

    def _validated_inputs(
        self,
        observation: np.ndarray,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(observation, np.ndarray):
            raise TypeError("observation must be a numpy array")
        if observation.shape != (self.observation_dim,):
            raise ValueError(
                "observation must have shape "
                f"({self.observation_dim},)"
            )
        if not np.issubdtype(observation.dtype, np.number):
            raise TypeError("observation must contain numeric values")
        if not np.isfinite(observation).all():
            raise ValueError("observation values must be finite")
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden state must be a torch tensor")
        if hidden.shape != (self.hidden_dim,):
            raise ValueError(
                f"hidden state must have shape ({self.hidden_dim},)"
            )
        if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
            raise ValueError("hidden state must contain finite floating-point values")

        parameter = next(self._checkpoint.parameters())
        if hidden.device != parameter.device or hidden.dtype != parameter.dtype:
            raise ValueError(
                "hidden state device and dtype must match the checkpoint"
            )
        return torch.as_tensor(
            observation,
            dtype=parameter.dtype,
            device=parameter.device,
        )

    @staticmethod
    def _validated_random_draw(random_draw: float | None) -> float | None:
        if random_draw is None:
            return None
        if isinstance(random_draw, bool) or not isinstance(random_draw, Real):
            raise TypeError("random_draw must be a real number")
        numeric_draw = float(random_draw)
        if not math.isfinite(numeric_draw) or not 0.0 <= numeric_draw < 1.0:
            raise ValueError("random_draw must be finite and in [0, 1)")
        return numeric_draw
