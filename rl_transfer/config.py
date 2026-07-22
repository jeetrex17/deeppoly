from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AttackConfig:
    epsilon: float = 8 / 255
    step_size: float = 2 / 255
    grid_size: int = 4
    max_queries: int = 20

    def __post_init__(self) -> None:
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        if not 0 < self.step_size <= self.epsilon:
            raise ValueError("step_size must be in (0, epsilon]")
        if self.grid_size < 1:
            raise ValueError("grid_size must be positive")
        if self.max_queries < 1:
            raise ValueError("max_queries must be positive")

    @property
    def action_dim(self) -> int:
        return self.grid_size * self.grid_size * 3 * 2

    @property
    def state_dim(self) -> int:
        return self.grid_size * self.grid_size + 6


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    device: str = "cpu"
    attack: AttackConfig = AttackConfig()
    adaptation_episodes: int = 25

    def __post_init__(self) -> None:
        if self.adaptation_episodes < 0:
            raise ValueError("adaptation_episodes cannot be negative")


@dataclass(frozen=True)
class DQNConfig:
    hidden_dims: tuple[int, ...] = (128, 128)
    gamma: float = 0.98
    batch_size: int = 32
    replay_capacity: int = 20_000
    min_replay_size: int = 32
    target_sync_interval: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    learning_rate: float = 5e-4
    gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        try:
            hidden_dims = tuple(self.hidden_dims)
        except TypeError as error:
            raise ValueError("hidden_dims must be a sequence of positive integers") from error
        object.__setattr__(self, "hidden_dims", hidden_dims)

        if not hidden_dims or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integers")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        replay_sizes = (self.batch_size, self.min_replay_size, self.replay_capacity)
        if any(isinstance(size, bool) or not isinstance(size, int) for size in replay_sizes):
            raise ValueError("replay sizes must be integers")
        if not 1 <= self.batch_size <= self.min_replay_size <= self.replay_capacity:
            raise ValueError(
                "replay sizes must satisfy batch_size <= min_replay_size <= replay_capacity"
            )
        if (
            isinstance(self.target_sync_interval, bool)
            or not isinstance(self.target_sync_interval, int)
            or self.target_sync_interval < 1
        ):
            raise ValueError("target_sync_interval must be a positive integer")
        epsilon_values = (self.epsilon_start, self.epsilon_end, self.epsilon_decay)
        if not all(math.isfinite(value) for value in epsilon_values):
            raise ValueError("epsilon values must be finite")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon schedule must satisfy 0 <= end <= start <= 1")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive and finite")
