import hashlib
import math
import random

import numpy as np


class FixedActionPolicy:
    def __init__(self, action: int, action_dim: int) -> None:
        if not 0 <= action < action_dim:
            raise ValueError("fixed action is outside the action catalog")
        self.action = action
        self.action_dim = action_dim

    def initial_state(self):
        return None

    def persistent_digest(self) -> str:
        return hashlib.sha256(f"fixed:{self.action}:{self.action_dim}".encode()).hexdigest()

    def act(self, observation: np.ndarray, hidden=None, deterministic: bool = True):
        return self.action, hidden


class RandomActionPolicy:
    def __init__(self, action_dim: int, seed: int) -> None:
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.seed = seed
        self.rng = random.Random(seed)

    def initial_state(self):
        return None

    def persistent_digest(self) -> str:
        return hashlib.sha256(f"random:{self.seed}:{self.action_dim}".encode()).hexdigest()

    def act(self, observation: np.ndarray, hidden=None, deterministic: bool = False):
        return self.rng.randrange(self.action_dim), hidden


class BanditActionPolicy:
    """Query-matched score-bandit control with episode-local adaptation.

    The policy receives exactly the same observation stream as the recurrent
    attacker.  It spends no extra victim queries: after observing the reward
    from its previous action, it updates an upper-confidence-bound estimate and
    selects the next action.  All learned state is returned as ephemeral hidden
    state, so the persistent policy digest remains frozen between episodes.
    """

    def __init__(
        self,
        action_dim: int,
        seed: int,
        exploration: float = 0.75,
        warmup_actions: int = 6,
    ) -> None:
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError("exploration must be finite and non-negative")
        if (
            not isinstance(warmup_actions, int)
            or isinstance(warmup_actions, bool)
            or warmup_actions < 1
        ):
            raise ValueError("warmup_actions must be a positive integer")
        self.action_dim = action_dim
        self.seed = seed
        self.exploration = exploration
        self.warmup_actions = min(action_dim, warmup_actions)

    def initial_state(self) -> dict[str, object]:
        return {
            "counts": [0] * self.action_dim,
            "values": [0.0] * self.action_dim,
            "last_action": None,
            "steps": 0,
            "candidates": None,
        }

    def persistent_digest(self) -> str:
        payload = (
            f"bandit:{self.seed}:{self.action_dim}:{self.exploration:.12g}:"
            f"{self.warmup_actions}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def act(
        self,
        observation: np.ndarray,
        hidden: dict[str, object] | None = None,
        deterministic: bool = True,
    ) -> tuple[int, dict[str, object]]:
        del deterministic
        if observation.ndim != 1 or observation.size < 7:
            raise ValueError("bandit policy requires the standard attack observation")
        state = self.initial_state() if hidden is None else {
            "counts": list(hidden["counts"]),
            "values": list(hidden["values"]),
            "last_action": hidden["last_action"],
            "steps": int(hidden["steps"]),
            "candidates": hidden["candidates"],
        }
        counts = state["counts"]
        values = state["values"]
        last_action = state["last_action"]
        if last_action is not None:
            action = int(last_action)
            reward_signal = float(observation[6])
            counts[action] += 1
            values[action] += (reward_signal - values[action]) / counts[action]
        if state["candidates"] is None:
            observation_key = hashlib.sha256(observation.tobytes()).digest()
            episode_seed = self.seed ^ int.from_bytes(observation_key[:8], "big")
            order = list(range(self.action_dim))
            random.Random(episode_seed).shuffle(order)
            state["candidates"] = tuple(order[: self.warmup_actions])
        candidates = state["candidates"]
        untried = next((action for action in candidates if counts[action] == 0), None)
        if untried is not None:
            selected = untried
        else:
            log_steps = math.log(max(2, int(state["steps"]) + 1))
            selected = max(
                candidates,
                key=lambda action: values[action]
                + self.exploration * math.sqrt(log_steps / counts[action]),
            )
        state["last_action"] = selected
        state["steps"] = int(state["steps"]) + 1
        return selected, state
