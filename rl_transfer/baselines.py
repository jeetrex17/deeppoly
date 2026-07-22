import hashlib
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
