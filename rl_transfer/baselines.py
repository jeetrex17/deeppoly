import random

import numpy as np


class FixedActionPolicy:
    def __init__(self, action: int) -> None:
        self.action = action

    def act(self, observation: np.ndarray, hidden=None, deterministic: bool = True):
        return self.action, hidden


class RandomActionPolicy:
    def __init__(self, action_dim: int, seed: int) -> None:
        self.action_dim = action_dim
        self.rng = random.Random(seed)

    def act(self, observation: np.ndarray, hidden=None, deterministic: bool = False):
        return self.rng.randrange(self.action_dim), hidden
