from collections import deque
from copy import deepcopy
from dataclasses import asdict
import random
from typing import Any, NamedTuple
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch import nn, optim
from .config import DQNConfig


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, hidden_dims: tuple[int, ...] | None = None) -> None:
        super().__init__()
        dims = hidden_dims or (hidden, hidden)
        layers: list[nn.Module] = []
        previous = state_dim
        for width in dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, action_dim))
        self.layers = nn.Sequential(*layers)
        self.net = self.layers

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, seed: int = 0, batch_size: int = 32, config: DQNConfig | None = None, device: str = "cpu") -> None:
        torch.manual_seed(seed)
        self.config = config or DQNConfig(batch_size=batch_size, min_replay_size=batch_size)
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.batch_size = self.config.batch_size
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = self.config.gamma, self.config.epsilon_start, self.config.epsilon_end, self.config.epsilon_decay
        self.online = QNetwork(state_dim, action_dim, hidden_dims=self.config.hidden_dims).to(self.device)
        self.target = deepcopy(self.online).eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay: deque[Transition] = deque(maxlen=self.config.replay_capacity)
        self.updates = 0
        self.rng = random.Random(seed)

    def act(self, state: np.ndarray, greedy: bool = False, evaluate: bool = False) -> int:
        if not greedy and not evaluate and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_dim)
        with torch.inference_mode():
            return int(self.online(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)).argmax(1))

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.append(Transition(state.copy(), int(action), float(reward), next_state.copy(), bool(done)))

    def observe(self, transition: Transition) -> None:
        self.replay.append(transition)

    def learn(self) -> float | None:
        if len(self.replay) < max(self.batch_size, self.config.min_replay_size):
            return None
        batch = self.rng.sample(list(self.replay), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_tensor = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        current = self.online(state_tensor).gather(1, action_tensor).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_tensor).argmax(1)
            next_values = self.target(next_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = reward_tensor + self.gamma * next_values * (1 - done_tensor)
        loss = nn.functional.smooth_l1_loss(current, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_sync_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.detach())

    def checkpoint(self) -> dict[str, Any]:
        replay = [
            {
                "state": torch.as_tensor(transition.state.copy()),
                "action": transition.action,
                "reward": transition.reward,
                "next_state": torch.as_tensor(transition.next_state.copy()),
                "done": transition.done,
            }
            for transition in self.replay
        ]
        return {
            "schema_version": 1,
            "state_dim": self.online.layers[0].in_features,
            "action_dim": self.action_dim,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": replay,
            "epsilon": self.epsilon,
            "updates": self.updates,
            "rng_state": self.rng.getstate(),
            "config": asdict(self.config),
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("schema_version") != 1:
            raise ValueError("unsupported checkpoint schema")
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        replay = (
            Transition(
                item["state"].detach().cpu().numpy().copy(),
                int(item["action"]),
                float(item["reward"]),
                item["next_state"].detach().cpu().numpy().copy(),
                bool(item["done"]),
            )
            for item in checkpoint["replay"]
        )
        self.replay = deque(replay, maxlen=self.config.replay_capacity)
        self.epsilon, self.updates = checkpoint["epsilon"], checkpoint["updates"]
        self.rng.setstate(checkpoint["rng_state"])

    def clone(self) -> "DQNAgent":
        result = DQNAgent(self.online.layers[0].in_features, self.action_dim, config=self.config)
        result.load_checkpoint(deepcopy(self.checkpoint()))
        return result

    @property
    def update_count(self) -> int:
        return self.updates

    def _digest(self, module: nn.Module) -> str:
        hasher = hashlib.sha256()
        for tensor in module.state_dict().values():
            hasher.update(tensor.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

    def policy_digest(self) -> str:
        return self._digest(self.online)

    def target_digest(self) -> str:
        return self._digest(self.target)

    def training_digest(self) -> str:
        return hashlib.sha256(repr((self.policy_digest(), self.target_digest(), self.epsilon, self.updates, len(self.replay))).encode()).hexdigest()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        torch.save(self.checkpoint(), path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "DQNAgent":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config_payload = checkpoint.get("config")
        if not isinstance(config_payload, dict):
            raise ValueError("checkpoint config must use the safe dictionary schema")
        result = cls(
            int(checkpoint["state_dim"]),
            int(checkpoint["action_dim"]),
            config=DQNConfig(**config_payload),
            device=device,
        )
        result.load_checkpoint(checkpoint)
        return result
