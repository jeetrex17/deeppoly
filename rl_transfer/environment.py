from typing import Any

import numpy as np
import torch

from .config import AttackConfig


class EpisodeFinishedError(RuntimeError):
    pass


class IneligibleSampleError(ValueError):
    pass


class StepInfo(dict):
    @property
    def success(self) -> bool:
        return bool(self["success"])

    @property
    def attack_queries(self) -> int:
        return int(self["queries"] - 1)


class PatchAttackEnv:
    """Score-only, untargeted patch MDP with a raw-pixel L-infinity budget."""

    def __init__(self, victim: torch.nn.Module, image: torch.Tensor | AttackConfig, label: int | None = None, config: AttackConfig | None = None, seed: int = 0) -> None:
        if isinstance(image, AttackConfig):
            config, image, label = image, None, None
        if config is None:
            raise ValueError("config is required")
        self.victim = victim.eval()
        self.config = config
        self.generator = torch.Generator().manual_seed(seed)
        if image is None:
            self.original = None
            return
        if image.ndim != 3:
            raise ValueError("image must have shape [C, H, W]")
        if image.shape[1] % config.grid_size or image.shape[2] % config.grid_size:
            raise ValueError("image dimensions must divide evenly into grid_size")
        self.original = image.detach().clone().float().clamp(0, 1)
        self.label = int(label)
        self.patch_h = image.shape[1] // config.grid_size
        self.patch_w = image.shape[2] // config.grid_size
        self.patch_count = config.grid_size ** 2
        self.reset()

    def _query(self, image: torch.Tensor) -> tuple[int, float, float]:
        with torch.inference_mode():
            probabilities = self.victim(image.unsqueeze(0)).softmax(dim=1)[0]
        prediction = int(probabilities.argmax())
        true_probability = float(probabilities[self.label])
        rival_probability = float(torch.cat((probabilities[:self.label], probabilities[self.label + 1:])).max())
        return prediction, true_probability, rival_probability

    def reset(self, image: torch.Tensor | None = None, label: int | None = None, sample_id: str | None = None) -> np.ndarray:
        if image is not None:
            self.original = image.detach().clone().float().clamp(0, 1)
            if self.original.ndim != 3 or self.original.shape[1] % self.config.grid_size or self.original.shape[2] % self.config.grid_size:
                raise ValueError("image dimensions must divide evenly into grid_size")
            self.patch_h = self.original.shape[1] // self.config.grid_size
            self.patch_w = self.original.shape[2] // self.config.grid_size
            self.patch_count = self.config.grid_size ** 2
        if label is not None:
            self.label = int(label)
        if self.original is None or not hasattr(self, "label"):
            raise ValueError("reset requires image and label")
        self.adv = self.original.clone()
        self.touched = np.zeros(self.patch_count, dtype=np.float32)
        self.steps = 0
        self.queries = 1
        prediction, self.true_probability, self.rival_probability = self._query(self.adv)
        self.clean_true_probability = self.true_probability
        self.clean_prediction = prediction
        if prediction != self.label:
            raise IneligibleSampleError("sample is not clean-correct")
        self.done = False
        return self._state()

    def _state(self) -> np.ndarray:
        margin = self.true_probability - self.rival_probability
        remaining = 1.0 - self.steps / self.config.max_queries
        return np.concatenate((self.touched, np.array([self.true_probability, self.rival_probability, margin, remaining, float(self.clean_prediction == self.label), self.steps / self.config.max_queries], dtype=np.float32))).astype(np.float32)

    def _coordinates(self, action: int) -> tuple[int, int, int]:
        if not 0 <= action < self.config.action_dim:
            raise ValueError("action is outside the action space")
        sign = -1 if action % 2 == 0 else 1
        value = action // 2
        channel = value % 3
        patch = value // 3
        row, col = divmod(patch, self.config.grid_size)
        return row, col, channel * sign

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise EpisodeFinishedError("episode has terminated; call reset")
        row, col, _ = self._coordinates(int(action))
        # Decode channel/sign without exposing model gradients to the policy.
        sign = -1 if action % 2 == 0 else 1
        channel = (action // 2) % 3
        y0, y1 = row * self.patch_h, (row + 1) * self.patch_h
        x0, x1 = col * self.patch_w, (col + 1) * self.patch_w
        candidate = self.adv.clone()
        candidate[channel, y0:y1, x0:x1] += sign * self.config.step_size
        self.adv = torch.maximum(torch.minimum(candidate, self.original + self.config.epsilon), self.original - self.config.epsilon).clamp(0, 1)
        self.touched[row * self.config.grid_size + col] = 1.0
        self.steps += 1
        self.queries += 1
        prediction, self.true_probability, self.rival_probability = self._query(self.adv)
        success = prediction != self.label
        self.done = success or self.steps >= self.config.max_queries
        reward = (10.0 - 0.2 * self.steps) if success else -0.05 - self.true_probability
        return self._state(), float(reward), self.done, StepInfo(success=success, prediction=prediction, queries=self.queries, confidence_drop=self.clean_true_probability - self.true_probability)

    @property
    def adversarial_image(self) -> torch.Tensor:
        return self.adv
