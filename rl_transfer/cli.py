import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim

from .config import AttackConfig
from .dqn import DQNAgent
from .models import SmallCNN, TargetCNN, freeze_model
from .protocols import run_transfer_protocols, train_policy
from .reproducibility import seed_everything


def synthetic_samples(count: int, seed: int) -> list[tuple[torch.Tensor, int]]:
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(count):
        image = torch.rand((3, 32, 32), generator=generator)
        # Keep the smoke fixture clean-correct for both intentionally biased victims.
        label = 0
        samples.append((image, label))
    return samples


def train_victim(model: nn.Module, samples: list[tuple[torch.Tensor, int]], epochs: int = 1) -> nn.Module:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for image, label in samples:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(image.unsqueeze(0)), torch.tensor([label]))
            loss.backward()
            optimizer.step()
    return freeze_model(model)


def run_smoke(output: Path, seed: int = 7) -> dict:
    seed_everything(seed)
    attack = AttackConfig(epsilon=8 / 255, step_size=2 / 255, grid_size=4, max_queries=8)
    source_data = synthetic_samples(24, seed)
    target_data = synthetic_samples(24, seed + 1)
    source_victim = train_victim(SmallCNN(), source_data)
    target_victim = train_victim(TargetCNN(), target_data)
    policy = DQNAgent(attack.state_dim, attack.action_dim, seed)
    train_policy(policy, source_victim, source_data, attack, episodes=12, seed=seed)
    result = run_transfer_protocols(policy, source_victim, target_victim, source_data, target_data[:12], target_data[12:], attack, adaptation_episodes=12, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
            return None
        return value
    output.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare frozen vs continual RL attack transfer")
    parser.add_argument("--output", type=Path, default=Path("output/rl_transfer/smoke.json"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run_smoke(args.output, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True, default=lambda value: None))


if __name__ == "__main__":
    main()
