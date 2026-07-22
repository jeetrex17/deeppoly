import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn, optim

from .config import AttackConfig
from .dqn import DQNAgent
from .models import SmallCNN, TargetCNN, TinyPatchTransformer, freeze_model
from .protocols import run_transfer_protocols, train_policy
from .recurrent import RecurrentAttackPolicy
from .research_protocol import run_frozen_episode, train_population_policy
from .results import ResearchResultRow, write_jsonl
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


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        return None
    return value


def run_research_smoke(output: Path, seed: int = 7) -> dict:
    """Exercise the PDF's frozen recurrent boundary without claiming research validity."""
    seed_everything(seed)
    attack = AttackConfig(epsilon=8 / 255, step_size=2 / 255, grid_size=2, max_queries=5)
    samples = synthetic_samples(6, seed)
    source_victims = {
        "classical_cnn": ("small_cnn", freeze_model(SmallCNN())),
        "modern_cnn": ("target_cnn", freeze_model(TargetCNN())),
    }
    held_out = freeze_model(TinyPatchTransformer())
    policy = RecurrentAttackPolicy(observation_dim=8, action_dim=attack.action_dim, hidden_dim=32, seed=seed)
    training = train_population_policy(policy, source_victims, samples, attack, episodes=4, seed=seed)
    image, label = samples[-1]
    frozen = run_frozen_episode(policy, held_out, image, label, "smoke-target-0", "tiny_patch_transformer", "transformer", attack)
    result = {
        "manifest": {
            "schema_version": 1,
            "smoke": True,
            "research_valid": False,
            "primary_protocol": "T1-frozen-score-based",
            "t3_is_comparison_only": True,
            "source_families": sorted(source_victims),
            "outer_holdout_family": "transformer",
            "seed": seed,
            "epsilon": attack.epsilon,
            "total_query_budget": attack.max_queries,
        },
        "population_training": training,
        "frozen_t1": frozen.as_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".raw.jsonl")
    write_jsonl(raw_path, (ResearchResultRow(
        sample_id=frozen.sample_id,
        victim_id=frozen.victim_id,
        victim_family=frozen.family,
        method="groupdro_recurrent_ppo",
        threat_model="T1",
        seed=seed,
        query_budget=attack.max_queries,
        clean_correct=frozen.clean_correct,
        success=frozen.success,
        query_to_success=frozen.query_to_success,
        total_target_calls=frozen.total_target_calls,
        linf=frozen.linf,
        l2=frozen.l2,
        policy_digest=frozen.policy_digest_after,
        action_trace=frozen.actions,
    ),))
    result["manifest"]["raw_results"] = str(raw_path)
    output.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return _json_safe(result)


def validate_full_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    required_families = {"classical_cnn", "modern_cnn", "transformer", "hierarchical_transformer"}
    required_baselines = {"dqn", "single_source_ppo", "naive_pooled_ppo", "groupdro_ppo", "fixed", "random", "greedy", "square", "simba_dct"}
    if config.get("schema_version") != 1 or config.get("research_valid") is not True:
        raise ValueError("full research config must use schema 1 and be marked research-valid")
    if config.get("dataset") != "ImageNet-1K":
        raise ValueError("full research config must use ImageNet-1K")
    if config.get("primary_threat_model") != "T1-frozen-score-based":
        raise ValueError("full research config must use frozen score-based T1")
    holdout_families = tuple(config.get("outer_holdout_families", ()))
    if len(holdout_families) != len(required_families) or set(holdout_families) != required_families:
        raise ValueError("full config must hold out every major architecture family")
    if config.get("inner_source_family_validation") is not True:
        raise ValueError("full config requires nested source-family validation")
    epsilon_values = sorted(config.get("epsilon_values", ()))
    expected_epsilon = (4 / 255, 8 / 255)
    if len(epsilon_values) != 2 or not all(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12) for actual, expected in zip(epsilon_values, expected_epsilon)):
        raise ValueError("full config requires epsilon values 4/255 and 8/255")
    seeds = tuple(config.get("seeds", ()))
    if len(set(seeds)) < 5 or set(config.get("query_budgets", ())) != {0, 25, 100, 500}:
        raise ValueError("full config requires five seeds and budgets 0/25/100/500")
    if config.get("stage_images") != 1000 or config.get("confirmation_images") != 5000:
        raise ValueError("full config requires 1,000 development and 5,000 confirmation images")
    if config.get("robust_stress_models", 0) < 2:
        raise ValueError("full config requires at least two robust stress models")
    if not required_baselines.issubset(set(config.get("required_baselines", ()))):
        raise ValueError("full config is missing one or more matched baselines")
    if config.get("execute_in_ci") is not False:
        raise ValueError("paper-scale execution must remain disabled in CI")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-victim RL attack research harness")
    parser.add_argument("command", nargs="?", choices=("legacy-smoke", "research-smoke", "validate-full"), default="legacy-smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/rl_transfer/imagenet1k_lofo.json"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.command == "validate-full":
        result = validate_full_config(args.config)
    elif args.command == "research-smoke":
        result = run_research_smoke(args.output or Path("output/rl_transfer/research_smoke.json"), args.seed)
    else:
        result = run_smoke(args.output or Path("output/rl_transfer/smoke.json"), args.seed)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True, default=lambda value: None))


if __name__ == "__main__":
    main()
