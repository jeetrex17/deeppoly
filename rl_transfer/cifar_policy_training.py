"""Resumable hybrid and component-ablation policy training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from .artifacts import (
    load_recurrent_checkpoint,
    save_recurrent_checkpoint,
    sha256_file,
)
from .cifar_config import MacPilotConfig
from .config import AttackConfig
from .imitation import (
    behavior_clone_policy,
    collect_best_of_k_demonstrations,
    collect_gradient_demonstrations,
    evaluate_behavior_clone_policy,
)
from .paths import resolve_descendant
from .recurrent import PPOConfig, RecurrentAttackPolicy
from .research_protocol import train_population_policy


VictimPopulation = Mapping[str, Sequence[tuple[str, nn.Module]]]
Samples = Sequence[tuple[torch.Tensor, int]]
SOFT_GRADIENT_BC_ACTION_CONDITIONED_PPO_METHOD = (
    "soft_gradient_bc_action_conditioned_groupdro_ppo"
)


def main_policy_method_id(config: MacPilotConfig) -> str:
    """Return the stable evaluation/checkpoint identity for the main policy."""

    if (
        config.behavior_cloning_episodes > 0
        and config.behavior_cloning_teacher == "gradient"
        and config.behavior_cloning_soft_temperature is not None
        and config.policy_actor_mode == "action_conditioned"
    ):
        return SOFT_GRADIENT_BC_ACTION_CONDITIONED_PPO_METHOD
    if config.train_ablation_policies:
        return "gradient_bc_groupdro_ppo"
    return "groupdro_recurrent_ppo"


@dataclass(frozen=True)
class PolicyTrainingBundle:
    main: RecurrentAttackPolicy
    bc_only: RecurrentAttackPolicy | None
    ppo_only: RecurrentAttackPolicy | None
    training: dict[str, object]
    main_resumed: bool
    checkpoints: dict[str, dict[str, object]]


def _new_policy(
    config: MacPilotConfig,
    attack: AttackConfig,
    device: torch.device,
) -> RecurrentAttackPolicy:
    return RecurrentAttackPolicy(
        observation_dim=attack.recurrent_observation_dim,
        action_dim=attack.action_dim,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
        config=PPOConfig(
            learning_rate=config.policy_learning_rate,
            entropy_weight=config.policy_entropy_weight,
            update_epochs=config.policy_update_epochs,
        ),
        actor_mode=config.policy_actor_mode,
        action_grid_size=(
            config.grid_size
            if config.policy_actor_mode == "action_conditioned"
            else None
        ),
    ).to(device)


def _load_checked(
    path: Path,
    device: torch.device,
    fingerprint: str,
    config: MacPilotConfig,
    attack: AttackConfig,
) -> tuple[RecurrentAttackPolicy, dict[str, object]]:
    policy, metadata = load_recurrent_checkpoint(
        path,
        device,
        expected_observation_dim=attack.recurrent_observation_dim,
        expected_action_dim=attack.action_dim,
        expected_hidden_dim=config.hidden_dim,
        expected_actor_mode=config.policy_actor_mode,
    )
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("policy checkpoint fingerprint mismatch")
    return policy, metadata


def _collect_and_fit_bc(
    policy: RecurrentAttackPolicy,
    source_victims: VictimPopulation,
    policy_samples: Samples,
    validation_samples: Samples,
    attack: AttackConfig,
    config: MacPilotConfig,
) -> dict[str, object]:
    started = time.monotonic()
    collector = (
        collect_gradient_demonstrations
        if config.behavior_cloning_teacher == "gradient"
        else collect_best_of_k_demonstrations
    )
    shared = {
        "decisions": config.behavior_cloning_steps,
    }
    if config.behavior_cloning_teacher == "gradient":
        shared["soft_temperature"] = (
            config.behavior_cloning_soft_temperature
        )
    else:
        shared["candidates"] = config.behavior_cloning_candidates
    demonstrations, teacher_metrics = collector(
        source_victims,
        policy_samples,
        attack,
        episodes=config.behavior_cloning_episodes,
        seed=config.seed + 500_000,
        **shared,
    )
    validation, validation_teacher_metrics = collector(
        source_victims,
        validation_samples,
        attack,
        episodes=config.behavior_cloning_validation_episodes,
        seed=config.seed + 600_000,
        **shared,
    )
    digest_before = policy.persistent_digest()
    fit_metrics = behavior_clone_policy(
        policy,
        demonstrations,
        epochs=config.behavior_cloning_epochs,
        seed=config.seed + 700_000,
        batch_size=config.behavior_cloning_batch_size,
    )
    validation_metrics = evaluate_behavior_clone_policy(
        policy,
        validation,
    )
    if (
        validation_metrics.get("baseline_provenance")
        != "evaluated_labels_validation_oracle"
        or validation_metrics.get("baseline_estimator")
        != "empirical_best_constant_no_smoothing"
    ):
        raise ValueError(
            "behavior-cloning validation baseline provenance is invalid"
        )
    digest_after = policy.persistent_digest()
    uses_soft_targets = validation_metrics.get("target_mode") in {
        "soft",
        "mixed_soft_and_hard",
    }
    if uses_soft_targets:
        gate_passed = bool(
            validation_metrics["soft_cross_entropy"]
            <= min(
                validation_metrics["uniform_soft_cross_entropy"] - 0.05,
                validation_metrics[
                    "validation_oracle_soft_cross_entropy"
                ] - 0.02,
            )
            and validation_metrics["top5_accuracy"]
            >= validation_metrics[
                "validation_oracle_top5_accuracy"
            ] + 0.02
        )
        gate = {
            "passed": gate_passed,
            "objective": "soft_gradient_distillation",
            "baseline": "evaluated_labels_validation_oracle",
            "minimum_soft_ce_improvement_over_uniform": 0.05,
            "minimum_soft_ce_improvement_over_validation_oracle": 0.02,
            "minimum_top5_gain_over_validation_oracle": 0.02,
        }
    else:
        gate_passed = bool(
            validation_metrics["accuracy"]
            >= max(
                4.0 * validation_metrics["uniform_accuracy"],
                validation_metrics[
                    "validation_oracle_top1_accuracy"
                ] + 0.05,
            )
            and validation_metrics["nll"]
            <= min(
                validation_metrics["uniform_nll"] - 0.2,
                validation_metrics["validation_oracle_nll"] - 0.05,
            )
        )
        gate = {
            "passed": gate_passed,
            "objective": "hard_action_classification",
            "baseline": "evaluated_labels_validation_oracle",
            "minimum_accuracy_multiple_of_uniform": 4.0,
            "minimum_accuracy_gain_over_validation_oracle": 0.05,
            "minimum_nll_improvement_over_uniform": 0.2,
            "minimum_nll_improvement_over_validation_oracle": 0.05,
        }
    return {
        "enabled": True,
        "method": "sequence_filtered_hindsight_imitation",
        "teacher": teacher_metrics,
        "validation_teacher": validation_teacher_metrics,
        "fit": fit_metrics,
        "validation": validation_metrics,
        "policy_digest_before": digest_before,
        "policy_digest_after": digest_after,
        "gate": gate,
        "elapsed_seconds": time.monotonic() - started,
    }


def _continue_ppo(
    policy: RecurrentAttackPolicy,
    *,
    checkpoint_path: Path,
    fingerprint: str,
    split_digest: str,
    kind: str,
    behavior_cloning: dict[str, object],
    completed_episodes: int,
    training_blocks: list[dict[str, object]],
    source_victims: VictimPopulation,
    policy_samples: Samples,
    attack: AttackConfig,
    config: MacPilotConfig,
    report: Callable[[str], None],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    while completed_episodes < config.policy_episodes:
        block_episodes = min(
            config.policy_update_block,
            config.policy_episodes - completed_episodes,
        )
        report(
            f"training {kind} episodes {completed_episodes + 1}-"
            f"{completed_episodes + block_episodes}/"
            f"{config.policy_episodes}"
        )
        block_started = time.monotonic()
        raw_block = train_population_policy(
            policy,
            source_victims,
            policy_samples,
            attack,
            episodes=block_episodes,
            seed=config.seed + completed_episodes,
            initial_family_weights=(
                training_blocks[-1]["family_weights"]
                if training_blocks
                else None
            ),
            episode_offset=completed_episodes,
            initial_instance_offsets=(
                training_blocks[-1]["instance_offsets"]
                if training_blocks
                else None
            ),
        )
        block = {
            **raw_block,
            "elapsed_seconds": time.monotonic() - block_started,
        }
        training_blocks.append(block)
        completed_episodes += block_episodes
        metadata = {
            "fingerprint": fingerprint,
            "dataset": config.dataset,
            "split_digest": split_digest,
            "seed": config.seed,
            "kind": kind,
            "completed_episodes": completed_episodes,
            "training_blocks": training_blocks,
            "behavior_cloning": behavior_cloning,
        }
        save_recurrent_checkpoint(checkpoint_path, policy, metadata)
    return training_blocks, {
        "fingerprint": fingerprint,
        "dataset": config.dataset,
        "split_digest": split_digest,
        "seed": config.seed,
        "kind": kind,
        "completed_episodes": completed_episodes,
        "training_blocks": training_blocks,
        "behavior_cloning": behavior_cloning,
    }


def _training_summary(
    config: MacPilotConfig,
    policy_samples: Samples,
    source_victims: VictimPopulation,
    blocks: Sequence[dict[str, object]],
    behavior_cloning: dict[str, object],
) -> dict[str, object]:
    return {
        "episodes": config.policy_episodes,
        "completed_episodes": config.policy_episodes,
        "trained_episodes": sum(
            int(block["trained_episodes"]) for block in blocks
        ),
        "policy_sample_pool_size": len(policy_samples),
        "unique_policy_samples_visited": len(
            {
                int(sample_index)
                for block in blocks
                for sample_index in block.get("sample_indices", [])
            }
        ),
        "source_calls": sum(
            int(block["source_calls"]) for block in blocks
        ),
        "source_calls_by_family": {
            family: sum(
                int(
                    block.get(
                        "source_calls_by_family",
                        {},
                    ).get(family, 0)
                )
                for block in blocks
            )
            for family in source_victims
        },
        "source_calls_by_victim": {
            victim_id: sum(
                int(
                    block.get(
                        "source_calls_by_victim",
                        {},
                    ).get(victim_id, 0)
                )
                for block in blocks
            )
            for instances in source_victims.values()
            for victim_id, _ in instances
        },
        "blocks": list(blocks),
        "final_family_weights": blocks[-1]["family_weights"],
        "behavior_cloning": behavior_cloning,
    }


def train_policy_bundle(
    *,
    config: MacPilotConfig,
    attack: AttackConfig,
    source_victims: VictimPopulation,
    policy_samples: Samples,
    bc_validation_samples: Samples,
    run_dir: Path,
    fingerprint: str,
    split_digest: str,
    device: torch.device,
    resume: bool,
    report: Callable[[str], None],
) -> PolicyTrainingBundle:
    main_path = resolve_descendant(
        run_dir,
        "policy.pt",
        label="hybrid policy checkpoint",
    )
    bc_path = resolve_descendant(
        run_dir,
        "policy_bc_only.pt",
        label="BC-only policy checkpoint",
    )
    ppo_path = resolve_descendant(
        run_dir,
        "policy_ppo_only.pt",
        label="PPO-only policy checkpoint",
    )
    main_exists = (
        resume
        and main_path.is_file()
        and main_path.with_suffix(".pt.sha256").is_file()
    )
    if main_exists:
        report("loading hybrid recurrent policy checkpoint")
        main, metadata = _load_checked(
            main_path,
            device,
            fingerprint,
            config,
            attack,
        )
        completed = int(metadata["completed_episodes"])
        blocks = list(metadata["training_blocks"])
        behavior_cloning = dict(metadata["behavior_cloning"])
        main_resumed = True
    else:
        main = _new_policy(config, attack, device)
        completed = 0
        blocks = []
        behavior_cloning = (
            _collect_and_fit_bc(
                main,
                source_victims,
                policy_samples,
                bc_validation_samples,
                attack,
                config,
            )
            if config.behavior_cloning_episodes
            else {
                "enabled": False,
                "gate": {
                    "passed": True,
                    "reason": "disabled legacy configuration",
                },
            }
        )
        if config.train_ablation_policies:
            if not behavior_cloning["enabled"]:
                raise ValueError(
                    "BC-only ablation requires behavior cloning"
                )
            save_recurrent_checkpoint(
                bc_path,
                main,
                {
                    "fingerprint": fingerprint,
                    "dataset": config.dataset,
                    "split_digest": split_digest,
                    "seed": config.seed,
                    "kind": "gradient_bc_only",
                    "completed_episodes": 0,
                    "training_blocks": [],
                    "behavior_cloning": behavior_cloning,
                },
            )
        main_resumed = False
    main_method_id = main_policy_method_id(config)
    blocks, main_metadata = _continue_ppo(
        main,
        checkpoint_path=main_path,
        fingerprint=fingerprint,
        split_digest=split_digest,
        kind=main_method_id,
        behavior_cloning=behavior_cloning,
        completed_episodes=completed,
        training_blocks=blocks,
        source_victims=source_victims,
        policy_samples=policy_samples,
        attack=attack,
        config=config,
        report=report,
    )
    main_training = _training_summary(
        config,
        policy_samples,
        source_victims,
        blocks,
        behavior_cloning,
    )

    bc_only: RecurrentAttackPolicy | None = None
    ppo_only: RecurrentAttackPolicy | None = None
    ablation_training: dict[str, object] = {}
    if config.train_ablation_policies:
        if not (
            bc_path.is_file()
            and bc_path.with_suffix(".pt.sha256").is_file()
        ):
            raise RuntimeError(
                "BC-only checkpoint is missing; restart this policy run"
            )
        bc_only, bc_metadata = _load_checked(
            bc_path,
            device,
            fingerprint,
            config,
            attack,
        )
        ppo_exists = (
            resume
            and ppo_path.is_file()
            and ppo_path.with_suffix(".pt.sha256").is_file()
        )
        if ppo_exists:
            ppo_only, ppo_metadata = _load_checked(
                ppo_path,
                device,
                fingerprint,
                config,
                attack,
            )
            ppo_completed = int(ppo_metadata["completed_episodes"])
            ppo_blocks = list(ppo_metadata["training_blocks"])
        else:
            ppo_only = _new_policy(config, attack, device)
            ppo_completed = 0
            ppo_blocks = []
        disabled_bc = {
            "enabled": False,
            "gate": {
                "passed": True,
                "reason": "PPO-only component ablation",
            },
        }
        ppo_blocks, ppo_metadata = _continue_ppo(
            ppo_only,
            checkpoint_path=ppo_path,
            fingerprint=fingerprint,
            split_digest=split_digest,
            kind="ppo_only",
            behavior_cloning=disabled_bc,
            completed_episodes=ppo_completed,
            training_blocks=ppo_blocks,
            source_victims=source_victims,
            policy_samples=policy_samples,
            attack=attack,
            config=config,
            report=report,
        )
        ablation_training = {
            "bc_only": {
                "checkpoint_sha256": sha256_file(bc_path),
                "metadata": bc_metadata,
            },
            "ppo_only": _training_summary(
                config,
                policy_samples,
                source_victims,
                ppo_blocks,
                disabled_bc,
            ),
        }
    training = {
        **main_training,
        "method_id": main_method_id,
        "component_ablations": ablation_training,
    }
    checkpoints = {
        "main": {
            "path": str(main_path),
            "sha256": sha256_file(main_path),
            "method_id": main_method_id,
        }
    }
    if config.train_ablation_policies:
        checkpoints.update(
            {
                "bc_only": {
                    "path": str(bc_path),
                    "sha256": sha256_file(bc_path),
                },
                "ppo_only": {
                    "path": str(ppo_path),
                    "sha256": sha256_file(ppo_path),
                },
            }
        )
    return PolicyTrainingBundle(
        main=main,
        bc_only=bc_only,
        ppo_only=ppo_only,
        training=training,
        main_resumed=main_resumed,
        checkpoints=checkpoints,
    )
