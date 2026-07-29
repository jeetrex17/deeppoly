"""Fresh source-gradient teacher collection for Phase 2 D1."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from .artifacts import sha256_file
from .cifar_data import indices_digest
from .cifar_manifest import code_digest
from .config import AttackConfig
from .imitation import BehaviorCloneStep, collect_gradient_demonstrations
from .operator import AttackOperatorContract
from .phase2_residual_d1 import (
    D1_SOURCE_FAMILIES,
    ResidualCacheBinding,
    ResidualD1Request,
)
from .phase2_residual_d1_cache import (
    RESIDUAL_TEACHER_CACHE_SCHEMA_VERSION,
    ResidualTeacherCache,
    load_or_create_residual_teacher_cache,
)
from .phase2_residual_d1_source import D1SourceContext, _canonical_digest
from .population import balanced_family_schedule


D1_TRAIN_DECISIONS = 12
D1_VALIDATION_DECISIONS = 6
D1_BC_EPOCHS = 12
D1_BC_BLOCK = 2
D1_HIDDEN_DIM = 128
D1_PRIOR_TEMPERATURE = 24.0
D1_SOFT_TEMPERATURE = 0.5


def _protocol_record(
    request: ResidualD1Request,
    context: D1SourceContext,
) -> dict[str, object]:
    attack = context.config.attack_config()
    return {
        "schema": "phase2-d1-residual-teacher-v2",
        "request_sha256": request.digest(),
        "code_digest": code_digest(),
        "train_decisions": D1_TRAIN_DECISIONS,
        "validation_decisions": D1_VALIDATION_DECISIONS,
        "bc_epochs": D1_BC_EPOCHS,
        "soft_temperature": D1_SOFT_TEMPERATURE,
        "prior_temperature": D1_PRIOR_TEMPERATURE,
        "operator_digest": AttackOperatorContract.from_config(attack).digest(),
        "role_indices_sha256": {
            "train": indices_digest(context.train_indices),
            "threshold_selection": indices_digest(context.threshold_indices),
            "competence_gate": indices_digest(context.competence_indices),
            "source_holdout_evaluation": indices_digest(context.evaluation_indices),
            "source_ppo_evaluation": indices_digest(context.ppo_evaluation_indices),
        },
        "teacher_victim_ids": {
            family: [victim_id for victim_id, _ in victims]
            for family, victims in context.teacher_victims.items()
        },
        "evaluation_victim_ids": {
            family: [victim_id for victim_id, _ in victims]
            for family, victims in context.evaluation_victims.items()
        },
        "victim_cache_digest": context.victim_cache_digest,
    }


def _cache_binding(
    request: ResidualD1Request,
    context: D1SourceContext,
) -> tuple[ResidualCacheBinding, dict[str, object]]:
    protocol = _protocol_record(request, context)
    binding = ResidualCacheBinding(
        source_manifest_sha256=context.source_manifest_sha256,
        dataset_content_sha256=context.dataset_content_sha256,
        victim_cache_digest=context.victim_cache_digest,
        request_sha256=_canonical_digest(protocol),
    )
    return binding, protocol


def _write_verified_jsonl(
    path: Path,
    records: Iterable[Mapping[str, object]],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    checksum_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=".tmp",
            mode="w",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(
                    json.dumps(
                        dict(record),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, path)
        digest = sha256_file(path)
        checksum = path.with_suffix(path.suffix + ".sha256")
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=".tmp",
            mode="w",
            delete=False,
        ) as handle:
            checksum_temporary = Path(handle.name)
            handle.write(digest + "\n")
        os.replace(checksum_temporary, checksum)
        return digest
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        if checksum_temporary is not None and checksum_temporary.exists():
            checksum_temporary.unlink()


def _trajectory_family(trajectory_id: str) -> str:
    marker = "bc-gradient-source:"
    if marker not in trajectory_id:
        raise ValueError("D1 teacher trajectory lacks source-family provenance")
    family = trajectory_id.split(marker, 1)[1].split(":", 1)[0]
    if family not in D1_SOURCE_FAMILIES:
        raise ValueError("D1 teacher trajectory has invalid family provenance")
    return family


def _step_record(step: BehaviorCloneStep, *, role: str) -> dict[str, object]:
    return {
        "role": role,
        "source_family": _trajectory_family(step.trajectory_id),
        "observation": list(step.observation),
        "action": step.action,
        "accepted": step.accepted,
        "trajectory_id": step.trajectory_id,
        "step_index": step.step_index,
        "action_distribution": list(step.action_distribution or ()),
        "target_calls": 0,
    }


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _rotate_victims(
    victims: Mapping[str, Sequence[tuple[str, nn.Module]]],
    block_number: int,
) -> dict[str, tuple[tuple[str, nn.Module], ...]]:
    rotated: dict[str, tuple[tuple[str, nn.Module], ...]] = {}
    for family, family_victims in victims.items():
        instances = tuple(family_victims)
        if not instances:
            raise ValueError("D1 teacher families require source victims")
        offset = block_number % len(instances)
        rotated[family] = instances[offset:] + instances[:offset]
    return rotated


def _scheduled_victims(
    victims: Mapping[str, Sequence[tuple[str, nn.Module]]],
    *,
    episodes: int,
    seed: int,
) -> tuple[tuple[str, str], ...]:
    schedule = balanced_family_schedule(tuple(victims), episodes, seed)
    family_offsets = {family: 0 for family in victims}
    scheduled: list[tuple[str, str]] = []
    for family in schedule:
        instances = tuple(victims[family])
        offset = family_offsets[family] % len(instances)
        family_offsets = {
            **family_offsets,
            family: family_offsets[family] + 1,
        }
        scheduled.append((family, instances[offset][0]))
    return tuple(scheduled)


def _block_source_calls_by_family(
    steps: Sequence[BehaviorCloneStep],
    scheduled: Sequence[tuple[str, str]],
    metrics: Mapping[str, object],
) -> dict[str, int]:
    calls = {
        family: sum(item_family == family for item_family, _ in scheduled)
        + sum(_trajectory_family(step.trajectory_id) == family for step in steps)
        for family in {family for family, _ in scheduled}
    }
    recorded = metrics.get("source_calls_by_family")
    if isinstance(recorded, Mapping) and any(
        int(recorded.get(family, -1)) != count for family, count in calls.items()
    ):
        raise ValueError("D1 teacher per-family source-call audit mismatch")
    if sum(calls.values()) != int(metrics["source_calls"]):
        raise ValueError("D1 teacher source-call audit mismatch")
    return calls


def _collect_teacher_blocks(
    *,
    victims: Mapping[str, Sequence[tuple[str, nn.Module]]],
    samples: Sequence[tuple[torch.Tensor, int]],
    config: AttackConfig,
    episodes: int,
    decisions: int,
    seed: int,
    role: str,
    deadline_check: Callable[[], None],
    progress: Callable[[str], None],
) -> tuple[tuple[BehaviorCloneStep, ...], dict[str, object]]:
    collected: list[BehaviorCloneStep] = []
    source_calls = 0
    gradient_evaluations = 0
    accepted_steps = 0
    blocks = math.ceil(episodes / D1_BC_BLOCK)
    victim_ids = tuple(
        victim_id
        for family_victims in victims.values()
        for victim_id, _ in family_victims
    )
    family_ids = tuple(victims)
    scheduled_episodes_by_victim = {victim_id: 0 for victim_id in victim_ids}
    source_calls_by_victim = {victim_id: 0 for victim_id in victim_ids}
    scheduled_episodes_by_family = {family: 0 for family in family_ids}
    source_calls_by_family = {family: 0 for family in family_ids}
    for block_start in range(0, episodes, D1_BC_BLOCK):
        deadline_check()
        block_number = block_start // D1_BC_BLOCK
        progress(f"[d1] collecting {role} teacher block {block_number + 1}/{blocks}")
        block_episodes = min(D1_BC_BLOCK, episodes - block_start)
        block_seed = seed + block_start
        rotated_victims = _rotate_victims(victims, block_number)
        scheduled = _scheduled_victims(
            rotated_victims,
            episodes=block_episodes,
            seed=block_seed,
        )
        steps, metrics = collect_gradient_demonstrations(
            rotated_victims,
            samples[block_start : block_start + block_episodes],
            config,
            episodes=block_episodes,
            decisions=decisions,
            seed=block_seed,
            soft_temperature=D1_SOFT_TEMPERATURE,
        )
        block_calls = _block_source_calls_by_family(
            steps,
            scheduled,
            metrics,
        )
        for family, victim_id in scheduled:
            scheduled_episodes_by_victim = {
                **scheduled_episodes_by_victim,
                victim_id: scheduled_episodes_by_victim[victim_id] + 1,
            }
            scheduled_episodes_by_family = {
                **scheduled_episodes_by_family,
                family: scheduled_episodes_by_family[family] + 1,
            }
        for family, family_calls in block_calls.items():
            scheduled_ids = {
                victim_id
                for scheduled_family, victim_id in scheduled
                if scheduled_family == family
            }
            if len(scheduled_ids) != 1:
                raise ValueError("D1 teacher block cannot audit source calls by victim")
            victim_id = next(iter(scheduled_ids))
            source_calls_by_victim = {
                **source_calls_by_victim,
                victim_id: source_calls_by_victim[victim_id] + family_calls,
            }
            source_calls_by_family = {
                **source_calls_by_family,
                family: source_calls_by_family[family] + family_calls,
            }
        for step in steps:
            observation = list(step.observation)
            observation[4] = (
                config.max_queries - (step.step_index + 1)
            ) / config.max_queries
            observation[7] = (step.step_index + 1) / config.max_queries
            collected.append(
                BehaviorCloneStep(
                    observation,
                    step.action,
                    step.accepted,
                    trajectory_id=(
                        f"d1-{role}-block-{block_number}:{step.trajectory_id}"
                    ),
                    step_index=step.step_index,
                    action_distribution=step.action_distribution,
                )
            )
        source_calls += int(metrics["source_calls"])
        gradient_evaluations += int(metrics["gradient_evaluations"])
        accepted_steps += int(metrics["accepted_steps"])
        deadline_check()
    if (
        sum(scheduled_episodes_by_victim.values()) != episodes
        or sum(source_calls_by_victim.values()) != source_calls
    ):
        raise ValueError("D1 teacher per-victim audit totals mismatch")
    victim_diagnostics = {
        victim_id: {
            "scheduled_episodes": scheduled_episodes_by_victim[victim_id],
            "source_calls": source_calls_by_victim[victim_id],
        }
        for victim_id in victim_ids
    }
    return tuple(collected), {
        "role": role,
        "episodes": episodes,
        "decisions_per_episode": decisions,
        "steps": len(collected),
        "accepted_steps": accepted_steps,
        "source_calls": source_calls,
        "gradient_evaluations": gradient_evaluations,
        "scheduled_episodes_by_family": scheduled_episodes_by_family,
        "source_calls_by_family": source_calls_by_family,
        "scheduled_episodes_by_victim": scheduled_episodes_by_victim,
        "source_calls_by_victim": source_calls_by_victim,
        "victim_diagnostics": victim_diagnostics,
        "target_calls": 0,
        "hidden_target_calls": 0,
        "hidden_target_evaluation_performed": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _teacher_examples(
    request: ResidualD1Request,
    context: D1SourceContext,
    *,
    deadline_check: Callable[[], None],
    progress: Callable[[str], None],
) -> tuple[
    tuple[BehaviorCloneStep, ...],
    tuple[BehaviorCloneStep, ...],
    tuple[BehaviorCloneStep, ...],
    Mapping[str, object],
]:
    attack = context.config.attack_config()
    binding, protocol = _cache_binding(request, context)

    def create() -> ResidualTeacherCache:
        train, train_metrics = _collect_teacher_blocks(
            victims=context.teacher_victims,
            samples=context.train_samples,
            config=attack,
            episodes=request.bc_episodes,
            decisions=D1_TRAIN_DECISIONS,
            seed=request.seed + 10_000,
            role="train",
            deadline_check=deadline_check,
            progress=progress,
        )
        threshold, threshold_metrics = _collect_teacher_blocks(
            victims=context.teacher_victims,
            samples=context.threshold_samples,
            config=attack,
            episodes=len(context.threshold_samples),
            decisions=D1_VALIDATION_DECISIONS,
            seed=request.seed + 20_000,
            role="threshold_selection",
            deadline_check=deadline_check,
            progress=progress,
        )
        competence, competence_metrics = _collect_teacher_blocks(
            victims=context.teacher_victims,
            samples=context.competence_samples,
            config=attack,
            episodes=len(context.competence_samples),
            decisions=D1_VALIDATION_DECISIONS,
            seed=request.seed + 30_000,
            role="competence_gate",
            deadline_check=deadline_check,
            progress=progress,
        )
        return ResidualTeacherCache(
            binding=binding,
            protocol=protocol,
            heldout_family=request.heldout_family,
            source_families=context.source_families,
            train_steps=train,
            threshold_steps=threshold,
            competence_steps=competence,
            role_metrics={
                "train": train_metrics,
                "threshold_selection": threshold_metrics,
                "competence_gate": competence_metrics,
            },
        )

    cache = load_or_create_residual_teacher_cache(
        request.output_dir,
        expected_binding=binding,
        expected_protocol=protocol,
        action_dim=attack.action_dim,
        observation_dim=attack.recurrent_observation_dim,
        create=create,
    )
    manifest: dict[str, object] = {
        "schema_version": RESIDUAL_TEACHER_CACHE_SCHEMA_VERSION,
        "name": "phase2-d1-fresh-source-gradient-teacher",
        "binding": asdict(cache.binding),
        "protocol": _plain(cache.protocol),
        "heldout_family": request.heldout_family,
        "source_families": list(context.source_families),
        "roles": _plain(cache.role_metrics),
        "examples_sha256": cache.examples_sha256,
        "metadata_sha256": cache.metadata_sha256,
        "cache_reused": cache.reused,
        "target_calls": 0,
        "target_evaluation_available": False,
        "hidden_target_calls": 0,
        "hidden_target_evaluation_performed": False,
        "authorizes_hidden_target_evaluation": False,
    }
    return (
        cache.train_steps,
        cache.threshold_steps,
        cache.competence_steps,
        manifest,
    )


__all__ = (
    "D1_BC_BLOCK",
    "D1_BC_EPOCHS",
    "D1_HIDDEN_DIM",
    "D1_PRIOR_TEMPERATURE",
    "D1_SOFT_TEMPERATURE",
    "D1_TRAIN_DECISIONS",
    "D1_VALIDATION_DECISIONS",
    "_teacher_examples",
    "_write_verified_jsonl",
)
