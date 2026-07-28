"""Exact-source checkpoint reconstruction and Stage A fold evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset

from .artifacts import (
    load_model_checkpoint,
    load_recurrent_checkpoint,
    sha256_file,
)
from .cifar_config import MacPilotConfig
from .cifar_data import (
    build_cifar_split,
    dataset_samples,
    disjoint_balanced_subsets,
    indices_digest,
)
from .cifar_models import build_cifar_victim_population
from .config import AttackConfig
from .models import freeze_model
from .phase2_policy import FrozenTemperaturePolicy
from .phase2_temperature_manifest import (
    Phase1SourceFold,
    StageARequest,
    _require_digest,
    _require_mapping,
    select_fixed_source_development_indices,
)
from .research_metrics import AttackOutcome, asr_at_budgets, asr_query_auc
from .research_protocol import run_frozen_episode, run_score_greedy_episode
from .results import ResearchResultRow, read_jsonl


def _source_gate_indices(
    fold: Phase1SourceFold,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> tuple[int, ...]:
    config = MacPilotConfig(**fold.run_manifest["config"])
    split = build_cifar_split(
        train_dataset.targets,
        test_dataset.targets,
        config.victim_train_images,
        config.policy_train_images,
        config.source_validation_images,
        config.outer_test_images,
        config.split_seed if config.split_seed is not None else config.seed,
    )
    if split.digest != fold.run_manifest.get("split_digest"):
        raise ValueError("reconstructed CIFAR split digest mismatch")
    if not (
        config.victim_validation_images > 0
        and config.behavior_cloning_validation_episodes > 0
        and config.source_evaluation_images > 0
    ):
        raise ValueError("Phase 1 source-role split is incomplete")
    _, _, source_indices = disjoint_balanced_subsets(
        train_dataset,
        split.source_validation,
        (
            config.victim_validation_images,
            config.behavior_cloning_validation_episodes,
            config.source_evaluation_images,
        ),
    )
    roles = _require_mapping(
        fold.run_manifest.get("data_role_digests"),
        label="Phase 1 role digests",
    )
    if indices_digest(source_indices) != roles.get("source_gate"):
        raise ValueError("reconstructed source-development split mismatch")
    return source_indices


def _fixed_indices_by_family(
    fold: Phase1SourceFold,
    train_dataset: Dataset,
    test_dataset: Dataset,
    count: int,
) -> dict[str, tuple[int, ...]]:
    candidates = _source_gate_indices(fold, train_dataset, test_dataset)
    rows = read_jsonl(fold.source_results_path)
    return {
        family: select_fixed_source_development_indices(
            rows,
            family=family,
            exact_source_victim_ids=tuple(
                str(spec["victim_id"])
                for spec in fold.source_victims[family]
            ),
            candidate_indices=candidates,
            count=count,
        )
        for family in fold.source_families
    }


def _phase1_policy_digest(policy: torch.nn.Module) -> str:
    """Reproduce the schema-1 digest used when Phase 1 was executed."""

    hasher = hashlib.sha256()

    def update(value: object) -> None:
        if isinstance(value, torch.Tensor):
            hasher.update(str(value.dtype).encode("utf-8"))
            hasher.update(
                value.detach().cpu().contiguous().numpy().tobytes()
            )
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                hasher.update(str(key).encode("utf-8"))
                update(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                update(item)
        else:
            hasher.update(repr(value).encode("utf-8"))

    update(policy.state_dict())
    update(policy.optimizer.state_dict())
    hasher.update(
        json.dumps(
            asdict(policy.config),
            sort_keys=True,
        ).encode("utf-8")
    )
    return hasher.hexdigest()


class _ManifestDigestTemperaturePolicy(FrozenTemperaturePolicy):
    """Expose the verified schema-1 digest during frozen replay."""

    def __init__(
        self,
        checkpoint,
        temperature: float,
        manifest_digest: str,
    ) -> None:
        super().__init__(checkpoint, temperature)
        self._manifest_digest = _require_digest(
            manifest_digest,
            label="manifest policy digest",
        )

    def persistent_digest(self) -> str:
        return self._manifest_digest


def _load_policy(
    fold: Phase1SourceFold,
    *,
    device: torch.device,
    attack: AttackConfig,
) -> torch.nn.Module:
    config = MacPilotConfig(**fold.run_manifest["config"])
    policy, metadata = load_recurrent_checkpoint(
        fold.policy_path,
        device,
        expected_observation_dim=attack.recurrent_observation_dim,
        expected_action_dim=attack.action_dim,
        expected_hidden_dim=config.hidden_dim,
        expected_actor_mode="flat",
    )
    if (
        metadata.get("seed") != fold.seed
        or metadata.get("split_digest")
        != fold.run_manifest.get("split_digest")
        or metadata.get("kind") != "gradient_bc_groupdro_ppo"
        or int(metadata.get("completed_episodes", -1))
        != config.policy_episodes
    ):
        raise ValueError("Phase 1 policy checkpoint metadata mismatch")
    expected_digest = _require_mapping(
        fold.run_manifest.get("policy"),
        label="Phase 1 policy",
    ).get("persistent_digest")
    if _phase1_policy_digest(policy) != expected_digest:
        raise ValueError("Phase 1 persistent policy digest mismatch")
    return policy


def _load_exact_source_victims(
    fold: Phase1SourceFold,
    *,
    device: torch.device,
) -> dict[str, tuple[tuple[str, nn.Module], ...]]:
    config = MacPilotConfig(**fold.run_manifest["config"])
    population = build_cifar_victim_population(
        config.victim_seed if config.victim_seed is not None else config.seed,
        {
            family: config.source_instances_per_family
            for family in fold.source_families
        },
        families=fold.source_families,
        profile=config.victim_profile,
    )
    cache_digest = fold.run_manifest.get("victim_cache_digest")
    cache_contract = fold.run_manifest.get("victim_cache_contract")
    loaded: dict[str, tuple[tuple[str, nn.Module], ...]] = {}
    for family in fold.source_families:
        expected_specs = fold.source_victims[family]
        instances = population.get(family)
        if instances is None or len(instances) != len(expected_specs):
            raise ValueError("reconstructed exact-source victim count mismatch")
        family_victims: list[tuple[str, nn.Module]] = []
        for (victim_id, model), spec in zip(instances, expected_specs):
            if victim_id != spec.get("victim_id"):
                raise ValueError("reconstructed exact-source victim ID mismatch")
            checkpoint = Path(str(spec["local_checkpoint"]))
            metadata = load_model_checkpoint(checkpoint, model, device)
            if (
                metadata.get("fingerprint") != cache_digest
                or metadata.get("cache_contract") != cache_contract
                or metadata.get("family") != family
                or metadata.get("instance_index")
                != spec.get("instance_index")
                or metadata.get("training_seed")
                != spec.get("training_seed")
            ):
                raise ValueError("exact-source victim metadata mismatch")
            family_victims.append(
                (victim_id, freeze_model(model.to(device)))
            )
        loaded[family] = tuple(family_victims)
    return loaded


def _episode_seed(
    fold: Phase1SourceFold,
    family: str,
    victim_id: str,
    sample_index: int,
) -> int:
    payload = (
        f"phase2-stage-a-common-draw-v1:{fold.seed}:"
        f"{fold.heldout_family}:{family}:{victim_id}:{sample_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _temperature_method(temperature: float) -> str:
    return f"temperature_{temperature:.2f}".replace(".", "p")


def _to_result_row(
    result,
    *,
    method: str,
    seed: int,
    family: str,
    query_budget: int,
) -> ResearchResultRow:
    return ResearchResultRow(
        sample_id=result.sample_id,
        victim_id=result.victim_id,
        victim_family=family,
        method=method,
        threat_model="T1",
        seed=seed,
        query_budget=query_budget,
        clean_correct=result.clean_correct,
        success=result.success,
        query_to_success=result.query_to_success,
        total_target_calls=result.total_target_calls,
        linf=result.linf,
        l2=result.l2,
        policy_digest=result.policy_digest_after,
        action_trace=result.actions,
    )


def _method_metrics(
    rows: Sequence[ResearchResultRow],
    *,
    attack: AttackConfig,
) -> dict[str, object]:
    if not rows or any(not row.clean_correct for row in rows):
        raise ValueError("Stage A rows must use the fixed eligible cohort")
    outcomes = tuple(
        AttackOutcome(row.clean_correct, row.query_to_success)
        for row in rows
    )
    curve = asr_at_budgets(
        outcomes,
        tuple(range(attack.max_queries + 1)),
    )
    counts = Counter(
        action for row in rows for action in row.action_trace
    )
    action_total = sum(counts.values())
    entropy = 0.0
    if action_total:
        entropy = -sum(
            (count / action_total) * math.log(count / action_total)
            for count in counts.values()
        ) / math.log(attack.action_dim)
    sample_ids = tuple(sorted(row.sample_id for row in rows))
    digests = {row.policy_digest for row in rows}
    return {
        "eligible": len(rows),
        "successes": sum(row.success for row in rows),
        "asr": curve[attack.max_queries],
        "asr_query_auc": asr_query_auc(curve),
        "normalized_action_entropy": entropy,
        "eligible_sample_ids_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest(),
        "policy_digests": sorted(digests),
        "source_model_calls": sum(row.total_target_calls for row in rows),
    }


class _StageDeadlineReached(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        rows: Sequence[ResearchResultRow] = (),
        partial_fold: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.rows = tuple(rows)
        self.partial_fold = (
            dict(partial_fold) if partial_fold is not None else None
        )


def _check_deadline(
    clock: Callable[[], float],
    absolute_deadline: float,
) -> None:
    if clock() >= absolute_deadline:
        raise _StageDeadlineReached("Stage A scheduling deadline reached")


def evaluate_temperature_fold(
    *,
    fold: Phase1SourceFold,
    request: StageARequest,
    train_dataset: Dataset,
    test_dataset: Dataset,
    absolute_deadline: float,
    clock: Callable[[], float],
    progress: Callable[[str], None],
) -> tuple[dict[str, object], list[ResearchResultRow]]:
    """Evaluate one complete leave-one-family-out source fold."""

    config = MacPilotConfig(**fold.run_manifest["config"])
    attack = config.attack_config()
    if (
        attack.max_queries != 50
        or config.source_instances_per_family != 2
        or fold.seed != 17
    ):
        raise ValueError("Stage A fold does not match the locked attack contract")
    device = torch.device(request.device)
    policy = _load_policy(fold, device=device, attack=attack)
    runtime_policy_digest = policy.persistent_digest()
    policy_digest = _require_mapping(
        fold.run_manifest.get("policy"),
        label="Phase 1 policy",
    ).get("persistent_digest")
    if _phase1_policy_digest(policy) != policy_digest:
        raise ValueError("loaded Phase 1 policy digest mismatch")
    victims = _load_exact_source_victims(fold, device=device)
    indices_by_family = _fixed_indices_by_family(
        fold,
        train_dataset,
        test_dataset,
        request.eligible_images_per_family,
    )

    rows: list[ResearchResultRow] = []
    family_summaries: dict[str, dict[str, object]] = {}
    completed_method_blocks: list[str] = []

    def check_episode_deadline(family: str, method: str) -> None:
        try:
            _check_deadline(clock, absolute_deadline)
        except _StageDeadlineReached as error:
            raise _StageDeadlineReached(
                str(error),
                rows=rows,
                partial_fold={
                    "seed": fold.seed,
                    "heldout_family": fold.heldout_family,
                    "source_families": list(fold.source_families),
                    "policy_checkpoint_sha256": _require_mapping(
                        fold.run_manifest.get("policy"),
                        label="Phase 1 policy",
                    ).get("checkpoint_sha256"),
                    "policy_persistent_digest": policy_digest,
                    "runtime_policy_digest": runtime_policy_digest,
                    "complete": False,
                    "interrupted": True,
                    "interrupted_family": family,
                    "interrupted_method": method,
                    "completed_method_blocks": list(
                        completed_method_blocks
                    ),
                    "partial_row_count": len(rows),
                    "source_model_calls": sum(
                        row.total_target_calls for row in rows
                    ),
                    "target_calls": 0,
                    "target_evaluation_performed": False,
                },
            ) from error

    for family in fold.source_families:
        family_rows: dict[str, list[ResearchResultRow]] = {
            "score_greedy": [],
            **{
                _temperature_method(temperature): []
                for temperature in request.temperatures
            },
        }
        indices = indices_by_family[family]
        samples = dataset_samples(train_dataset, indices)
        source_victims = victims[family]

        check_episode_deadline(family, "score_greedy")
        progress(
            f"[stage-a] {fold.heldout_family}/seed-{fold.seed}/"
            f"{family}: score-greedy"
        )
        for victim_id, victim in source_victims:
            for (image, label), sample_index in zip(samples, indices):
                check_episode_deadline(family, "score_greedy")
                result = run_score_greedy_episode(
                    victim,
                    image,
                    label,
                    (
                        f"cifar10:{family}:{victim_id}:"
                        f"{sample_index}"
                    ),
                    victim_id,
                    family,
                    attack,
                    seed=_episode_seed(
                        fold,
                        family,
                        victim_id,
                        sample_index,
                    ),
                )
                if not result.clean_correct:
                    raise ValueError(
                        "fixed Phase 1 eligible cohort changed under replay"
                    )
                if (
                    result.policy_digest_before
                    != result.policy_digest_after
                ):
                    raise ValueError("score-greedy state changed during replay")
                row = _to_result_row(
                    result,
                    method="score_greedy",
                    seed=fold.seed,
                    family=family,
                    query_budget=attack.max_queries,
                )
                family_rows["score_greedy"].append(row)
                rows.append(row)
        completed_method_blocks.append(f"{family}/score_greedy")

        for temperature in request.temperatures:
            method = _temperature_method(temperature)
            check_episode_deadline(family, method)
            progress(
                f"[stage-a] {fold.heldout_family}/seed-{fold.seed}/"
                f"{family}: temperature={temperature:.2f}"
            )
            controlled = _ManifestDigestTemperaturePolicy(
                policy,
                temperature,
                str(policy_digest),
            )
            digest_before = controlled.persistent_digest()
            for victim_id, victim in source_victims:
                for (image, label), sample_index in zip(samples, indices):
                    check_episode_deadline(family, method)
                    result = run_frozen_episode(
                        controlled,
                        victim,
                        image,
                        label,
                        (
                            f"cifar10:{family}:{victim_id}:"
                            f"{sample_index}"
                        ),
                        victim_id,
                        family,
                        attack,
                        deterministic=False,
                        episode_seed=_episode_seed(
                            fold,
                            family,
                            victim_id,
                            sample_index,
                        ),
                    )
                    if not result.clean_correct:
                        raise ValueError(
                            "fixed Phase 1 eligible cohort changed under replay"
                        )
                    if (
                        result.policy_digest_before
                        != result.policy_digest_after
                    ):
                        raise ValueError(
                            "temperature policy changed during replay"
                        )
                    row = _to_result_row(
                        result,
                        method=method,
                        seed=fold.seed,
                        family=family,
                        query_budget=attack.max_queries,
                    )
                    family_rows[method].append(row)
                    rows.append(row)
            if (
                digest_before != policy_digest
                or controlled.persistent_digest() != policy_digest
                or policy.persistent_digest() != runtime_policy_digest
            ):
                raise ValueError("temperature evaluation mutated the policy")
            completed_method_blocks.append(f"{family}/{method}")

        cohorts = {
            method: {row.sample_id for row in method_rows}
            for method, method_rows in family_rows.items()
        }
        if len({frozenset(value) for value in cohorts.values()}) != 1:
            raise ValueError("Stage A methods do not share one source cohort")
        family_summaries[family] = {
            "indices": list(indices),
            "indices_sha256": indices_digest(indices),
            "victim_ids": [
                victim_id for victim_id, _ in source_victims
            ],
            "methods": {
                method: _method_metrics(method_rows, attack=attack)
                for method, method_rows in family_rows.items()
            },
        }

    expected_methods = {
        "score_greedy",
        *{
            _temperature_method(temperature)
            for temperature in request.temperatures
        },
    }
    pooled = {
        method: _method_metrics(
            tuple(row for row in rows if row.method == method),
            attack=attack,
        )
        for method in expected_methods
    }
    score = pooled["score_greedy"]
    temperature_metrics: dict[str, dict[str, object]] = {}
    for temperature in request.temperatures:
        method = _temperature_method(temperature)
        metrics = pooled[method]
        frozen = (
            metrics["policy_digests"] == [policy_digest]
            and policy.persistent_digest() == runtime_policy_digest
            and _phase1_policy_digest(policy) == policy_digest
        )
        temperature_metrics[str(temperature)] = {
            "asr": metrics["asr"],
            "auc": metrics["asr_query_auc"],
            "normalized_action_entropy": metrics[
                "normalized_action_entropy"
            ],
            "asr_gain_vs_score": float(metrics["asr"])
            - float(score["asr"]),
            "auc_gain_vs_score": float(metrics["asr_query_auc"])
            - float(score["asr_query_auc"]),
            "eligible": metrics["eligible"],
            "source_model_calls": metrics["source_model_calls"],
            "policy_digest_before": policy_digest,
            "policy_digest_after": policy_digest,
            "runtime_policy_digest": runtime_policy_digest,
            "frozen": frozen,
        }
        if not frozen:
            raise ValueError("Stage A frozen-policy audit failed")
    return (
        {
            "seed": fold.seed,
            "heldout_family": fold.heldout_family,
            "source_families": list(fold.source_families),
            "policy_checkpoint_sha256": sha256_file(fold.policy_path),
            "policy_persistent_digest": policy_digest,
            "runtime_policy_digest": runtime_policy_digest,
            "score_greedy": {
                "asr": score["asr"],
                "auc": score["asr_query_auc"],
                "normalized_action_entropy": score[
                    "normalized_action_entropy"
                ],
                "eligible": score["eligible"],
                "source_model_calls": score["source_model_calls"],
            },
            "temperatures": temperature_metrics,
            "families": family_summaries,
            "source_model_calls": sum(
                row.total_target_calls for row in rows
            ),
            "target_calls": 0,
            "target_evaluation_performed": False,
            "complete": True,
        },
        rows,
    )
