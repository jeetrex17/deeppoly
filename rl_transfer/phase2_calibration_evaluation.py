"""Frozen Phase 2 checkpoint replay for the calibration diagnostic."""

from __future__ import annotations

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
from .cifar_evaluation import evaluate_method_set
from .cifar_models import build_cifar_victim_population
from .evaluation_audit import audit_evaluation
from .models import freeze_model
from .paths import resolve_descendant
from .phase2_calibration_manifest import Phase2CalibrationRequest
from .phase2_policy import FrozenTemperaturePolicy
from .results import ResearchResultRow, read_jsonl


class CalibrationDeadlineReached(TimeoutError):
    """Raised only between bounded calibration method blocks."""

    def __init__(
        self,
        message: str,
        *,
        rows: Sequence[Mapping[str, object]] = (),
        traces: Sequence[Mapping[str, object]] = (),
        partial_fold: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.rows = tuple(dict(row) for row in rows)
        self.traces = tuple(dict(trace) for trace in traces)
        self.partial_fold = dict(partial_fold) if partial_fold is not None else None


def temperature_method(temperature: float) -> str:
    return f"temperature_{temperature:.2f}".replace(".", "p")


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def verify_temperature_one_reproduction(
    replayed: Mapping[str, object],
    original: Mapping[str, object],
) -> None:
    """Fail closed unless temperature 1.0 reproduces Phase 2 evidence."""

    fields = (
        "eligible",
        "successes",
        "asr_at_budgets",
        "asr_query_auc",
        "eligible_sample_ids_sha256",
        "action_histogram",
        "by_victim",
        "max_total_target_calls",
        "initialization_included",
        "operator_digest",
        "normalized_action_entropy",
        "sampling_temperature",
        "deterministic_actions",
    )
    if any(
        _canonical(replayed.get(field)) != _canonical(original.get(field))
        for field in fields
    ):
        raise ValueError("temperature 1.0 did not reproduce verified Phase 2 evidence")


def _row_reproduction_record(row: ResearchResultRow) -> dict[str, object]:
    return {key: value for key, value in asdict(row).items() if key != "method"}


def verify_temperature_one_rows(
    replayed: Sequence[ResearchResultRow],
    original: Sequence[ResearchResultRow],
) -> str:
    """Verify exact per-sample outcomes and action traces for temperature 1."""

    replayed_records = sorted(
        (_row_reproduction_record(row) for row in replayed),
        key=lambda row: (str(row["victim_id"]), str(row["sample_id"])),
    )
    original_records = sorted(
        (_row_reproduction_record(row) for row in original),
        key=lambda row: (str(row["victim_id"]), str(row["sample_id"])),
    )
    encoded = _canonical(replayed_records)
    if encoded != _canonical(original_records):
        raise ValueError(
            "temperature 1.0 did not reproduce exact per-sample source rows"
        )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _fold_mapping(fold: object) -> Mapping[str, object]:
    if not isinstance(fold, Mapping):
        raise ValueError("calibration fold must be an object")
    return fold


def _source_indices(
    run: Mapping[str, object],
    config: MacPilotConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> tuple[int, ...]:
    split = build_cifar_split(
        train_dataset.targets,
        test_dataset.targets,
        config.victim_train_images,
        config.policy_train_images,
        config.source_validation_images,
        config.outer_test_images,
        config.split_seed if config.split_seed is not None else config.seed,
    )
    if split.digest != run.get("split_digest"):
        raise ValueError("reconstructed Phase 2 CIFAR split digest mismatch")
    _, _, source_indices = disjoint_balanced_subsets(
        train_dataset,
        split.source_validation,
        (
            config.victim_validation_images,
            config.behavior_cloning_validation_episodes,
            config.source_evaluation_images,
        ),
    )
    roles = _mapping(
        run.get("data_role_digests"),
        label="Phase 2 data-role digests",
    )
    if indices_digest(source_indices) != roles.get("source_gate"):
        raise ValueError("reconstructed Phase 2 source cohort mismatch")
    return source_indices


def _load_policy(
    fold: Mapping[str, object],
    run: Mapping[str, object],
    config: MacPilotConfig,
    device: torch.device,
):
    attack = config.attack_config()
    checkpoint = Path(str(fold["checkpoint_path"])).resolve()
    policy, metadata = load_recurrent_checkpoint(
        checkpoint,
        device,
        expected_observation_dim=attack.recurrent_observation_dim,
        expected_action_dim=attack.action_dim,
        expected_hidden_dim=config.hidden_dim,
        expected_actor_mode="action_conditioned",
    )
    policy_record = _mapping(run.get("policy"), label="Phase 2 policy")
    if (
        metadata.get("seed") != config.seed
        or metadata.get("split_digest") != run.get("split_digest")
        or metadata.get("kind") != "soft_gradient_bc_action_conditioned_groupdro_ppo"
        or int(metadata.get("completed_episodes", -1)) != config.policy_episodes
        or metadata.get("fingerprint") != policy_record.get("training_fingerprint")
        or policy.persistent_digest() != fold.get("persistent_digest")
        or sha256_file(checkpoint) != fold.get("checkpoint_sha256")
    ):
        raise ValueError("Phase 2 policy checkpoint metadata mismatch")
    return policy


def _load_source_victims(
    fold: Mapping[str, object],
    run: Mapping[str, object],
    config: MacPilotConfig,
    device: torch.device,
) -> dict[str, dict[str, tuple[tuple[str, nn.Module], ...]]]:
    source_families = tuple(fold.get("source_families", ()))
    expected_count = (
        config.source_instances_per_family + config.source_holdout_instances_per_family
    )
    population = build_cifar_victim_population(
        config.victim_seed if config.victim_seed is not None else config.seed,
        {family: expected_count for family in source_families},
        families=source_families,
        profile=config.victim_profile,
    )
    records = _mapping(
        run.get("victim_instances"),
        label="Phase 2 victim records",
    )
    cache_digest = run.get("victim_cache_digest")
    cache_contract = run.get("victim_cache_contract")
    runs_root = Path(str(fold["runs_root"])).resolve()
    loaded: dict[str, tuple[tuple[str, nn.Module], ...]] = {}
    for family in source_families:
        family_records = records.get(family)
        instances = population.get(family)
        if (
            not isinstance(family_records, list)
            or instances is None
            or len(family_records) != expected_count
            or len(instances) != expected_count
        ):
            raise ValueError("Phase 2 source victim count mismatch")
        family_victims: list[tuple[str, nn.Module]] = []
        for (victim_id, model), raw_record in zip(
            instances,
            family_records,
        ):
            record = _mapping(
                raw_record,
                label="Phase 2 source victim",
            )
            if victim_id != record.get("victim_id"):
                raise ValueError("Phase 2 source victim ID mismatch")
            checkpoint = resolve_descendant(
                runs_root,
                Path(str(record.get("checkpoint"))),
                label="Phase 2 source victim checkpoint",
            )
            if sha256_file(checkpoint) != record.get("checkpoint_sha256"):
                raise ValueError("Phase 2 victim checksum mismatch")
            metadata = load_model_checkpoint(checkpoint, model, device)
            if (
                metadata.get("fingerprint") != cache_digest
                or metadata.get("cache_contract") != cache_contract
                or metadata.get("family") != family
                or metadata.get("instance_index") != record.get("instance_index")
                or metadata.get("training_seed") != record.get("training_seed")
            ):
                raise ValueError("Phase 2 victim metadata mismatch")
            family_victims.append((victim_id, freeze_model(model.to(device))))
        loaded[family] = tuple(family_victims)
    return {
        "exact_source": {
            family: victims[: config.source_instances_per_family]
            for family, victims in loaded.items()
        },
        "seen_family_new_instance": {
            family: victims[config.source_instances_per_family :]
            for family, victims in loaded.items()
        },
    }


def _pool_metrics(
    metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not metrics:
        raise ValueError("cannot pool an empty calibration metric set")
    eligible = sum(int(item["eligible"]) for item in metrics)
    if eligible <= 0:
        raise ValueError("calibration metric set has no eligible images")
    successes = sum(int(item["successes"]) for item in metrics)
    condition_asr = tuple(
        int(item["successes"]) / int(item["eligible"]) for item in metrics
    )
    return {
        "eligible": eligible,
        "successes": successes,
        "asr": sum(condition_asr) / len(condition_asr),
        "auc": sum(float(item["asr_query_auc"]) for item in metrics) / len(metrics),
        "pooled_asr": successes / eligible,
        "frozen": all(item.get("frozen") is True for item in metrics),
        "source_model_calls": sum(
            int(item.get("source_model_calls", 0)) for item in metrics
        ),
    }


def _enriched_row(
    row: ResearchResultRow,
    *,
    heldout_family: str,
    source_slice: str,
    temperature: float,
) -> dict[str, object]:
    return {
        **asdict(row),
        "heldout_family": heldout_family,
        "source_slice": source_slice,
        "temperature": temperature,
        "hidden_target_calls": 0,
    }


def evaluate_calibration_fold(
    *,
    fold: object,
    request: Phase2CalibrationRequest,
    train_dataset: Dataset,
    test_dataset: Dataset,
    absolute_deadline: float,
    clock: Callable[[], float],
    progress: Callable[[str], None],
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Replay all five temperatures on the two verified source slices."""

    selected = _fold_mapping(fold)
    run = _mapping(
        selected.get("run_manifest"),
        label="Phase 2 run manifest",
    )
    config = MacPilotConfig(
        **dict(
            _mapping(
                run.get("config"),
                label="Phase 2 run config",
            )
        )
    )
    if (
        config.seed != 17
        or config.query_budget != 50
        or config.source_instances_per_family != 2
        or config.source_holdout_instances_per_family != 1
        or config.policy_actor_mode != "action_conditioned"
        or not math.isclose(config.policy_evaluation_temperature, 1.0)
    ):
        raise ValueError("Phase 2 fold does not match the calibration contract")
    heldout = str(selected.get("heldout_family", ""))
    source_families = tuple(selected.get("source_families", ()))
    if heldout in source_families or len(source_families) != 2:
        raise ValueError("calibration fold exposes the held-out family")

    device = torch.device(request.device)
    attack = config.attack_config()
    policy = _load_policy(selected, run, config, device)
    persistent_digest = policy.persistent_digest()
    populations = _load_source_victims(
        selected,
        run,
        config,
        device,
    )
    indices = _source_indices(
        run,
        config,
        train_dataset,
        test_dataset,
    )
    samples = dataset_samples(train_dataset, indices)
    source_evaluation = _mapping(
        run.get("source_evaluation"),
        label="verified Phase 2 source evaluation",
    )
    policy_record = _mapping(run.get("policy"), label="Phase 2 policy")
    checkpoints = _mapping(
        policy_record.get("checkpoints"),
        label="Phase 2 policy checkpoints",
    )
    main_checkpoint = _mapping(
        checkpoints.get("main"),
        label="Phase 2 main checkpoint",
    )
    method_id = main_checkpoint.get("method_id")
    if not isinstance(method_id, str) or not method_id:
        raise ValueError("Phase 2 learned method ID is missing")
    stochastic_method = f"{method_id}_stochastic"
    original_source_rows = read_jsonl(
        Path(str(selected.get("score_rows_path"))).resolve()
    )
    if not original_source_rows:
        raise ValueError("verified Phase 2 source rows are empty")

    result_rows: list[dict[str, object]] = []
    result_traces: list[dict[str, object]] = []
    condition_summaries: dict[str, dict[str, object]] = {}
    temperature_blocks: dict[float, list[Mapping[str, object]]] = {
        temperature: [] for temperature in request.temperatures
    }
    score_blocks: list[Mapping[str, object]] = []
    completed_method_blocks: list[str] = []
    temperature_one_row_digests: list[str] = []

    def deadline_error(message: str) -> CalibrationDeadlineReached:
        return CalibrationDeadlineReached(
            message,
            rows=result_rows,
            traces=result_traces,
            partial_fold={
                "seed": config.seed,
                "heldout_family": heldout,
                "source_families": list(source_families),
                "policy_checkpoint_sha256": selected["checkpoint_sha256"],
                "policy_persistent_digest": persistent_digest,
                "completed_method_blocks": list(completed_method_blocks),
                "source_model_calls": sum(
                    int(row["total_target_calls"]) for row in result_rows
                ),
                "raw_evidence_audited": True,
                "target_calls": 0,
                "target_evaluation_performed": False,
                "complete": False,
            },
        )

    for slice_offset, (source_slice, families) in enumerate(populations.items()):
        verified_slice = _mapping(
            source_evaluation.get(source_slice),
            label=f"Phase 2 source slice {source_slice}",
        )
        for family_offset, family in enumerate(source_families):
            victims = families[family]
            verified_family = _mapping(
                verified_slice.get(family),
                label=f"Phase 2 source condition {source_slice}/{family}",
            )
            verified_score = _mapping(
                verified_family.get("score_greedy"),
                label="verified score-greedy metrics",
            )
            score_eligible = int(verified_score.get("eligible", 0))
            if score_eligible <= 0:
                raise ValueError("verified score control has no eligible cohort")
            score_asr = int(verified_score.get("successes", 0)) / score_eligible
            score_auc = float(verified_score.get("asr_query_auc", -1))
            score_blocks.append(verified_score)
            condition_temperatures: dict[str, Mapping[str, object]] = {}
            condition_audits: dict[str, Mapping[str, object]] = {}
            condition_seed = (
                config.seed + 800_000 + 10_000 * slice_offset + family_offset
            )
            for temperature in request.temperatures:
                if clock() >= absolute_deadline:
                    raise deadline_error("calibration method-block deadline reached")
                method = temperature_method(temperature)
                progress(
                    f"[calibration] {heldout}/seed-{config.seed}/"
                    f"{source_slice}/{family}: temperature={temperature:.2f}"
                )
                controlled = FrozenTemperaturePolicy(policy, temperature)
                rows, traces, summary = evaluate_method_set(
                    {method: (controlled, False)},
                    victims,
                    samples,
                    indices,
                    attack,
                    condition_seed,
                    family,
                    progress,
                    trace_samples_per_method=1,
                    stochastic_seed_namespace=stochastic_method,
                )
                replayed_metrics = _mapping(
                    summary.get(method),
                    label=f"replayed {method} metrics",
                )
                metrics = {
                    **replayed_metrics,
                    "source_model_calls": sum(row.total_target_calls for row in rows),
                    "asr": int(replayed_metrics["successes"])
                    / int(replayed_metrics["eligible"]),
                    "auc": float(replayed_metrics["asr_query_auc"]),
                }
                if (
                    metrics.get("frozen") is not True
                    or metrics.get("policy_digest_before") != persistent_digest
                    or metrics.get("policy_digest_after") != persistent_digest
                    or policy.persistent_digest() != persistent_digest
                ):
                    raise ValueError("calibration mutated the frozen policy")
                if metrics.get("eligible_sample_ids_sha256") != verified_score.get(
                    "eligible_sample_ids_sha256"
                ):
                    raise ValueError(
                        "calibration and score controls use different cohorts"
                    )
                metrics = {
                    **metrics,
                    "asr_gain_vs_score": float(metrics["asr"]) - score_asr,
                    "auc_gain_vs_score": float(metrics["auc"]) - score_auc,
                    "cohort_matched_score": True,
                }
                victim_ids = {victim_id for victim_id, _ in victims}
                expected_sample_ids = {
                    f"cifar10:{family}:{victim_id}:{sample_index}"
                    for victim_id in victim_ids
                    for sample_index in indices
                }
                audit = audit_evaluation(
                    rows,
                    {method: metrics},
                    attack,
                    expected_sample_ids=expected_sample_ids,
                    expected_victim_ids=victim_ids,
                )
                if audit.get("passed") is not True:
                    raise ValueError("calibration raw rows failed the evaluation audit")
                if math.isclose(temperature, 1.0):
                    original = _mapping(
                        verified_family.get(stochastic_method),
                        label="verified Phase 2 stochastic metrics",
                    )
                    verify_temperature_one_reproduction(metrics, original)
                    original_rows = tuple(
                        row
                        for row in original_source_rows
                        if row.method == stochastic_method
                        and row.victim_family == family
                        and row.victim_id in victim_ids
                    )
                    temperature_one_row_digests.append(
                        verify_temperature_one_rows(rows, original_rows)
                    )
                condition_temperatures[str(temperature)] = dict(metrics)
                condition_audits[str(temperature)] = audit
                temperature_blocks[temperature].append(metrics)
                result_rows.extend(
                    _enriched_row(
                        row,
                        heldout_family=heldout,
                        source_slice=source_slice,
                        temperature=temperature,
                    )
                    for row in rows
                )
                result_traces.extend(
                    {
                        **trace,
                        "heldout_family": heldout,
                        "source_slice": source_slice,
                        "temperature": temperature,
                        "hidden_target_calls": 0,
                    }
                    for trace in traces
                )
                completed_method_blocks.append(f"{source_slice}/{family}/{method}")
                if clock() >= absolute_deadline:
                    raise deadline_error(
                        "calibration deadline crossed after a method block"
                    )
            condition_summaries[f"{source_slice}/{family}"] = {
                "source_slice": source_slice,
                "family": family,
                "condition_seed": condition_seed,
                "indices_sha256": indices_digest(indices),
                "victim_ids": [victim_id for victim_id, _ in victims],
                "score_greedy": dict(verified_score),
                "temperatures": condition_temperatures,
                "raw_row_audits": condition_audits,
            }

    pooled_score = _pool_metrics(score_blocks)
    pooled_temperatures = {
        temperature: _pool_metrics(blocks)
        for temperature, blocks in temperature_blocks.items()
    }
    if len(temperature_one_row_digests) != 4:
        raise ValueError("temperature 1.0 did not reproduce all four source conditions")
    summary = {
        "seed": config.seed,
        "heldout_family": heldout,
        "source_families": list(source_families),
        "policy_checkpoint_sha256": selected["checkpoint_sha256"],
        "policy_persistent_digest": persistent_digest,
        "score_greedy": {
            **pooled_score,
            "source_model_calls": 0,
            "reused_from_verified_phase2_evidence": True,
        },
        "temperatures": {
            str(temperature): {
                **metrics,
                "asr_gain_vs_score": float(metrics["asr"]) - float(pooled_score["asr"]),
                "auc_gain_vs_score": float(metrics["auc"]) - float(pooled_score["auc"]),
            }
            for temperature, metrics in pooled_temperatures.items()
        },
        "conditions": condition_summaries,
        "temperature_one_reproduced": True,
        "temperature_one_row_sha256": hashlib.sha256(
            "\n".join(sorted(temperature_one_row_digests)).encode("utf-8")
        ).hexdigest(),
        "raw_evidence_audited": True,
        "source_model_calls": sum(row["total_target_calls"] for row in result_rows),
        "target_calls": 0,
        "target_evaluation_performed": False,
        "complete": True,
    }
    return summary, result_rows, result_traces


__all__ = (
    "CalibrationDeadlineReached",
    "evaluate_calibration_fold",
    "temperature_method",
    "verify_temperature_one_reproduction",
)
