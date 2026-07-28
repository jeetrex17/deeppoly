"""Time-bounded, source-only Phase 2 screening orchestration.

This module intentionally has no target-phase argument or target-evaluation
callable. A positive screen can only authorize a larger source-only run.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

from torch.utils.data import Dataset

from .artifacts import sha256_file
from .cifar_config import MacPilotConfig
from .cifar_pilot import (
    _code_digest,
    run_cifar_pilot_from_datasets,
)
from .gpu_environment import (
    capture_runtime_environment,
    require_clean_protocol_tree,
)
from .paths import resolve_descendant, resolve_within_repository
from .phase2_cache import (
    expected_victim_cache_fingerprint,
    mirror_verified_victim_cache,
)
from .phase2_config import Phase2ScreenConfig
from .phase2_promotion import screen_promotion_decision
from .phase2_validation import (
    load_resumable_screen_manifest,
    validate_source_run_artifacts,
)
from .reproducibility import tree_digest
from .verified_artifacts import write_verified_json


_PHASE2_PROTOCOL_PATH = (
    Path(__file__).parents[1]
    / "docs"
    / "research"
    / "cifar10_rtx_phase2_protocol.md"
)

_LOCKED_SCREEN_FIELDS: dict[str, object] = {
    "name": "cifar10-rtx-phase2-action-conditioned-screen",
    "base_config": "configs/rl_transfer/cifar10_rtx_phase2_base.json",
    "output_dir": "output/rl_transfer/cifar10_rtx_phase2_screen",
    "device": "cuda",
    "seeds": (17,),
    "target_families": (
        "classical_cnn",
        "modern_cnn",
        "transformer",
    ),
    "resume": True,
    "split_seed": 20260727,
    "victim_seed": 1000000,
    "victim_cache_source": (
        "output/rl_transfer/cifar10_rtx_publication/"
        "cifar10-rtx-publication/runs/victim_cache"
    ),
    "victim_study_manifest": (
        "output/rl_transfer/cifar10_rtx_publication/"
        "cifar10-rtx-publication/study_manifest.json"
    ),
    "victim_study_manifest_sha256": (
        "791140871a987ec400cca083aea9b1192d8e73f2a5e70e5504dcfcae7f85911d"
    ),
    "require_verified_victim_cache": True,
    "max_wall_clock_minutes": 60,
    "estimated_minutes_per_cell": 12,
    "minimum_mean_bc_accuracy_gain": 0.01,
    "minimum_mean_bc_nll_improvement": 0.02,
    "minimum_mean_score_asr_gain": 0.01,
    "minimum_mean_score_auc_gain": 0.005,
    "minimum_positive_condition_fraction": 0.67,
}

_LOCKED_BASE_FIELDS: dict[str, object] = {
    "dataset": "CIFAR-10",
    "device": "cuda",
    "download": True,
    "data_root": "data/cifar10",
    "seed": 17,
    "victim_train_images": 40000,
    "policy_train_images": 4000,
    "source_validation_images": 1000,
    "outer_test_images": 1000,
    "victim_epochs": 50,
    "policy_episodes": 600,
    "policy_update_block": 50,
    "policy_learning_rate": 0.0003,
    "policy_entropy_weight": 0.0001,
    "policy_update_epochs": 6,
    "query_budget": 50,
    "grid_size": 4,
    "epsilon": 8 / 255,
    "step_size": 2 / 255,
    "batch_size": 256,
    "num_workers": 4,
    "hidden_dim": 256,
    "victim_learning_rate": 0.001,
    "target_family": "transformer",
    "source_instances_per_family": 2,
    "source_holdout_instances_per_family": 1,
    "target_instances_per_family": 3,
    "victim_profile": "research",
    "reward_mode": "margin_delta",
    "margin_reward_scale": 5.0,
    "terminal_success_bonus": 2.0,
    "query_penalty": 0.01,
    "rollback_on_non_improvement": True,
    "action_history_features": True,
    "image_patch_features": True,
    "behavior_cloning_teacher": "gradient",
    "behavior_cloning_episodes": 600,
    "behavior_cloning_validation_episodes": 100,
    "behavior_cloning_epochs": 25,
    "behavior_cloning_batch_size": 512,
    "behavior_cloning_candidates": 8,
    "behavior_cloning_steps": 49,
    "train_ablation_policies": False,
    "victim_validation_images": 500,
    "source_evaluation_images": 100,
    "classical_cnn_min_accuracy": 0.85,
    "modern_cnn_min_accuracy": 0.8,
    "transformer_min_accuracy": 0.75,
    "minimum_source_asr_gain": 0.05,
    "minimum_source_auc_gain": 0.02,
    "source_entropy_min": 0.1,
    "source_entropy_max": 0.95,
    "query_trace_samples_per_method": 5,
    "policy_actor_mode": "action_conditioned",
    "image_patch_feature_mode": "statistics",
    "behavior_cloning_soft_temperature": 0.5,
    "policy_evaluation_temperature": 1.0,
}


def _json_record(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _config_record(config: Phase2ScreenConfig) -> dict[str, object]:
    record = _json_record(asdict(config))
    if not isinstance(record, dict):
        raise RuntimeError("Phase 2 configuration serialization failed")
    return record


def validate_phase2_base_contract(
    screen: Phase2ScreenConfig,
    base: MacPilotConfig,
) -> None:
    """Reject expensive, target-capable, or incomparable screen cells."""

    errors: list[str] = []
    for field, expected in _LOCKED_SCREEN_FIELDS.items():
        if getattr(screen, field) != expected:
            errors.append(
                f"screen {field} must remain preregistered as {expected!r}"
            )
    for field, expected in _LOCKED_BASE_FIELDS.items():
        if getattr(base, field) != expected:
            errors.append(
                f"base {field} must remain preregistered as {expected!r}"
            )
    if base.device != screen.device or base.device != "cuda":
        errors.append("base device must be cuda")
    if base.research_valid is not False:
        errors.append("screen base cannot be marked research-valid")
    if base.query_budget != 50:
        errors.append("query budget must remain 50 including initialization")
    if abs(base.epsilon - 8 / 255) > 1e-12:
        errors.append("epsilon must remain 8/255")
    if abs(base.step_size - 2 / 255) > 1e-12:
        errors.append("step size must remain 2/255")
    if base.grid_size != 4:
        errors.append("the Phase 1 four-by-four action grid must be retained")
    if not base.rollback_on_non_improvement:
        errors.append("the matched rollback operator is required")
    if not base.action_history_features:
        errors.append("action-history features are required")
    if not base.image_patch_features:
        errors.append("patch image features are required")
    if base.behavior_cloning_teacher != "gradient":
        errors.append("the source-only gradient teacher is required")
    if not 1 <= base.behavior_cloning_episodes <= 2_000:
        errors.append("BC episodes must be between 1 and 2000")
    if not 1 <= base.behavior_cloning_epochs <= 40:
        errors.append("BC epochs must be between 1 and 40")
    if base.behavior_cloning_validation_episodes < 50:
        errors.append("at least 50 balanced BC validation episodes are required")
    if base.train_ablation_policies:
        errors.append("component ablations are forbidden in the short screen")
    if not 100 <= base.policy_episodes <= 2_000:
        errors.append("policy episodes must be between 100 and 2000")
    if base.policy_update_block > 250:
        errors.append("policy update blocks must be at most 250 episodes")
    if base.policy_update_epochs > 8:
        errors.append("PPO update epochs must be at most eight")
    if base.policy_evaluation_temperature != 1.0:
        errors.append(
            "Stage B evaluation temperature is locked at 1.0; "
            "Stage A is a Phase 1 diagnostic only"
        )
    if not 50 <= base.source_evaluation_images <= 100:
        errors.append("source evaluation must use 50 to 100 images")
    if base.victim_validation_images < 100:
        errors.append("victim validation requires at least 100 images")
    if base.source_holdout_instances_per_family < 1:
        errors.append("one unseen source-family instance is required")
    if base.source_instances_per_family < 1:
        errors.append("one fitted source instance is required")
    if base.target_instances_per_family != 3:
        errors.append("the fixed Phase 1 victim-bank cardinality is required")
    if (
        base.victim_validation_images
        + base.behavior_cloning_validation_episodes
        + base.source_evaluation_images
        > base.source_validation_images
    ):
        errors.append("source-side validation roles must remain disjoint")
    if errors:
        raise ValueError("; ".join(errors))


def load_validated_phase2_config(
    config_path: Path,
) -> tuple[Phase2ScreenConfig, MacPilotConfig, Path]:
    locked_path = resolve_within_repository(
        config_path,
        allowed_directory="configs/rl_transfer",
        label="Phase 2 screen config",
    )
    screen = Phase2ScreenConfig.from_json(locked_path)
    base_path = resolve_within_repository(
        screen.base_config,
        allowed_directory="configs/rl_transfer",
        label="Phase 2 base config",
    )
    base = MacPilotConfig.from_json(base_path)
    validate_phase2_base_contract(screen, base)
    return screen, base, locked_path


def _derived_config(
    screen: Phase2ScreenConfig,
    base: MacPilotConfig,
    *,
    target_family: str,
    seed: int,
) -> MacPilotConfig:
    return replace(
        base,
        name=f"{screen.name}-{target_family}-seed-{seed}",
        output_dir=f"{screen.output_dir}/runs",
        device=screen.device,
        seed=seed,
        target_family=target_family,
        split_seed=screen.split_seed,
        victim_seed=screen.victim_seed,
        train_ablation_policies=False,
    )


def _estimated_source_calls_per_cell(base: MacPilotConfig) -> int:
    bc_calls = (
        base.behavior_cloning_episodes
        + base.behavior_cloning_validation_episodes
    ) * (1 + base.behavior_cloning_steps)
    ppo_calls = base.policy_episodes * base.query_budget
    source_family_count = 2
    evaluated_victims_per_family = (
        base.source_instances_per_family
        + base.source_holdout_instances_per_family
    )
    evaluated_method_count = 6
    evaluation_calls = (
        source_family_count
        * evaluated_victims_per_family
        * base.source_evaluation_images
        * evaluated_method_count
        * base.query_budget
    )
    return bc_calls + ppo_calls + evaluation_calls


def build_phase2_dry_run(
    config: Phase2ScreenConfig,
    base: MacPilotConfig,
    *,
    completed_cells: int = 0,
) -> dict[str, object]:
    validate_phase2_base_contract(config, base)
    total_cells = len(config.target_families) * len(config.seeds)
    if (
        not isinstance(completed_cells, int)
        or isinstance(completed_cells, bool)
        or not 0 <= completed_cells <= total_cells
    ):
        raise ValueError("completed_cells is outside the Phase 2 grid")
    pending = total_cells - completed_cells
    estimated = pending * config.estimated_minutes_per_cell
    return {
        "schema_version": 1,
        "mode": "source_only_screen",
        "name": config.name,
        "families": list(config.target_families),
        "seeds": list(config.seeds),
        "total_cells": total_cells,
        "completed_cells": completed_cells,
        "pending_cells": pending,
        "estimated_minutes_per_cell": (
            config.estimated_minutes_per_cell
        ),
        "estimated_remaining_minutes": estimated,
        "maximum_invocation_minutes": (
            config.max_wall_clock_minutes
        ),
        "estimated_cells_fit_per_invocation": min(
            pending,
            max(
                1,
                int(
                    config.max_wall_clock_minutes
                    // config.estimated_minutes_per_cell
                ),
            ),
        ),
        "maximum_scheduled_source_calls_per_cell": (
            _estimated_source_calls_per_cell(base)
        ),
        "victim_training_expected": False,
        "verified_victim_cache_required": True,
        "victim_cache_source": config.victim_cache_source,
        "component_ablations": False,
        "policy_actor_mode": base.policy_actor_mode,
        "image_patch_feature_mode": base.image_patch_feature_mode,
        "behavior_cloning_target_mode": (
            "soft"
            if base.behavior_cloning_soft_temperature is not None
            else "hard"
        ),
        "behavior_cloning_soft_temperature": (
            base.behavior_cloning_soft_temperature
        ),
        "policy_evaluation_temperature": (
            base.policy_evaluation_temperature
        ),
        "target_evaluation_available": False,
        "promotion_scope": (
            "a positive result permits only a larger source-only screen"
        ),
        "protocol_sha256": sha256_file(_PHASE2_PROTOCOL_PATH),
    }


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    write_verified_json(path, payload)


def _manifest_payload(
    *,
    config: Phase2ScreenConfig,
    base_config_digest: str,
    code_digest: str,
    dataset_version: str,
    protocol_sha256: str,
    status: str,
    source_runs: Sequence[Mapping[str, object]],
    cache_reuse: Mapping[str, object],
    runtime_environment: Mapping[str, object],
    elapsed_seconds: float,
    error: str | None = None,
) -> dict[str, object]:
    decision = screen_promotion_decision(source_runs, config)
    payload: dict[str, object] = {
        "schema_version": 1,
        "name": config.name,
        "status": status,
        "research_valid": False,
        "publication_candidate": False,
        "config": _config_record(config),
        "base_config_digest": base_config_digest,
        "study_code_digest": code_digest,
        "dataset_version": dataset_version,
        "protocol_sha256": protocol_sha256,
        "runtime_environment": dict(runtime_environment),
        "victim_cache_reuse": dict(cache_reuse),
        "source_runs": list(source_runs),
        "screen_promotion_decision": decision,
        "target_evaluation_performed": False,
        "target_calls": 0,
        "elapsed_seconds": elapsed_seconds,
    }
    if error is not None:
        payload["errors"] = [error]
    return payload


def _checked_cache_reuse(
    config: Phase2ScreenConfig,
    base: MacPilotConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str,
    run_output_dir: Path,
) -> dict[str, object]:
    cache_source = resolve_within_repository(
        config.victim_cache_source,
        allowed_directory="output/rl_transfer",
        label="verified Phase 1 victim cache",
    )
    cache_destination = resolve_descendant(
        run_output_dir,
        "victim_cache",
        label="Phase 2 victim cache",
    )
    expected = expected_victim_cache_fingerprint(
        config,
        base,
        train_dataset,
        test_dataset,
        dataset_version=dataset_version,
    )
    study_manifest = resolve_within_repository(
        config.victim_study_manifest,
        allowed_directory="output/rl_transfer",
        label="pinned Phase 1 study manifest",
    )
    mirrored = mirror_verified_victim_cache(
        cache_source,
        cache_destination,
        study_manifest_path=study_manifest,
        expected_study_manifest_sha256=(
            config.victim_study_manifest_sha256
        ),
        expected_cache_fingerprint=expected,
    )
    available = mirrored.get("cache_fingerprints")
    if (
        not isinstance(available, list)
        or expected not in available
    ):
        raise ValueError(
            "verified Phase 1 victim cache does not match this run; "
            "refusing to retrain victims during the short screen"
        )
    return {
        **mirrored,
        "expected_fingerprint": expected,
        "exact_fingerprint_available": True,
    }


def run_phase2_screen_from_datasets(
    config: Phase2ScreenConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str,
    progress: Callable[[str], None] | None = None,
    runtime_environment: Mapping[str, object] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Run resumable source cells until complete or the time budget expires."""

    report = progress or (lambda _message: None)
    base = MacPilotConfig.from_json(
        resolve_within_repository(
            config.base_config,
            allowed_directory="configs/rl_transfer",
            label="Phase 2 base config",
        )
    )
    validate_phase2_base_contract(config, base)
    started = clock()
    code_digest = _code_digest()
    protocol_sha256 = sha256_file(_PHASE2_PROTOCOL_PATH)
    base_digest = base.digest()
    study_dir = resolve_within_repository(
        config.output_dir,
        allowed_directory="output/rl_transfer",
        label="Phase 2 output directory",
    )
    run_output_dir = resolve_descendant(
        study_dir,
        "runs",
        label="Phase 2 run directory",
    )
    manifest_path = resolve_descendant(
        study_dir,
        "screen_manifest.json",
        label="Phase 2 screen manifest",
    )
    cache_reuse = _checked_cache_reuse(
        config,
        base,
        train_dataset,
        test_dataset,
        dataset_version=dataset_version,
        run_output_dir=run_output_dir,
    )
    existing = load_resumable_screen_manifest(
        manifest_path,
        config=config,
        base_config_digest=base_digest,
        code_digest=code_digest,
        dataset_version=dataset_version,
        protocol_sha256=protocol_sha256,
    )
    source_runs = (
        list(existing["source_runs"])
        if existing is not None
        else []
    )
    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    for run in source_runs:
        family = str(run.get("target_family"))
        raw_seed = run.get("seed")
        if (
            not isinstance(raw_seed, int)
            or isinstance(raw_seed, bool)
        ):
            raise ValueError("resumed source cell seed is invalid")
        key = (family, raw_seed)
        if key in completed:
            raise ValueError("resumed source grid contains duplicates")
        if (
            family not in config.target_families
            or raw_seed not in config.seeds
        ):
            raise ValueError("resumed source grid contains an unknown cell")
        derived = _derived_config(
            config,
            base,
            target_family=family,
            seed=raw_seed,
        )
        validate_source_run_artifacts(
            run,
            derived_config=derived,
            run_output_dir=run_output_dir,
        )
        completed[key] = run
    deadline_seconds = config.max_wall_clock_minutes * 60
    environment = runtime_environment or {}
    for family in config.target_families:
        for seed in config.seeds:
            key = (family, seed)
            if key in completed:
                report(f"resumed completed source cell {family}/seed-{seed}")
                continue
            before_cell = clock()
            if before_cell - started >= deadline_seconds:
                payload = _manifest_payload(
                    config=config,
                    base_config_digest=base_digest,
                    code_digest=code_digest,
                    dataset_version=dataset_version,
                    protocol_sha256=protocol_sha256,
                    status="screen_deadline_reached",
                    source_runs=source_runs,
                    cache_reuse=cache_reuse,
                    runtime_environment=environment,
                    elapsed_seconds=before_cell - started,
                )
                _write_manifest(manifest_path, payload)
                return payload
            if _code_digest() != code_digest:
                raise RuntimeError(
                    "package code changed during Phase 2 screening"
                )
            if sha256_file(_PHASE2_PROTOCOL_PATH) != protocol_sha256:
                raise RuntimeError(
                    "Phase 2 protocol changed during screening"
                )
            report(f"source-only screen target={family} seed={seed}")
            derived = _derived_config(
                config,
                base,
                target_family=family,
                seed=seed,
            )
            try:
                run = run_cifar_pilot_from_datasets(
                    derived,
                    train_dataset,
                    test_dataset,
                    resume=True,
                    dataset_version=dataset_version,
                    progress=lambda message, fold=family, run_seed=seed: report(
                        f"[{fold}/seed-{run_seed}] {message}"
                    ),
                    evaluate_target=False,
                    source_victims_only=True,
                    victim_cache_only=True,
                    portable_paths=True,
                )
                validate_source_run_artifacts(
                    run,
                    derived_config=derived,
                    run_output_dir=run_output_dir,
                )
            except Exception as error:
                failed_at = clock()
                payload = _manifest_payload(
                    config=config,
                    base_config_digest=base_digest,
                    code_digest=code_digest,
                    dataset_version=dataset_version,
                    protocol_sha256=protocol_sha256,
                    status="screen_failed",
                    source_runs=source_runs,
                    cache_reuse=cache_reuse,
                    runtime_environment=environment,
                    elapsed_seconds=failed_at - started,
                    error=f"{type(error).__name__}: {error}",
                )
                _write_manifest(manifest_path, payload)
                raise
            source_runs.append(run)
            completed_at = clock()
            partial = _manifest_payload(
                config=config,
                base_config_digest=base_digest,
                code_digest=code_digest,
                dataset_version=dataset_version,
                protocol_sha256=protocol_sha256,
                status="screen_running",
                source_runs=source_runs,
                cache_reuse=cache_reuse,
                runtime_environment=environment,
                elapsed_seconds=completed_at - started,
            )
            _write_manifest(manifest_path, partial)
    finished_at = clock()
    result = _manifest_payload(
        config=config,
        base_config_digest=base_digest,
        code_digest=code_digest,
        dataset_version=dataset_version,
        protocol_sha256=protocol_sha256,
        status="screen_complete",
        source_runs=source_runs,
        cache_reuse=cache_reuse,
        runtime_environment=environment,
        elapsed_seconds=finished_at - started,
    )
    _write_manifest(manifest_path, result)
    return result


def run_phase2_screen(
    config_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Load CIFAR-10 and execute the source-only screen."""

    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError(
            "install the vision extra before running Phase 2"
        ) from error
    config, base, locked_config_path = load_validated_phase2_config(
        config_path
    )
    repository = locked_config_path.parents[2]
    require_clean_protocol_tree(repository)
    data_root = resolve_within_repository(
        base.data_root,
        allowed_directory="data",
        label="CIFAR data root",
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=ToTensor(),
        download=base.download,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=ToTensor(),
        download=base.download,
    )
    study_dir = resolve_within_repository(
        config.output_dir,
        allowed_directory="output/rl_transfer",
        label="Phase 2 output directory",
    )
    requirements_path = resolve_within_repository(
        "requirements/rtx-publication.txt",
        allowed_directory="requirements",
        label="RTX requirements",
    )
    environment = capture_runtime_environment(
        study_dir,
        repository,
        requirements_path,
    )
    dataset_version = (
        f"torchvision-{torchvision.__version__};"
        f"content-sha256={tree_digest(data_root)};"
        f"environment-sha256={environment['pip_freeze_sha256']}"
    )
    return run_phase2_screen_from_datasets(
        config,
        train_dataset,
        test_dataset,
        dataset_version=dataset_version,
        progress=progress,
        runtime_environment=environment,
    )
