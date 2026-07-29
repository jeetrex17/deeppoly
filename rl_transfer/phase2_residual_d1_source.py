"""Verified source context for the Phase 2 D1 residual-ranker diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.utils.data import Dataset

from .artifacts import sha256_file
from .cifar_config import MacPilotConfig
from .cifar_data import (
    balanced_subset_indices,
    build_cifar_split,
    dataset_samples,
    disjoint_balanced_subsets,
    indices_digest,
)
from .paths import resolve_descendant
from .phase2_calibration_evaluation import (
    _load_source_victims,
    _source_indices,
)
from .phase2_residual_d1 import (
    D1_SOURCE_FAMILIES,
    ResidualD1Request,
    validate_d1_attack_contract,
    validate_source_only_payload,
)
from .phase2_validation import validate_source_run_artifacts
from .verified_artifacts import load_verified_json


_DIGEST = re.compile(r"[0-9a-f]{64}")
D1_SOURCE_MANIFEST_SHA256 = (
    "efd96c5775187ac29fbd1453e3d1654d26373fc17b7c0d22b0e4955215a0e054"
)
_RUNTIME_FIELDS = {
    "python_version",
    "torch_version",
    "torchvision_version",
    "cuda_runtime_version",
    "cudnn_version",
    "gpu_name",
    "gpu_total_memory_bytes",
    "deterministic_algorithms",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "environment_sha256",
}


@dataclass(frozen=True)
class D1SourceContext:
    """Verified source-only datasets, victims, and artifact identities."""

    fold: Mapping[str, object]
    run_manifest: Mapping[str, object]
    config: MacPilotConfig
    source_families: tuple[str, str]
    teacher_victims: Mapping[str, tuple[tuple[str, nn.Module], ...]]
    evaluation_victims: Mapping[str, tuple[tuple[str, nn.Module], ...]]
    train_indices: tuple[int, ...]
    threshold_indices: tuple[int, ...]
    competence_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    ppo_evaluation_indices: tuple[int, ...]
    train_samples: tuple[tuple[torch.Tensor, int], ...]
    threshold_samples: tuple[tuple[torch.Tensor, int], ...]
    competence_samples: tuple[tuple[torch.Tensor, int], ...]
    evaluation_samples: tuple[tuple[torch.Tensor, int], ...]
    ppo_evaluation_samples: tuple[tuple[torch.Tensor, int], ...]
    role_audit: Mapping[str, object]
    source_manifest_sha256: str
    dataset_content_sha256: str
    victim_cache_digest: str


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_d1_role_indices(
    roles: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    """Validate all locked D1 development roles and return their identities."""

    expected_sizes = {
        "train": 200,
        "threshold": 50,
        "competence": 50,
        "d1a_evaluation": 50,
        "d1b_evaluation": 50,
    }
    if set(roles) != set(expected_sizes):
        raise ValueError("D1 role index names do not match the locked protocol")
    normalized: dict[str, tuple[int, ...]] = {}
    for role, expected_size in expected_sizes.items():
        values = tuple(roles[role])
        if len(values) != expected_size or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError(f"D1 {role} role has invalid indices or size")
        if len(set(values)) != len(values):
            raise ValueError(f"D1 {role} role contains duplicate indices")
        normalized = {**normalized, role: values}
    names = tuple(expected_sizes)
    if any(
        set(normalized[left]) & set(normalized[right])
        for offset, left in enumerate(names)
        for right in names[offset + 1 :]
    ):
        raise ValueError("D1 role indices overlap and are not pairwise disjoint")
    return {
        "pairwise_disjoint": True,
        "role_sizes": {role: len(normalized[role]) for role in names},
        "role_indices_sha256": {
            role: indices_digest(normalized[role]) for role in names
        },
    }


def _content_digest(dataset_version: str) -> str | None:
    match = re.search(
        r"(?:^|;)content-sha256=([0-9a-f]{64})(?:;|$)",
        dataset_version,
    )
    return match.group(1) if match is not None else None


def _portable_run_directory(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("D1 source run directory is missing")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.name != value:
        raise ValueError("D1 source run directory must be portable")
    return path


def _validated_runtime_environment(
    environment: Mapping[str, object],
) -> dict[str, object]:
    if set(environment) != _RUNTIME_FIELDS:
        raise ValueError("D1 runtime environment does not match the safe schema")
    payload = {
        key: environment[key]
        for key in sorted(_RUNTIME_FIELDS - {"environment_sha256"})
    }
    for value in payload.values():
        if isinstance(value, str) and (
            len(value) > 200 or "\n" in value or "\r" in value
        ):
            raise ValueError("D1 runtime environment contains unsafe text")
    expected = environment["environment_sha256"]
    if (
        not isinstance(expected, str)
        or _DIGEST.fullmatch(expected) is None
        or _canonical_digest(payload) != expected
    ):
        raise ValueError("D1 runtime environment digest mismatch")
    return {**payload, "environment_sha256": expected}


def load_residual_d1_source(
    request: ResidualD1Request,
) -> dict[str, object]:
    """Verify and expose only seed 17's modern-CNN-heldout source fold."""

    manifest_sha256 = sha256_file(request.source_manifest)
    if manifest_sha256 != D1_SOURCE_MANIFEST_SHA256:
        raise ValueError("D1 source manifest does not match the locked SHA-256")
    study = load_verified_json(request.source_manifest)
    if sha256_file(request.source_manifest) != manifest_sha256:
        raise ValueError("D1 source manifest changed while it was being verified")
    validate_source_only_payload(study, "D1 sealed source manifest")
    if (
        study.get("schema_version") != 1
        or study.get("status") != "screen_complete"
        or study.get("research_valid") is not False
        or study.get("target_calls") != 0
        or study.get("target_evaluation_performed") is not False
    ):
        raise ValueError("D1 source manifest violates the source-only seal")
    dataset_version = study.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("D1 source dataset identity is missing")
    raw_runs = study.get("source_runs")
    if not isinstance(raw_runs, list):
        raise ValueError("D1 source runs are missing")
    selected = tuple(
        _mapping(run, label="D1 source run")
        for run in raw_runs
        if isinstance(run, Mapping)
        and run.get("seed") == request.seed
        and run.get("target_family") == request.heldout_family
    )
    if len(selected) != 1:
        raise ValueError("D1 requires exactly one sealed source fold")
    run = selected[0]
    config = MacPilotConfig(
        **dict(_mapping(run.get("config"), label="D1 source config"))
    )
    runs_root = resolve_descendant(
        request.source_root,
        "runs",
        label="D1 source runs root",
    )
    validate_source_run_artifacts(
        run,
        derived_config=config,
        run_output_dir=runs_root,
    )
    run_dir = resolve_descendant(
        runs_root,
        _portable_run_directory(run.get("run_dir")),
        label="D1 selected source run",
    )
    policy = _mapping(run.get("policy"), label="D1 source policy")
    checkpoint_name = policy.get("checkpoint")
    if (
        not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise ValueError("D1 source policy checkpoint is invalid")
    checkpoint_path = resolve_descendant(
        run_dir,
        checkpoint_name,
        label="D1 source policy checkpoint",
    )
    checkpoint_sha256 = policy.get("checkpoint_sha256")
    persistent_digest = policy.get("persistent_digest")
    if (
        not isinstance(checkpoint_sha256, str)
        or _DIGEST.fullmatch(checkpoint_sha256) is None
        or sha256_file(checkpoint_path) != checkpoint_sha256
        or not isinstance(persistent_digest, str)
        or _DIGEST.fullmatch(persistent_digest) is None
    ):
        raise ValueError("D1 source policy identity failed")
    source_families = tuple(run.get("source_families", ()))
    if (
        set(source_families) != set(D1_SOURCE_FAMILIES)
        or request.heldout_family in source_families
    ):
        raise ValueError("D1 selected fold exposes the held-out family")
    fold = {
        "seed": request.seed,
        "heldout_family": request.heldout_family,
        "source_families": source_families,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "persistent_digest": persistent_digest,
        "score_rows_path": run_dir / "source_results.jsonl",
        "run_dir": run_dir,
        "runs_root": runs_root,
        "run_manifest": dict(run),
    }
    return {
        "manifest_path": request.source_manifest,
        "manifest_sha256": manifest_sha256,
        "dataset_version": dataset_version,
        "dataset_content_sha256": _content_digest(dataset_version),
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "authorizes_hidden_target_evaluation": False,
        "folds": (fold,),
    }


def load_d1_source_context(
    request: ResidualD1Request,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_content_sha256: str,
) -> D1SourceContext:
    """Construct disjoint roles and load only verified source victims."""

    source = load_residual_d1_source(request)
    validate_source_only_payload(source, "D1 verified source context")
    if (
        source.get("dataset_content_sha256") != dataset_content_sha256
        or source.get("target_calls") != 0
        or source.get("hidden_target_calls") != 0
        or source.get("target_evaluation_performed") is not False
        or source.get("hidden_target_evaluation_performed") is not False
        or source.get("authorizes_hidden_target_evaluation") is not False
    ):
        raise ValueError("D1 dataset or source-only seal failed")
    fold = _mapping(source["folds"][0], label="D1 selected fold")
    run = _mapping(fold.get("run_manifest"), label="D1 source run")
    config = MacPilotConfig(
        **dict(_mapping(run.get("config"), label="D1 source config"))
    )
    source_families = tuple(fold.get("source_families", ()))
    if config.seed != request.seed:
        raise ValueError("D1 attack/source-family contract failed")
    validate_d1_attack_contract(
        config.attack_config(),
        source_families,
    )

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
        raise ValueError("D1 reconstructed CIFAR split digest mismatch")
    train_indices = balanced_subset_indices(
        train_dataset,
        split.policy_train,
        request.bc_episodes,
    )
    _, bc_validation_indices, source_gate_indices = disjoint_balanced_subsets(
        train_dataset,
        split.source_validation,
        (
            config.victim_validation_images,
            config.behavior_cloning_validation_episodes,
            config.source_evaluation_images,
        ),
    )
    threshold_indices, competence_indices = disjoint_balanced_subsets(
        train_dataset,
        bc_validation_indices,
        (50, 50),
    )
    verified_source_gate = _source_indices(
        run,
        config,
        train_dataset,
        test_dataset,
    )
    if tuple(source_gate_indices) != tuple(verified_source_gate):
        raise ValueError("D1 source gate did not reproduce sealed evidence")
    evaluation_indices, ppo_evaluation_indices = disjoint_balanced_subsets(
        train_dataset,
        source_gate_indices,
        (request.source_images, request.source_images),
    )
    role_audit = validate_d1_role_indices(
        {
            "train": train_indices,
            "threshold": threshold_indices,
            "competence": competence_indices,
            "d1a_evaluation": evaluation_indices,
            "d1b_evaluation": ppo_evaluation_indices,
        }
    )

    populations = _load_source_victims(
        fold,
        run,
        config,
        torch.device(request.device),
    )
    exact = _mapping(populations.get("exact_source"), label="D1 teacher victims")
    heldout = _mapping(
        populations.get("seen_family_new_instance"),
        label="D1 source holdout victims",
    )
    teacher_victims = {family: tuple(exact[family]) for family in source_families}
    evaluation_victims = {family: tuple(heldout[family]) for family in source_families}
    if (
        any(len(victims) != 2 for victims in teacher_victims.values())
        or any(len(victims) != 1 for victims in evaluation_victims.values())
        or any(
            {victim_id for victim_id, _ in teacher_victims[family]}
            & {victim_id for victim_id, _ in evaluation_victims[family]}
            for family in source_families
        )
    ):
        raise ValueError("D1 teacher and held-out source victims are invalid")
    cache_digest = run.get("victim_cache_digest")
    if not isinstance(cache_digest, str) or _DIGEST.fullmatch(cache_digest) is None:
        raise ValueError("D1 victim cache identity is missing")

    return D1SourceContext(
        fold=dict(fold),
        run_manifest=dict(run),
        config=config,
        source_families=tuple(source_families),
        teacher_victims=teacher_victims,
        evaluation_victims=evaluation_victims,
        train_indices=tuple(train_indices),
        threshold_indices=tuple(threshold_indices),
        competence_indices=tuple(competence_indices),
        evaluation_indices=tuple(evaluation_indices),
        ppo_evaluation_indices=tuple(ppo_evaluation_indices),
        train_samples=dataset_samples(train_dataset, train_indices),
        threshold_samples=dataset_samples(train_dataset, threshold_indices),
        competence_samples=dataset_samples(train_dataset, competence_indices),
        evaluation_samples=dataset_samples(train_dataset, evaluation_indices),
        ppo_evaluation_samples=dataset_samples(
            train_dataset,
            ppo_evaluation_indices,
        ),
        role_audit=role_audit,
        source_manifest_sha256=str(source["manifest_sha256"]),
        dataset_content_sha256=dataset_content_sha256,
        victim_cache_digest=cache_digest,
    )


__all__ = (
    "D1SourceContext",
    "_mapping",
    "_validated_runtime_environment",
    "load_d1_source_context",
    "load_residual_d1_source",
    "validate_d1_role_indices",
)
