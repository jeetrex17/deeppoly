"""Fresh, source-only data roles for the Phase 2 D2 GroupDRO study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from .cifar_config import MacPilotConfig
from .cifar_data import (
    build_cifar_split,
    dataset_samples,
    disjoint_balanced_subsets,
)
from .phase2_residual_d1 import ResidualD1Request
from .phase2_residual_d1_source import (
    D1SourceContext,
    load_d1_source_context,
)
from .phase2_residual_d2 import (
    D2_COMPETENCE_IMAGES,
    D2_EVALUATION_IMAGES,
    D2_GROUPDRO_TRAIN_IMAGES,
    D2_THRESHOLD_IMAGES,
    D2_VISITED_POLICY_TRAIN_IMAGES,
    ResidualD2Request,
    D2SourceRole,
    D2SourceRoles,
    validate_d2_role_exclusions,
)


@dataclass(frozen=True)
class D2SourceContext:
    """Verified victims plus historically fresh D2 source-only roles."""

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
    train_samples: tuple[tuple[torch.Tensor, int], ...]
    threshold_samples: tuple[tuple[torch.Tensor, int], ...]
    competence_samples: tuple[tuple[torch.Tensor, int], ...]
    evaluation_samples: tuple[tuple[torch.Tensor, int], ...]
    roles: D2SourceRoles
    forbidden_policy_indices: tuple[int, ...]
    forbidden_source_validation_indices: tuple[int, ...]
    source_manifest_sha256: str
    dataset_content_sha256: str
    victim_cache_digest: str


def _d1_context_request(request: ResidualD2Request) -> ResidualD1Request:
    """Build the sealed fold-17 request used only to load verified source assets."""

    return ResidualD1Request(
        source_manifest=request.source_manifest,
        source_root=request.source_root,
        output_dir=request.output_dir / ".d1-source-context",
        data_root=request.data_root,
        device=request.device,
        download=request.download,
    )


def _fresh_roles(
    base: D1SourceContext,
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> tuple[D2SourceRoles, tuple[int, ...], tuple[int, ...]]:
    config = base.config
    split = build_cifar_split(
        train_dataset.targets,
        test_dataset.targets,
        config.victim_train_images,
        config.policy_train_images,
        config.source_validation_images,
        config.outer_test_images,
        config.split_seed if config.split_seed is not None else config.seed,
    )
    if split.digest != base.run_manifest.get("split_digest"):
        raise ValueError("D2 reconstructed CIFAR split digest mismatch")

    historical_policy, train = disjoint_balanced_subsets(
        train_dataset,
        split.policy_train,
        (D2_VISITED_POLICY_TRAIN_IMAGES, D2_GROUPDRO_TRAIN_IMAGES),
    )
    (
        victim_validation,
        bc_validation,
        historical_source_evaluation,
        threshold,
        competence,
        evaluation,
    ) = disjoint_balanced_subsets(
        train_dataset,
        split.source_validation,
        (
            config.victim_validation_images,
            config.behavior_cloning_validation_episodes,
            config.source_evaluation_images,
            D2_THRESHOLD_IMAGES,
            D2_COMPETENCE_IMAGES,
            D2_EVALUATION_IMAGES,
        ),
    )
    forbidden_validation = tuple(
        index
        for role in (victim_validation, bc_validation, historical_source_evaluation)
        for index in role
    )
    if not set(base.train_indices).issubset(set(historical_policy)):
        raise ValueError("D2 historical policy exclusion does not contain D1 training")
    if set(base.threshold_indices) | set(base.competence_indices) != set(
        bc_validation
    ):
        raise ValueError("D2 historical BC-validation exclusion does not match D1")
    if set(base.evaluation_indices) | set(base.ppo_evaluation_indices) != set(
        historical_source_evaluation
    ):
        raise ValueError("D2 historical source-evaluation exclusion does not match D1")

    roles = D2SourceRoles(
        groupdro_training=D2SourceRole(
            "groupdro_training",
            "policy_train",
            tuple(train),
        ),
        threshold_selection=D2SourceRole(
            "threshold_selection",
            "source_validation",
            tuple(threshold),
        ),
        competence_gate=D2SourceRole(
            "competence_gate",
            "source_validation",
            tuple(competence),
        ),
        evaluation=D2SourceRole(
            "evaluation",
            "source_validation",
            tuple(evaluation),
        ),
    )
    validate_d2_role_exclusions(
        roles,
        forbidden_policy_train_indices=historical_policy,
        forbidden_source_validation_indices=forbidden_validation,
    )
    # CIFAR train and test indices belong to different index spaces.  D2 roles
    # are allocated exclusively from the train-side policy/source-validation
    # partitions, so comparing their raw integer labels to ``outer_test``
    # would create false overlap whenever the same number appears in both
    # datasets.
    if (
        set(roles.groupdro_training.sample_ids) & set(historical_policy)
        or any(
            set(role.sample_ids) & set(forbidden_validation)
            for role in (
                roles.threshold_selection,
                roles.competence_gate,
                roles.evaluation,
            )
        )
    ):
        raise ValueError("D2 roles touched historical source data")
    return roles, tuple(historical_policy), forbidden_validation


def load_d2_source_context(
    request: ResidualD2Request,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_content_sha256: str,
) -> D2SourceContext:
    """Load sealed source victims and allocate untouched D2 image roles."""

    base = load_d1_source_context(
        _d1_context_request(request),
        train_dataset,
        test_dataset,
        dataset_content_sha256=dataset_content_sha256,
    )
    roles, forbidden_policy, forbidden_validation = _fresh_roles(
        base,
        train_dataset,
        test_dataset,
    )
    return D2SourceContext(
        fold=dict(base.fold),
        run_manifest=dict(base.run_manifest),
        config=base.config,
        source_families=base.source_families,
        teacher_victims={
            family: tuple(victims)
            for family, victims in base.teacher_victims.items()
        },
        evaluation_victims={
            family: tuple(victims)
            for family, victims in base.evaluation_victims.items()
        },
        train_indices=roles.groupdro_training.sample_ids,
        threshold_indices=roles.threshold_selection.sample_ids,
        competence_indices=roles.competence_gate.sample_ids,
        evaluation_indices=roles.evaluation.sample_ids,
        train_samples=dataset_samples(
            train_dataset,
            roles.groupdro_training.sample_ids,
        ),
        threshold_samples=dataset_samples(
            train_dataset,
            roles.threshold_selection.sample_ids,
        ),
        competence_samples=dataset_samples(
            train_dataset,
            roles.competence_gate.sample_ids,
        ),
        evaluation_samples=dataset_samples(
            train_dataset,
            roles.evaluation.sample_ids,
        ),
        roles=roles,
        forbidden_policy_indices=forbidden_policy,
        forbidden_source_validation_indices=forbidden_validation,
        source_manifest_sha256=base.source_manifest_sha256,
        dataset_content_sha256=base.dataset_content_sha256,
        victim_cache_digest=base.victim_cache_digest,
    )


__all__ = ("D2SourceContext", "load_d2_source_context")
