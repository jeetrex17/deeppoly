"""Verified D1a policy reconstruction and source-only D1b adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re

import torch
from torch import nn

from .artifacts import load_recurrent_checkpoint, sha256_file
from .phase2_residual_d1 import (
    D1_HELDOUT_FAMILY,
    D1_SEED,
    D1_SOURCE_FAMILIES,
    ResidualD1Request,
    validate_source_only_payload as _source_only,
)
from .phase2_residual_d1_cache import ResidualTeacherCache
from .phase2_residual_d1_evidence import (
    load_verified_jsonl_records,
    verify_d1_raw_evidence,
    verify_d1_recorded_summaries,
)
from .phase2_residual_d1_runner import _verify_complete_d1_children
from .phase2_residual_d1_source import validate_d1_role_indices
from .phase2_residual_d1_teacher import (
    D1_HIDDEN_DIM,
    D1_PRIOR_TEMPERATURE,
)
from .phase2_residual_d1b import (
    ResidualD1BSourceRole,
    ResidualD1BSourceRoles,
    ResidualD1BTrainingPayload,
    VerifiedD1AArtifacts,
)
from .phase2_residual_d1b_artifacts import (
    canonical_json_digest,
    clone_residual_policy,
)
from .residual_ranker import (
    ResidualRankerPolicy,
    evaluate_residual_ranker_examples,
    select_confidence_threshold,
)
from .verified_artifacts import load_verified_json


_DIGEST = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_METADATA_FIELDS = {
    "schema_version",
    "kind",
    "seed",
    "heldout_family",
    "source_manifest_sha256",
    "request_sha256",
    "confidence_threshold",
    "prior_temperature",
    "overrides_enabled",
    "target_calls",
    "hidden_target_calls",
    "target_evaluation_performed",
    "hidden_target_evaluation_performed",
    "target_evaluation_available",
    "authorizes_hidden_target_evaluation",
}


def _plain(value: object, label: str) -> object:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    decoded = _plain(dict(value), label)
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must be an object")
    return decoded


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        or (positive and float(value) <= 0)
    ):
        raise ValueError(f"{label} must be finite and valid")
    return float(value)


@dataclass(frozen=True)
class ResidualD1AArtifactBundle:
    """Expected architecture and dataset identity for a completed D1a."""

    request: ResidualD1Request
    dataset_content_sha256: str
    observation_dim: int
    action_dim: int
    hidden_dim: int = D1_HIDDEN_DIM
    device: str | torch.device = "cuda"
    hidden_target_calls: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.request, ResidualD1Request):
            raise TypeError("D1a artifact bundle requires a D1 request")
        _digest(self.dataset_content_sha256, "D1a dataset digest")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                self.observation_dim,
                self.action_dim,
                self.hidden_dim,
            )
        ):
            raise ValueError("D1a expected checkpoint dimensions are invalid")
        if self.hidden_target_calls != 0:
            raise ValueError("D1a artifact bundle must remain source-only")


def _checkpoint_metadata(
    raw: object,
    *,
    bundle: ResidualD1AArtifactBundle,
    threshold: Mapping[str, object],
) -> dict[str, object]:
    metadata = _mapping(raw, "D1a checkpoint metadata")
    _source_only(metadata, "D1a checkpoint metadata")
    if set(metadata) != _CHECKPOINT_METADATA_FIELDS:
        raise ValueError("D1a checkpoint metadata schema is not exact")
    request = bundle.request
    expected = {
        "schema_version": 3,
        "kind": "phase2_d1a_source_only_residual_ranker_bc",
        "seed": D1_SEED,
        "heldout_family": D1_HELDOUT_FAMILY,
        "source_manifest_sha256": sha256_file(request.source_manifest),
        "request_sha256": request.digest(),
        "confidence_threshold": threshold.get("threshold"),
        "prior_temperature": D1_PRIOR_TEMPERATURE,
        "overrides_enabled": threshold.get("overrides_enabled"),
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }
    if metadata != expected:
        raise ValueError("D1a checkpoint metadata binding mismatch")
    _number(metadata["confidence_threshold"], "D1a confidence threshold")
    _number(metadata["prior_temperature"], "D1a prior temperature", positive=True)
    if not isinstance(metadata["overrides_enabled"], bool):
        raise ValueError("D1a checkpoint override flag is invalid")
    return metadata


def verify_d1a_artifacts(
    raw_manifest: Mapping[str, object],
    raw_bundle: object,
) -> VerifiedD1AArtifacts:
    """Recompute D1a evidence and reconstruct the checksum-bound BC policy."""

    if not isinstance(raw_bundle, ResidualD1AArtifactBundle):
        raise TypeError("D1a verifier requires its artifact bundle")
    bundle = raw_bundle
    request = bundle.request
    manifest = _mapping(raw_manifest, "D1a manifest")
    _source_only(manifest, "D1a manifest")
    disk_manifest = load_verified_json(request.output_dir / "d1_manifest.json")
    if disk_manifest != manifest:
        raise ValueError("D1a manifest differs from its verified disk artifact")
    if (
        manifest.get("schema_version") != 3
        or manifest.get("name") != "phase2-d1a-residual-ranker-bc"
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request.digest()
        or manifest.get("source_manifest_sha256")
        != sha256_file(request.source_manifest)
        or manifest.get("dataset_content_sha256") != bundle.dataset_content_sha256
    ):
        raise ValueError("D1a request, source, or dataset binding mismatch")
    _verify_complete_d1_children(request, manifest)
    row_records = load_verified_jsonl_records(
        request.output_dir / "source_results.jsonl"
    )
    trace_records = load_verified_jsonl_records(
        request.output_dir / "source_query_traces.jsonl"
    )
    verified_evidence = verify_d1_raw_evidence(
        row_records,
        trace_records,
        expected_methods=("score_greedy", "residual_ranker_bc"),
    )
    verify_d1_recorded_summaries(
        verified_evidence,
        _mapping(manifest.get("source_evaluation"), "D1a source evaluation"),
    )
    checkpoint = _mapping(manifest.get("checkpoint"), "D1a checkpoint")
    if (
        set(checkpoint) != {"name", "sha256", "persistent_digest"}
        or checkpoint.get("name") != "residual_ranker_bc.pt"
    ):
        raise ValueError("D1a checkpoint manifest schema is invalid")
    checkpoint_sha256 = _digest(
        checkpoint.get("sha256"),
        "D1a checkpoint file digest",
    )
    expected_policy_digest = _digest(
        checkpoint.get("persistent_digest"),
        "D1a persistent policy digest",
    )
    checkpoint_path = request.output_dir / "residual_ranker_bc.pt"
    if checkpoint_path.is_symlink():
        raise ValueError("D1a checkpoint cannot be a symlink")
    backbone, raw_metadata = load_recurrent_checkpoint(
        checkpoint_path,
        bundle.device,
        expected_observation_dim=bundle.observation_dim,
        expected_action_dim=bundle.action_dim,
        expected_hidden_dim=bundle.hidden_dim,
        expected_actor_mode="action_conditioned",
    )
    threshold = _mapping(
        manifest.get("threshold_selection"),
        "D1a threshold selection",
    )
    metadata = _checkpoint_metadata(
        raw_metadata,
        bundle=bundle,
        threshold=threshold,
    )
    policy = ResidualRankerPolicy(
        backbone,
        confidence_threshold=float(metadata["confidence_threshold"]),
        prior_temperature=float(metadata["prior_temperature"]),
        overrides_enabled=bool(metadata["overrides_enabled"]),
    )
    if policy.persistent_digest() != expected_policy_digest:
        raise ValueError("D1a loaded BC policy digest mismatch")
    return VerifiedD1AArtifacts(
        bc_policy=policy,
        manifest_digest=canonical_json_digest(manifest),
        checkpoint_sha256=checkpoint_sha256,
        bc_policy_digest=expected_policy_digest,
        hidden_target_calls=0,
    )


@dataclass(frozen=True)
class ResidualD1BEvaluationPayload:
    source_victims: Mapping[str, tuple[tuple[str, nn.Module], ...]]
    source_samples: Sequence[tuple[torch.Tensor, int]]
    sample_ids: tuple[int, ...]
    attack_config: object
    hidden_target_calls: int = 0

    def __post_init__(self) -> None:
        if self.hidden_target_calls != 0:
            raise ValueError("D1b evaluation payload must remain source-only")
        if (
            set(self.source_victims) != set(D1_SOURCE_FAMILIES)
            or len(self.sample_ids) != 50
            or len(set(self.sample_ids)) != 50
            or len(self.source_samples) != 50
        ):
            raise ValueError("D1b evaluation payload violates its source cohort")
        victim_ids: set[str] = set()
        for family in D1_SOURCE_FAMILIES:
            victims = tuple(self.source_victims[family])
            if len(victims) != 1:
                raise ValueError("D1b evaluation requires one held-out source victim")
            for victim_id, victim in victims:
                if (
                    not isinstance(victim_id, str)
                    or not victim_id
                    or D1_HELDOUT_FAMILY in victim_id
                    or not isinstance(victim, nn.Module)
                    or victim_id in victim_ids
                ):
                    raise ValueError("D1b evaluation victim allowlist is invalid")
                victim_ids.add(victim_id)


def build_d1b_source_roles(
    context: object,
    cache: object,
) -> ResidualD1BSourceRoles:
    """Bind PPO, threshold, competence, and evaluation to exact D1 roles."""

    required = (
        "source_families",
        "teacher_victims",
        "evaluation_victims",
        "train_indices",
        "threshold_indices",
        "competence_indices",
        "evaluation_indices",
        "ppo_evaluation_indices",
        "train_samples",
        "ppo_evaluation_samples",
        "config",
    )
    if any(not hasattr(context, name) for name in required):
        raise TypeError("D1b source context is incomplete")
    if not isinstance(cache, ResidualTeacherCache) and any(
        not hasattr(cache, name) for name in ("threshold_steps", "competence_steps")
    ):
        raise TypeError("D1b requires a verified teacher cache")
    if tuple(context.source_families) != D1_SOURCE_FAMILIES:
        raise ValueError("D1b context contains a held-out or unknown family")
    validate_d1_role_indices(
        {
            "train": context.train_indices,
            "threshold": context.threshold_indices,
            "competence": context.competence_indices,
            "d1a_evaluation": context.evaluation_indices,
            "d1b_evaluation": context.ppo_evaluation_indices,
        }
    )
    teacher_ids: set[str] = set()
    for family in D1_SOURCE_FAMILIES:
        victims = tuple(context.teacher_victims[family])
        if len(victims) != 2:
            raise ValueError("D1b requires the exact teacher victim allowlist")
        for victim_id, victim in victims:
            if (
                not isinstance(victim_id, str)
                or not victim_id
                or D1_HELDOUT_FAMILY in victim_id
                or not isinstance(victim, nn.Module)
                or victim_id in teacher_ids
            ):
                raise ValueError("D1b teacher victim allowlist is invalid")
            teacher_ids.add(victim_id)
    evaluation = ResidualD1BEvaluationPayload(
        source_victims=context.evaluation_victims,
        source_samples=tuple(context.ppo_evaluation_samples),
        sample_ids=tuple(context.ppo_evaluation_indices),
        attack_config=context.config.attack_config(),
    )
    if teacher_ids & {
        victim_id
        for victims in evaluation.source_victims.values()
        for victim_id, _ in victims
    }:
        raise ValueError("D1b teacher and evaluation victim IDs overlap")
    return ResidualD1BSourceRoles(
        ppo_training=ResidualD1BSourceRole(
            "ppo_training",
            tuple(context.train_indices),
            ResidualD1BTrainingPayload(
                source_victims=context.teacher_victims,
                source_samples=tuple(context.train_samples),
                attack_config=context.config.attack_config(),
            ),
        ),
        threshold_selection=ResidualD1BSourceRole(
            "threshold_selection",
            tuple(context.threshold_indices),
            tuple(cache.threshold_steps),
        ),
        competence_gate=ResidualD1BSourceRole(
            "competence_gate",
            tuple(context.competence_indices),
            tuple(cache.competence_steps),
        ),
        evaluation=ResidualD1BSourceRole(
            "d1b_evaluation",
            tuple(context.ppo_evaluation_indices),
            evaluation,
        ),
    )


def select_d1b_threshold(
    policy: object,
    steps: object,
    deadline_check: Callable[[], None],
) -> dict[str, object]:
    if not isinstance(policy, ResidualRankerPolicy):
        raise TypeError("D1b threshold selection requires a residual policy")
    if not callable(deadline_check):
        raise TypeError("D1b threshold deadline check must be callable")
    selection = select_confidence_threshold(
        policy.backbone,
        tuple(steps),  # type: ignore[arg-type]
        seed=D1_SEED + 50_000,
        prior_temperature=D1_PRIOR_TEMPERATURE,
        required_source_families=D1_SOURCE_FAMILIES,
        deadline_check=deadline_check,
    )
    return {
        **selection,
        "selection_role": "d1b_threshold_selection_only",
        "target_calls": 0,
        "hidden_target_calls": 0,
    }


def apply_d1b_threshold(
    policy: object,
    raw_threshold: object,
) -> ResidualRankerPolicy:
    if not isinstance(policy, ResidualRankerPolicy):
        raise TypeError("D1b threshold application requires a residual policy")
    threshold = _mapping(raw_threshold, "D1b threshold selection")
    _source_only(threshold, "D1b threshold selection")
    if threshold.get("selection_role") != "d1b_threshold_selection_only":
        raise ValueError("D1b threshold role is invalid")
    clone = clone_residual_policy(policy)
    return ResidualRankerPolicy(
        clone.backbone,
        confidence_threshold=_number(
            threshold.get("threshold"),
            "D1b confidence threshold",
        ),
        prior_temperature=policy.prior_temperature,
        overrides_enabled=threshold.get("overrides_enabled"),  # type: ignore[arg-type]
    )


def evaluate_d1b_competence(
    policy: object,
    steps: object,
    deadline_check: Callable[[], None],
) -> dict[str, object]:
    if not isinstance(policy, ResidualRankerPolicy):
        raise TypeError("D1b competence requires a residual policy")
    if not callable(deadline_check):
        raise TypeError("D1b competence deadline check must be callable")
    result = evaluate_residual_ranker_examples(
        policy,
        tuple(steps),  # type: ignore[arg-type]
        prior_seed=D1_SEED + 50_000,
        required_source_families=D1_SOURCE_FAMILIES,
        deadline_check=deadline_check,
    )
    for name in ("by_source_family", "equal_family_macro", "worst_family"):
        if not isinstance(result.get(name), Mapping):
            raise ValueError("D1b competence lacks family-balanced diagnostics")
    if set(result["by_source_family"]) != set(D1_SOURCE_FAMILIES):
        raise ValueError("D1b competence lacks a locked source family")
    return {
        **result,
        "target_calls": 0,
        "hidden_target_calls": 0,
    }


__all__ = (
    "ResidualD1AArtifactBundle",
    "ResidualD1BEvaluationPayload",
    "apply_d1b_threshold",
    "build_d1b_source_roles",
    "evaluate_d1b_competence",
    "select_d1b_threshold",
    "verify_d1a_artifacts",
)
