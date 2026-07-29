"""Pure, source-only orchestration for conditional D1b residual PPO."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .phase2_residual_d1 import (
    D1_BC_EPISODES,
    D1_HELDOUT_FAMILY,
    D1_SEED,
    D1_SOURCE_FAMILIES,
    validate_source_only_payload as _source_only,
)
from .phase2_residual_d1b_contracts import (
    D1B_BLOCK_ENDPOINTS,
    D1B_BLOCK_EPISODES,
    D1B_METHODS,
    D1B_TOTAL_EPISODES,
    ResidualD1BBlockState,
    ResidualD1BCheckpointReceipt,
    ResidualD1BDependencies,
    ResidualD1BEvaluationInputs,
    ResidualD1BLoadedCheckpoint,
    ResidualD1BResult,
    ResidualD1BResumeState,
    ResidualD1BSourceRole,
    ResidualD1BSourceRoles,
    ResidualD1BTrainingPayload,
    VerifiedD1AArtifacts,
    _block_output,
    _checkpoint_id,
    _digest,
    _false,
    _freeze,
    _integer,
    _number,
    _object,
    _seal,
    _sha,
    _thaw,
    _zero,
)
from .residual_ppo import train_residual_ranker_ppo


_D1A_NAME = "phase2-d1a-residual-ranker-bc"
_D1B_NAME = "phase2-d1b-residual-ranker-ppo"
_ROLE_SIZES = {
    "train": 200,
    "threshold": 50,
    "competence": 50,
    "d1a_evaluation": 50,
    "d1b_evaluation": 50,
}
_ROLE_BINDINGS = {
    "ppo_training": "train",
    "threshold_selection": "threshold",
    "competence_gate": "competence",
    "d1b_evaluation": "d1b_evaluation",
}


def _manifest(value: object) -> tuple[dict[str, object], bool, str | None]:
    manifest = _object(value, "D1a manifest")
    _source_only(manifest, "D1a manifest")
    if (
        manifest.get("schema_version") != 3
        or manifest.get("name") != _D1A_NAME
        or manifest.get("heldout_family") != D1_HELDOUT_FAMILY
        or tuple(manifest.get("source_families", ())) != D1_SOURCE_FAMILIES
        or manifest.get("seed") != D1_SEED
        or manifest.get("bc_episodes") != D1_BC_EPISODES
        or manifest.get("ppo_episodes_completed") != 0
    ):
        raise ValueError("D1a manifest violates the locked protocol")
    _zero(manifest.get("target_calls"), "D1a target calls")
    for key in (
        "target_evaluation_performed",
        "target_evaluation_available",
        "authorizes_hidden_target_evaluation",
    ):
        _false(manifest.get(key), f"D1a {key}")
    decision = _object(manifest.get("d1_decision"), "D1a decision")
    passed = decision.get("passed")
    eligible = decision.get("eligible_for_d1b_source_only_ppo")
    if (
        not isinstance(passed, bool)
        or not isinstance(eligible, bool)
        or passed != eligible
    ):
        raise ValueError("D1a gate and PPO eligibility are inconsistent")
    _false(
        decision.get("authorizes_hidden_target_evaluation"),
        "D1a decision authorization",
    )
    status = manifest.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("D1a status is invalid")
    if passed and status != "complete":
        raise ValueError("D1a cannot pass before completion")
    if not passed:
        reason = (
            "d1a_source_gate_failed"
            if status == "complete"
            else f"d1a_not_complete:{status}"
        )
        return manifest, False, reason
    if (
        manifest.get("training_performed") is not True
        or manifest.get("teacher_completed") is not True
    ):
        raise ValueError("passing D1a requires BC and teacher completion")
    checkpoint = _object(manifest.get("checkpoint"), "D1a checkpoint")
    _digest(checkpoint.get("sha256"), "D1a checkpoint digest")
    _digest(checkpoint.get("persistent_digest"), "D1a policy digest")
    return manifest, True, None


def _role_audit(
    manifest: Mapping[str, object],
    roles: ResidualD1BSourceRoles,
) -> dict[str, object]:
    audit = _object(manifest.get("source_split_roles"), "D1a role audit")
    sizes = _object(audit.get("role_sizes"), "D1a role sizes")
    digests = _object(audit.get("role_indices_sha256"), "D1a role digests")
    if (
        audit.get("pairwise_disjoint") is not True
        or set(sizes) != set(_ROLE_SIZES)
        or set(digests) != set(_ROLE_SIZES)
        or any(sizes[name] != size for name, size in _ROLE_SIZES.items())
    ):
        raise ValueError("D1a source role audit violates D1b protocol")
    for name in _ROLE_SIZES:
        _digest(digests[name], f"D1a {name} role digest")
    for role in roles.as_tuple:
        if digests[_ROLE_BINDINGS[role.name]] != role.sample_ids_sha256:
            raise ValueError(f"D1b {role.name} cohort digest binding mismatch")
    return audit


def _checkpoint_metadata(
    block_index: int,
    metrics: Mapping[str, object],
    weights: Mapping[str, object],
    offsets: Mapping[str, object],
    policy_digest: str,
    state: ResidualD1BResumeState,
) -> dict[str, object]:
    parent = state.blocks[-1].checkpoint if state.blocks else None
    return {
        "schema_version": 1,
        "name": "phase2-d1b-residual-ranker-ppo-block",
        "block_index": block_index,
        "episode_offset": (block_index - 1) * D1B_BLOCK_EPISODES,
        "episodes": D1B_BLOCK_EPISODES,
        "episodes_completed": block_index * D1B_BLOCK_EPISODES,
        "d1a_manifest_digest": state.d1a_manifest_digest,
        "d1a_checkpoint_sha256": state.d1a_checkpoint_sha256,
        "bc_policy_digest": state.bc_policy_digest,
        "source_roles_digest": state.source_roles_digest,
        "ppo_policy_digest": policy_digest,
        "ppo_metrics_digest": _sha(metrics, "D1b block metrics"),
        "parent_checkpoint": (
            None
            if parent is None
            else {
                "reference": parent.reference,
                "policy_digest": parent.policy_digest,
                "metadata_digest": parent.metadata_digest,
            }
        ),
        "family_weights": dict(weights),
        "instance_offsets": dict(offsets),
        **_seal(),
    }


def _validate_resume(
    state: object,
    expected: ResidualD1BResumeState,
) -> ResidualD1BResumeState:
    if not isinstance(state, ResidualD1BResumeState):
        raise TypeError("resume state must use ResidualD1BResumeState")
    if (
        state.d1a_manifest_digest != expected.d1a_manifest_digest
        or state.d1a_checkpoint_sha256 != expected.d1a_checkpoint_sha256
        or state.bc_policy_digest != expected.bc_policy_digest
        or state.source_roles_digest != expected.source_roles_digest
    ):
        raise ValueError("D1b resume identity or role binding mismatch")
    for label, value in (
        ("manifest", state.d1a_manifest_digest),
        ("checkpoint", state.d1a_checkpoint_sha256),
        ("BC policy", state.bc_policy_digest),
        ("source roles", state.source_roles_digest),
    ):
        _digest(value, f"D1b resume {label} digest")
    if not isinstance(state.blocks, tuple) or len(state.blocks) > 4:
        raise ValueError("D1b resume has too many blocks")
    for index, block in enumerate(state.blocks, start=1):
        if (
            not isinstance(block, ResidualD1BBlockState)
            or block.block_index != index
            or block.episode_offset != (index - 1) * D1B_BLOCK_EPISODES
            or not isinstance(block.checkpoint, ResidualD1BCheckpointReceipt)
        ):
            raise ValueError("D1b resume block chain is not contiguous")
        metrics, weights, offsets = _block_output(
            block.metrics,
            block.episode_offset,
        )
        _digest(block.policy_digest, "D1b resume PPO policy digest")
        if (
            weights != _object(block.family_weights, "D1b resume weights")
            or offsets != _object(block.instance_offsets, "D1b resume offsets")
            or block.checkpoint.policy_digest != block.policy_digest
            or block.checkpoint.reference
            != _checkpoint_id(
                block.policy_digest,
                block.checkpoint.metadata_digest,
            )
        ):
            raise ValueError("D1b resume PPO state binding mismatch")
        parent_state = ResidualD1BResumeState(
            state.d1a_manifest_digest,
            state.d1a_checkpoint_sha256,
            state.bc_policy_digest,
            state.source_roles_digest,
            state.blocks[: index - 1],
        )
        metadata = _checkpoint_metadata(
            block.block_index,
            metrics,
            weights,
            offsets,
            block.policy_digest,
            parent_state,
        )
        if _thaw(
            block.checkpoint_metadata
        ) != metadata or block.checkpoint.metadata_digest != _sha(
            metadata, "D1b resume metadata"
        ):
            raise ValueError("D1b resume checkpoint metadata binding mismatch")
    return state


def _verified(
    manifest: Mapping[str, object],
    artifacts: object,
    dependencies: ResidualD1BDependencies,
) -> VerifiedD1AArtifacts:
    verified = dependencies.verify_d1a(manifest, artifacts)
    if not isinstance(verified, VerifiedD1AArtifacts):
        raise TypeError("D1a verifier must return VerifiedD1AArtifacts")
    checkpoint = _object(manifest["checkpoint"], "D1a checkpoint")
    if (
        verified.manifest_digest != _sha(manifest, "D1a manifest")
        or verified.checkpoint_sha256 != checkpoint["sha256"]
        or verified.bc_policy_digest != checkpoint["persistent_digest"]
    ):
        raise ValueError("verified D1a artifact binding mismatch")
    actual = dependencies.policy_digest(verified.bc_policy)
    _digest(actual, "loaded D1a policy digest")
    if actual != verified.bc_policy_digest:
        raise ValueError("loaded D1a policy digest binding mismatch")
    return verified


def _bc_unchanged(
    dependencies: ResidualD1BDependencies,
    policy: object,
    digest: str,
) -> None:
    actual = dependencies.policy_digest(policy)
    _digest(actual, "frozen D1a BC policy digest")
    if actual != digest:
        raise ValueError("frozen D1a BC checkpoint was mutated")


def _load_resume(
    state: ResidualD1BResumeState,
    dependencies: ResidualD1BDependencies,
    deadline: Callable[[], None],
) -> object:
    loaded: ResidualD1BLoadedCheckpoint | None = None
    for block in state.blocks:
        deadline()
        candidate = dependencies.load_block_checkpoint(block.checkpoint)
        if not isinstance(candidate, ResidualD1BLoadedCheckpoint):
            raise TypeError("loader must return ResidualD1BLoadedCheckpoint")
        if _thaw(candidate.metadata) != _thaw(block.checkpoint_metadata) or (
            _sha(candidate.metadata, "loaded checkpoint metadata")
            != block.checkpoint.metadata_digest
        ):
            raise ValueError("loaded resume checkpoint metadata binding mismatch")
        digest = dependencies.policy_digest(candidate.policy)
        _digest(digest, "loaded D1b policy digest")
        if digest != block.policy_digest:
            raise ValueError("loaded resume checkpoint policy digest mismatch")
        loaded = candidate
    if loaded is None:
        raise ValueError("D1b resume state did not contain a checkpoint")
    return loaded.policy


def _threshold(value: object) -> dict[str, object]:
    result = _object(value, "D1b threshold selection")
    _source_only(result, "D1b threshold selection")
    _zero(result.get("target_calls"), "D1b threshold target calls")
    if result.get("selection_role") != "d1b_threshold_selection_only":
        raise ValueError("D1b threshold used the wrong source role")
    _number(result.get("threshold"), "D1b threshold")
    _integer(result.get("accepted_steps"), "D1b threshold accepted steps")
    if not isinstance(result.get("overrides_enabled"), bool):
        raise ValueError("D1b threshold override flag must be boolean")
    return result


def _competence(value: object) -> dict[str, object]:
    result = _object(value, "D1b competence")
    _source_only(result, "D1b competence")
    _zero(result.get("target_calls"), "D1b competence target calls")
    if result.get("target_mode") != "all_soft":
        raise ValueError("D1b competence requires all-soft targets")
    _integer(result.get("accepted_steps"), "D1b competence accepted steps")
    for key in (
        "gated_top1_accuracy",
        "prior_top1_accuracy",
        "residual_use_fraction",
    ):
        if _number(result.get(key), f"D1b competence {key}") > 1:
            raise ValueError(f"D1b competence {key} must be in [0, 1]")
    for key in ("soft_cross_entropy", "prior_soft_cross_entropy"):
        _number(result.get(key), f"D1b competence {key}")
    return result


def _skipped(
    manifest: Mapping[str, object],
    reason: str,
) -> ResidualD1BResult:
    return ResidualD1BResult(
        _freeze(
            {
                "schema_version": 3,
                "name": _D1B_NAME,
                "status": "skipped",
                "d1a_status": manifest["status"],
                "d1a_source_gate_passed": False,
                "ppo_episodes_completed": 0,
                "ppo_blocks_completed": 0,
                "ppo_skipped_reason": reason,
                **_seal(),
            }
        ),
        None,
        None,
    )


def run_residual_d1b(
    d1a_manifest: Mapping[str, object],
    d1a_artifacts: object,
    source_roles: ResidualD1BSourceRoles,
    dependencies: ResidualD1BDependencies,
    *,
    resume_state: ResidualD1BResumeState | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> ResidualD1BResult:
    """Run four committed PPO blocks after, and only after, a passing D1a."""

    manifest, passed, reason = _manifest(d1a_manifest)
    if not passed:
        if reason is None:
            raise ValueError("D1a rejection reason is missing")
        return _skipped(manifest, reason)
    if not isinstance(source_roles, ResidualD1BSourceRoles):
        raise TypeError("D1b source roles use the locked schema")
    if not isinstance(dependencies, ResidualD1BDependencies):
        raise TypeError("D1b dependencies use the locked schema")
    if deadline_check is not None and not callable(deadline_check):
        raise TypeError("deadline_check must be callable")
    deadline = deadline_check or (lambda: None)
    audit = _role_audit(manifest, source_roles)
    deadline()
    verified = _verified(manifest, d1a_artifacts, dependencies)
    deadline()
    bc_policy = verified.bc_policy
    bc_digest = verified.bc_policy_digest
    expected = ResidualD1BResumeState(
        verified.manifest_digest,
        verified.checkpoint_sha256,
        bc_digest,
        source_roles.digest,
    )
    state = (
        expected if resume_state is None else _validate_resume(resume_state, expected)
    )
    _bc_unchanged(dependencies, bc_policy, bc_digest)
    if state.blocks:
        ppo_policy = _load_resume(state, dependencies, deadline)
    else:
        ppo_policy = dependencies.clone_policy(bc_policy)
        if ppo_policy is bc_policy:
            raise ValueError("D1b clone must be distinct from frozen BC")
        clone_digest = dependencies.policy_digest(ppo_policy)
        _digest(clone_digest, "D1b clone digest")
        if clone_digest != bc_digest:
            raise ValueError("D1b clone digest differs from frozen BC")
    _bc_unchanged(dependencies, bc_policy, bc_digest)
    blocks = state.blocks
    for index in range(len(blocks) + 1, 5):
        offset = (index - 1) * D1B_BLOCK_EPISODES
        previous = blocks[-1] if blocks else None
        deadline()
        output = dependencies.train_ppo_block(
            ppo_policy,
            source_roles.ppo_training.payload,
            episodes=D1B_BLOCK_EPISODES,
            seed=D1_SEED + 80_000,
            prior_seed=D1_SEED + 50_000,
            initial_family_weights=(
                None if previous is None else dict(previous.family_weights)
            ),
            episode_offset=offset,
            initial_instance_offsets=(
                None if previous is None else dict(previous.instance_offsets)
            ),
            deadline_check=deadline,
        )
        metrics, weights, offsets = _block_output(output, offset)
        deadline()
        _bc_unchanged(dependencies, bc_policy, bc_digest)
        policy_digest = dependencies.policy_digest(ppo_policy)
        _digest(policy_digest, "D1b PPO policy digest")
        metadata = _checkpoint_metadata(
            index,
            metrics,
            weights,
            offsets,
            policy_digest,
            state,
        )
        receipt = dependencies.save_block_checkpoint(ppo_policy, metadata)
        if not isinstance(receipt, ResidualD1BCheckpointReceipt):
            raise TypeError("saver must return ResidualD1BCheckpointReceipt")
        if (
            receipt.policy_digest != policy_digest
            or receipt.metadata_digest != _sha(metadata, "D1b metadata")
            or receipt.reference
            != _checkpoint_id(
                receipt.policy_digest,
                receipt.metadata_digest,
            )
        ):
            raise ValueError("D1b checkpoint metadata digest mismatch")
        block = ResidualD1BBlockState(
            index,
            offset,
            _freeze(metrics),
            _freeze(weights),
            _freeze(offsets),
            policy_digest,
            receipt,
            _freeze(metadata),
        )
        blocks = (*blocks, block)
        state = ResidualD1BResumeState(
            state.d1a_manifest_digest,
            state.d1a_checkpoint_sha256,
            state.bc_policy_digest,
            state.source_roles_digest,
            blocks,
        )
    uncalibrated_digest = dependencies.policy_digest(ppo_policy)
    _digest(uncalibrated_digest, "uncalibrated D1b PPO digest")
    deadline()
    threshold = _threshold(
        dependencies.select_threshold(
            ppo_policy,
            source_roles.threshold_selection.payload,
            deadline,
        )
    )
    calibrated = dependencies.apply_threshold(ppo_policy, threshold)
    if calibrated is ppo_policy or calibrated is bc_policy:
        raise ValueError("D1b calibration must create a fresh policy")
    if dependencies.policy_digest(ppo_policy) != uncalibrated_digest:
        raise ValueError("D1b calibration mutated the PPO checkpoint")
    _bc_unchanged(dependencies, bc_policy, bc_digest)
    calibrated_digest = dependencies.policy_digest(calibrated)
    _digest(calibrated_digest, "calibrated D1b PPO digest")
    deadline()
    competence = _competence(
        dependencies.evaluate_competence(
            calibrated,
            source_roles.competence_gate.payload,
            deadline,
        )
    )
    if dependencies.policy_digest(calibrated) != calibrated_digest:
        raise ValueError("D1b competence mutated the PPO policy")
    _bc_unchanged(dependencies, bc_policy, bc_digest)
    evaluation = ResidualD1BEvaluationInputs(
        source_roles.evaluation.payload,
        source_roles.evaluation.sample_ids,
        D1_SOURCE_FAMILIES,
        D1B_METHODS,
        D1_SEED,
        D1_SEED + 50_000,
        D1B_BLOCK_EPISODES,
        bc_policy,
        calibrated,
        bc_digest,
        calibrated_digest,
        _freeze(threshold),
        _freeze(competence),
    )
    final = {
        "schema_version": 3,
        "name": _D1B_NAME,
        "status": "complete",
        "diagnostic_only": True,
        "research_valid": False,
        "publication_candidate": False,
        "heldout_family": D1_HELDOUT_FAMILY,
        "source_families": list(D1_SOURCE_FAMILIES),
        "seed": D1_SEED,
        "d1a_source_gate_passed": True,
        "d1a_manifest_digest": state.d1a_manifest_digest,
        "d1a_checkpoint_sha256": state.d1a_checkpoint_sha256,
        "source_split_roles": audit,
        "source_roles_digest": state.source_roles_digest,
        "bc_policy_digest_before": bc_digest,
        "bc_policy_digest_after": dependencies.policy_digest(bc_policy),
        "bc_checkpoint_preserved": True,
        "ppo_policy_digest": calibrated_digest,
        "ppo_block_episodes": D1B_BLOCK_EPISODES,
        "ppo_block_endpoints": list(D1B_BLOCK_ENDPOINTS),
        "ppo_episodes_completed": state.completed_episodes,
        "ppo_blocks_completed": len(state.blocks),
        "ppo_skipped_reason": None,
        "source_model_calls": sum(
            int(block.metrics["source_calls"]) for block in state.blocks
        ),
        "threshold_selection": threshold,
        "competence_gate": competence,
        "evaluation_role": "d1b_evaluation",
        "evaluation_sample_ids_sha256": (source_roles.evaluation.sample_ids_sha256),
        "evaluation_methods": list(D1B_METHODS),
        **_seal(),
    }
    _bc_unchanged(dependencies, bc_policy, bc_digest)
    return ResidualD1BResult(_freeze(final), state, evaluation)


def existing_residual_ppo_block(
    policy: object,
    payload: ResidualD1BTrainingPayload,
    **kwargs: Any,
) -> dict[str, object]:
    """Adapt the existing trainer to the injected D1b block interface."""

    if not isinstance(payload, ResidualD1BTrainingPayload):
        raise TypeError("PPO adapter requires ResidualD1BTrainingPayload")
    return train_residual_ranker_ppo(
        policy,
        payload.source_victims,
        payload.source_samples,
        payload.attack_config,
        **kwargs,
    )


__all__ = (
    "D1B_BLOCK_ENDPOINTS",
    "D1B_BLOCK_EPISODES",
    "D1B_METHODS",
    "D1B_TOTAL_EPISODES",
    "ResidualD1BBlockState",
    "ResidualD1BCheckpointReceipt",
    "ResidualD1BDependencies",
    "ResidualD1BEvaluationInputs",
    "ResidualD1BLoadedCheckpoint",
    "ResidualD1BResult",
    "ResidualD1BResumeState",
    "ResidualD1BSourceRole",
    "ResidualD1BSourceRoles",
    "ResidualD1BTrainingPayload",
    "VerifiedD1AArtifacts",
    "existing_residual_ppo_block",
    "run_residual_d1b",
)
