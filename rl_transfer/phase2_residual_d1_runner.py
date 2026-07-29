"""Bounded source-only orchestration for the Phase 2 D1a residual ranker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset

from .artifacts import (
    exclusive_file_lock,
    load_recurrent_checkpoint,
    save_recurrent_checkpoint,
    sha256_file,
)
from .cifar_manifest import code_digest, git_revision, git_worktree_state
from .phase2_residual_d1 import (
    D1_SOURCE_FAMILIES,
    ResidualD1Request,
    validate_residual_source_records,
)
from .phase2_residual_d1_evaluation import _decision, evaluate_residual_d1
from .phase2_residual_d1_evidence import (
    load_verified_jsonl_records,
    verify_d1_raw_evidence,
    verify_d1_recorded_summaries,
    write_d1_evidence_plots,
)
from .phase2_residual_d1_reporting import paired_source_statistics
from .phase2_residual_d1_source import (
    _mapping,
    _validated_runtime_environment,
    load_d1_source_context,
    load_residual_d1_source,
)
from .phase2_residual_d1_teacher import (
    D1_BC_EPOCHS,
    D1_HIDDEN_DIM,
    D1_PRIOR_TEMPERATURE,
    _teacher_examples,
    _write_verified_jsonl,
)
from .recurrent import PPOConfig, RecurrentAttackPolicy
from .residual_bc import fit_residual_ranker_bc
from .residual_ranker import (
    ResidualRankerPolicy,
    evaluate_residual_ranker_examples,
    select_confidence_threshold,
)
from .verified_artifacts import load_verified_json, write_verified_json


class ResidualD1Deadline(TimeoutError):
    """Raised between bounded D1a work units."""


_D1_RESUMABLE_FILES = frozenset(
    {
        ".d1.lock",
        ".teacher-cache.lock",
        "d1_manifest.json",
        "d1_manifest.json.sha256",
        "asr_by_query.svg",
        "asr_by_query.svg.sha256",
        "final_asr.svg",
        "final_asr.svg.sha256",
        "residual_ranker_bc.pt",
        "residual_ranker_bc.pt.sha256",
        "source_query_traces.jsonl",
        "source_query_traces.jsonl.sha256",
        "source_results.jsonl",
        "source_results.jsonl.sha256",
        "teacher_ranker_examples.jsonl",
        "teacher_ranker_examples.jsonl.sha256",
        "teacher_ranker_manifest.json",
        "teacher_ranker_manifest.json.sha256",
    }
)
_D1_NONRESUMABLE_PARTIAL_FILES = frozenset(
    {
        "asr_by_query.svg",
        "asr_by_query.svg.sha256",
        "final_asr.svg",
        "final_asr.svg.sha256",
        "residual_ranker_bc.pt",
        "residual_ranker_bc.pt.sha256",
        "source_query_traces.jsonl",
        "source_query_traces.jsonl.sha256",
        "source_results.jsonl",
        "source_results.jsonl.sha256",
    }
)


def _existing_d1_manifest(
    request: ResidualD1Request,
) -> Mapping[str, object] | None:
    names = {path.name for path in request.output_dir.iterdir()}
    unexpected = names - _D1_RESUMABLE_FILES
    if unexpected:
        raise ValueError(
            f"D1a output contains unexpected artifacts: {sorted(unexpected)}"
        )
    if any(path.is_symlink() for path in request.output_dir.iterdir()):
        raise ValueError("D1a output artifacts cannot be symlinks")
    manifest_path = request.output_dir / "d1_manifest.json"
    checksum_path = request.output_dir / "d1_manifest.json.sha256"
    if manifest_path.exists() != checksum_path.exists():
        raise ValueError("D1a manifest artifact pair is incomplete")
    if not manifest_path.exists():
        return None
    manifest = load_verified_json(manifest_path)
    if manifest.get("request_sha256") != request.digest():
        raise ValueError("D1a resume manifest request binding mismatch")
    if manifest.get("status") == "complete":
        _verify_complete_d1_children(request, manifest)
    elif names & _D1_NONRESUMABLE_PARTIAL_FILES:
        raise ValueError(
            "D1a partial promoted artifacts cannot be overwritten or resumed"
        )
    return manifest


def _verified_child(path: object, expected_digest: object) -> None:
    if (
        not isinstance(path, Path)
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != expected_digest
    ):
        raise ValueError("D1a complete child artifact failed checksum verification")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        sidecar.is_symlink()
        or not sidecar.is_file()
        or sidecar.read_text().strip() != expected_digest
    ):
        raise ValueError("D1a complete child checksum sidecar failed verification")


def _verify_complete_d1_children(
    request: ResidualD1Request,
    manifest: Mapping[str, object],
) -> None:
    if (
        manifest.get("target_calls") != 0
        or manifest.get("hidden_target_calls") != 0
        or manifest.get("target_evaluation_performed") is not False
        or manifest.get("hidden_target_evaluation_performed") is not False
        or manifest.get("authorizes_hidden_target_evaluation") is not False
    ):
        raise ValueError("D1a complete manifest violates the source-only seal")
    if sha256_file(request.source_manifest) != manifest.get("source_manifest_sha256"):
        raise ValueError("D1a sealed source manifest identity changed")
    checkpoint = _mapping(
        manifest.get("checkpoint"),
        label="D1a complete checkpoint",
    )
    if checkpoint.get("name") != "residual_ranker_bc.pt":
        raise ValueError("D1a complete checkpoint name is invalid")
    teacher = _mapping(
        manifest.get("teacher_cache"),
        label="D1a complete teacher cache",
    )
    figures = _mapping(
        manifest.get("figures"),
        label="D1a complete figures",
    )
    if set(figures) != {"asr_by_query.svg", "final_asr.svg"}:
        raise ValueError("D1a complete figure set is invalid")
    children = (
        (
            request.output_dir / "residual_ranker_bc.pt",
            checkpoint.get("sha256"),
        ),
        (
            request.output_dir / "source_results.jsonl",
            manifest.get("results_sha256"),
        ),
        (
            request.output_dir / "source_query_traces.jsonl",
            manifest.get("query_traces_sha256"),
        ),
        (
            request.output_dir / "teacher_ranker_examples.jsonl",
            teacher.get("examples_sha256"),
        ),
        (
            request.output_dir / "teacher_ranker_manifest.json",
            teacher.get("metadata_sha256"),
        ),
        *(
            (
                request.output_dir / name,
                digest,
            )
            for name, digest in figures.items()
        ),
    )
    for path, digest in children:
        _verified_child(path, digest)
    rows = load_verified_jsonl_records(request.output_dir / "source_results.jsonl")
    traces = load_verified_jsonl_records(
        request.output_dir / "source_query_traces.jsonl"
    )
    recomputed = verify_d1_raw_evidence(
        rows,
        traces,
        expected_methods=("score_greedy", "residual_ranker_bc"),
    )
    verify_d1_recorded_summaries(
        recomputed,
        _mapping(
            manifest.get("source_evaluation"),
            label="D1a source evaluation",
        ),
    )
    if recomputed != manifest.get("raw_evidence_verification"):
        raise ValueError("D1a raw evidence recomputation changed")


def build_residual_d1_dry_run(
    request: ResidualD1Request,
    source: Mapping[str, object],
) -> dict[str, object]:
    if request.ppo_episodes != 0:
        raise ValueError("D1a supports BC only; PPO is a conditional D1b stage")
    folds = tuple(
        _mapping(fold, label="D1 dry-run fold") for fold in source.get("folds", ())
    )
    if len(folds) != 1:
        raise ValueError("D1 dry run requires one selectively verified fold")
    return {
        "schema_version": 3,
        "name": "phase2-d1a-residual-ranker-bc",
        "diagnostic_only": True,
        "research_valid": False,
        "publication_candidate": False,
        "heldout_family": request.heldout_family,
        "source_families": list(folds[0]["source_families"]),
        "training_slice": "exact_source",
        "evaluation_slice": "seen_family_new_instance",
        "evaluation_role": "reused_source_development_gate_first_half",
        "seed": request.seed,
        "source_images": request.source_images,
        "bc_episodes": request.bc_episodes,
        "ppo_planned": False,
        "ppo_episodes": 0,
        "deadline_seconds": request.deadline_seconds,
        "query_budget": 50,
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _base_manifest(
    request: ResidualD1Request,
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "name": "phase2-d1a-residual-ranker-bc",
        "status": "running",
        "diagnostic_only": True,
        "research_valid": False,
        "publication_candidate": False,
        "request_sha256": request.digest(),
        "code_digest": code_digest(),
        "git_revision": git_revision(),
        "git_worktree_state": git_worktree_state(),
        "training_performed": False,
        "heldout_family": request.heldout_family,
        "source_families": list(D1_SOURCE_FAMILIES),
        "training_slice": "exact_source",
        "evaluation_slice": "seen_family_new_instance",
        "evaluation_role": "reused_source_development_gate_first_half",
        "seed": request.seed,
        "bc_episodes": request.bc_episodes,
        "ppo_planned": False,
        "ppo_episodes_completed": 0,
        "ppo_reason": "D1a isolates residual-ranking BC before PPO compute",
        "source_images": request.source_images,
        "deadline_seconds": request.deadline_seconds,
        "dataset_version": dataset_version,
        "dataset_content_sha256": dataset_content_sha256,
        "runtime_environment": _validated_runtime_environment(runtime_environment),
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _gpu_memory_record() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _run_residual_d1_from_datasets(
    request: ResidualD1Request,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
    progress: Callable[[str], None],
    clock: Callable[[], float],
    external_deadline_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    if request.ppo_episodes != 0:
        raise ValueError("D1a execution supports BC only")
    started = clock()
    absolute_deadline = started + request.deadline_seconds
    training_performed = False
    teacher_completed = False

    def deadline_check() -> None:
        if external_deadline_check is not None:
            external_deadline_check()
        if clock() >= absolute_deadline:
            raise ResidualD1Deadline("D1a bounded-work deadline reached")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    base = _base_manifest(
        request,
        dataset_version=dataset_version,
        dataset_content_sha256=dataset_content_sha256,
        runtime_environment=runtime_environment,
    )
    manifest_path = request.output_dir / "d1_manifest.json"
    write_verified_json(manifest_path, base)
    try:
        deadline_check()
        component_started = clock()
        context = load_d1_source_context(
            request,
            train_dataset,
            test_dataset,
            dataset_content_sha256=dataset_content_sha256,
        )
        source_load_seconds = clock() - component_started
        component_started = clock()
        train_steps, threshold_steps, competence_steps, teacher_manifest = (
            _teacher_examples(
                request,
                context,
                deadline_check=deadline_check,
                progress=progress,
            )
        )
        teacher_seconds = clock() - component_started
        teacher_completed = True
        deadline_check()
        component_started = clock()
        attack = context.config.attack_config()
        backbone = RecurrentAttackPolicy(
            attack.recurrent_observation_dim,
            attack.action_dim,
            hidden_dim=D1_HIDDEN_DIM,
            seed=request.seed + 40_000,
            config=PPOConfig(
                learning_rate=context.config.policy_learning_rate,
                entropy_weight=context.config.policy_entropy_weight,
                update_epochs=4,
            ),
            actor_mode="action_conditioned",
            action_grid_size=attack.grid_size,
        ).to(torch.device(request.device))
        prior_seed = request.seed + 50_000
        progress(f"[d1a] fitting aligned residual BC for {D1_BC_EPOCHS} epochs")
        training_performed = True
        training = fit_residual_ranker_bc(
            backbone,
            train_steps,
            epochs=D1_BC_EPOCHS,
            seed=request.seed + 60_000,
            prior_seed=prior_seed,
            prior_temperature=D1_PRIOR_TEMPERATURE,
            deadline_check=deadline_check,
            required_source_families=D1_SOURCE_FAMILIES,
        )
        threshold = select_confidence_threshold(
            backbone,
            threshold_steps,
            seed=prior_seed,
            prior_temperature=D1_PRIOR_TEMPERATURE,
            required_source_families=D1_SOURCE_FAMILIES,
            deadline_check=deadline_check,
        )
        policy = ResidualRankerPolicy(
            backbone,
            confidence_threshold=float(threshold["threshold"]),
            prior_temperature=D1_PRIOR_TEMPERATURE,
            overrides_enabled=bool(threshold["overrides_enabled"]),
        )
        competence = evaluate_residual_ranker_examples(
            policy,
            competence_steps,
            prior_seed=prior_seed,
            required_source_families=D1_SOURCE_FAMILIES,
            deadline_check=deadline_check,
        )
        bc_and_calibration_seconds = clock() - component_started
        checkpoint_path = request.output_dir / "residual_ranker_bc.pt"
        checkpoint_metadata = {
            "schema_version": 3,
            "kind": "phase2_d1a_source_only_residual_ranker_bc",
            "seed": request.seed,
            "heldout_family": request.heldout_family,
            "source_manifest_sha256": context.source_manifest_sha256,
            "request_sha256": request.digest(),
            "confidence_threshold": threshold["threshold"],
            "prior_temperature": D1_PRIOR_TEMPERATURE,
            "overrides_enabled": threshold["overrides_enabled"],
            "target_calls": 0,
            "hidden_target_calls": 0,
            "target_evaluation_performed": False,
            "hidden_target_evaluation_performed": False,
            "target_evaluation_available": False,
            "authorizes_hidden_target_evaluation": False,
        }
        checkpoint_sha256 = save_recurrent_checkpoint(
            checkpoint_path,
            backbone,
            checkpoint_metadata,
        )
        reloaded_backbone, reloaded_metadata = load_recurrent_checkpoint(
            checkpoint_path,
            request.device,
            expected_observation_dim=attack.recurrent_observation_dim,
            expected_action_dim=attack.action_dim,
            expected_hidden_dim=D1_HIDDEN_DIM,
            expected_actor_mode="action_conditioned",
        )
        if (
            reloaded_metadata != checkpoint_metadata
            or reloaded_backbone.persistent_digest() != backbone.persistent_digest()
        ):
            raise ValueError("D1a checkpoint round-trip verification failed")
        deadline_check()
        component_started = clock()
        conditions, rows, traces = evaluate_residual_d1(
            request,
            context,
            policy,
            deadline_check=deadline_check,
            progress=progress,
        )
        evaluation_seconds = clock() - component_started
        enriched_rows = [
            {
                **asdict(row),
                "action_trace": list(row.action_trace),
                "heldout_family": request.heldout_family,
                "source_slice": "seen_family_new_instance",
                "target_calls": 0,
                "hidden_target_calls": 0,
            }
            for row in rows
        ]
        validate_residual_source_records(
            enriched_rows,
            heldout_family=request.heldout_family,
        )
        results_sha256 = _write_verified_jsonl(
            request.output_dir / "source_results.jsonl",
            enriched_rows,
        )
        enriched_traces = [
            {
                **trace,
                "heldout_family": request.heldout_family,
                "source_slice": "seen_family_new_instance",
                "target_calls": 0,
                "hidden_target_calls": 0,
            }
            for trace in traces
        ]
        validate_residual_source_records(
            enriched_traces,
            heldout_family=request.heldout_family,
        )
        traces_sha256 = _write_verified_jsonl(
            request.output_dir / "source_query_traces.jsonl",
            enriched_traces,
        )
        raw_verification = verify_d1_raw_evidence(
            enriched_rows,
            enriched_traces,
            expected_methods=("score_greedy", "residual_ranker_bc"),
        )
        verify_d1_recorded_summaries(raw_verification, conditions)
        figure_paths = write_d1_evidence_plots(
            request.output_dir,
            raw_verification,
        )
        figures = {path.name: sha256_file(path) for path in figure_paths}
        uncertainty = paired_source_statistics(
            rows,
            learned_method="residual_ranker_bc",
            bootstrap_samples=10_000,
            seed=request.seed + 70_000,
        )
        decision = _decision(competence, threshold, conditions)
        deadline_check()
        role_digests = dict(context.role_audit)
        teacher_roles = _mapping(
            teacher_manifest["roles"],
            label="D1 teacher roles",
        )
        final = {
            **base,
            "status": "complete",
            "training_performed": True,
            "teacher_completed": True,
            "elapsed_seconds": clock() - started,
            "runtime_components_seconds": {
                "verified_source_loading": source_load_seconds,
                "teacher_collection_or_verified_reuse": teacher_seconds,
                "bc_threshold_and_competence": bc_and_calibration_seconds,
                "paired_source_evaluation": evaluation_seconds,
            },
            "source_manifest_sha256": context.source_manifest_sha256,
            "source_split_roles": role_digests,
            "teacher_cache": dict(teacher_manifest),
            "training_diagnostics": training,
            "threshold_selection": threshold,
            "competence_gate": competence,
            "checkpoint": {
                "name": checkpoint_path.name,
                "sha256": checkpoint_sha256,
                "persistent_digest": policy.persistent_digest(),
            },
            "source_evaluation": conditions,
            "paired_uncertainty": uncertainty,
            "source_model_calls": sum(row.total_target_calls for row in rows)
            + sum(
                int(_mapping(metrics, label="D1 teacher role")["source_calls"])
                for metrics in teacher_roles.values()
            ),
            "results_sha256": results_sha256,
            "query_traces_sha256": traces_sha256,
            "raw_evidence_verification": raw_verification,
            "figures": figures,
            "gpu_memory": _gpu_memory_record(),
            "d1_decision": decision,
            "limitations": (
                "D1a is one fixed seed on visible source families. The "
                "evaluation images were previously used by Phase 2/D0 source "
                "diagnostics, so intervals are descriptive and not confirmatory."
            ),
        }
        write_verified_json(manifest_path, final)
        _verify_complete_d1_children(request, final)
        return final
    except ResidualD1Deadline:
        failed = {
            **base,
            "status": "deadline_reached",
            "training_performed": training_performed,
            "teacher_completed": teacher_completed,
            "elapsed_seconds": clock() - started,
            "gpu_memory": _gpu_memory_record(),
            "failure_code": "d1a_bounded_deadline_reached",
            "d1_decision": {
                "passed": False,
                "eligible_for_d1b_source_only_ppo": False,
                "authorizes_hidden_target_evaluation": False,
            },
        }
        write_verified_json(manifest_path, failed)
        return failed
    except Exception as error:
        failed = {
            **base,
            "status": "failed",
            "training_performed": training_performed,
            "teacher_completed": teacher_completed,
            "elapsed_seconds": clock() - started,
            "gpu_memory": _gpu_memory_record(),
            "failure_code": f"d1a_{type(error).__name__.lower()}",
            "d1_decision": {
                "passed": False,
                "eligible_for_d1b_source_only_ppo": False,
                "authorizes_hidden_target_evaluation": False,
            },
        }
        write_verified_json(manifest_path, failed)
        raise


def run_residual_d1_from_datasets(
    request: ResidualD1Request,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
    progress: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
    external_deadline_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Run D1a under an exclusive lock in a fresh output directory."""

    if external_deadline_check is not None and not callable(external_deadline_check):
        raise TypeError("external D1 deadline check must be callable")
    request.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = request.output_dir / ".d1.lock"
    with exclusive_file_lock(lock_path):
        existing = _existing_d1_manifest(request)
        if existing is not None and existing.get("status") == "complete":
            return dict(existing)
        return _run_residual_d1_from_datasets(
            request,
            train_dataset,
            test_dataset,
            dataset_version=dataset_version,
            dataset_content_sha256=dataset_content_sha256,
            runtime_environment=runtime_environment,
            progress=progress,
            clock=clock,
            external_deadline_check=external_deadline_check,
        )


__all__ = (
    "ResidualD1Deadline",
    "_existing_d1_manifest",
    "build_residual_d1_dry_run",
    "evaluate_residual_d1",
    "load_d1_source_context",
    "load_residual_d1_source",
    "run_residual_d1_from_datasets",
)
