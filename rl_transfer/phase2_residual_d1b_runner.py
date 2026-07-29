"""Production source-only execution and evidence export for D1b."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import time

from torch.utils.data import Dataset

from .artifacts import (
    exclusive_file_lock,
    load_recurrent_checkpoint,
    save_recurrent_checkpoint,
    sha256_file,
)
from .cifar_manifest import code_digest, git_revision, git_worktree_state
from .phase2_residual_d1 import (
    D1_HELDOUT_FAMILY,
    D1_SEED,
    D1_SOURCE_FAMILIES,
    ResidualD1Request,
    validate_residual_source_records,
    validate_source_only_payload as _source_only,
)
from .phase2_residual_d1_cache import load_residual_teacher_cache
from .phase2_residual_d1_evaluation import evaluate_residual_policy_cohort
from .phase2_residual_d1_evidence import (
    verify_d1_raw_evidence,
    verify_d1_recorded_summaries,
    write_d1_evidence_plots,
)
from .phase2_residual_d1_reporting import paired_source_statistics
from .phase2_residual_d1_runner import (
    ResidualD1Deadline,
    _gpu_memory_record,
)
from .phase2_residual_d1_source import load_d1_source_context
from .phase2_residual_d1_teacher import (
    D1_HIDDEN_DIM,
    _cache_binding,
    _write_verified_jsonl,
)
from .phase2_residual_d1b import (
    D1B_BLOCK_ENDPOINTS,
    D1B_METHODS,
    ResidualD1BDependencies,
    ResidualD1BEvaluationInputs,
    ResidualD1BResult,
    existing_residual_ppo_block,
    run_residual_d1b,
)
from .phase2_residual_d1b_artifacts import (
    ResidualD1BBlockStore,
    ResidualD1BStoreBinding,
    canonical_json_digest,
    clone_residual_policy,
)
from .phase2_residual_d1b_policy import (
    ResidualD1AArtifactBundle,
    ResidualD1BEvaluationPayload,
    apply_d1b_threshold,
    build_d1b_source_roles,
    evaluate_d1b_competence,
    select_d1b_threshold,
    verify_d1a_artifacts,
)
from .phase2_residual_d1b_reporting import residual_d1b_selection_decision
from .phase2_residual_d1b_verification import (
    d1b_block_records,
    verify_complete_d1b_children,
)
from .residual_ranker import ResidualRankerPolicy
from .results import ResearchResultRow
from .verified_artifacts import load_verified_json, write_verified_json

_STATIC_FILES = {
    ".d1b.lock",
    "d1b_manifest.json",
    "d1b_manifest.json.sha256",
    "residual_ranker_ppo.pt",
    "residual_ranker_ppo.pt.sha256",
    "source_results.jsonl",
    "source_results.jsonl.sha256",
    "source_query_traces.jsonl",
    "source_query_traces.jsonl.sha256",
    "asr_by_query.svg",
    "asr_by_query.svg.sha256",
    "final_asr.svg",
    "final_asr.svg.sha256",
}
_BLOCK_FILES = {
    name
    for endpoint in D1B_BLOCK_ENDPOINTS
    for name in (
        f"ppo_block_{endpoint:03d}.pt",
        f"ppo_block_{endpoint:03d}.pt.sha256",
        f"ppo_block_{endpoint:03d}.receipt.json",
        f"ppo_block_{endpoint:03d}.receipt.json.sha256",
    )
}
_NONRESUMABLE_FINAL_FILES = _STATIC_FILES - {
    ".d1b.lock",
    "d1b_manifest.json",
    "d1b_manifest.json.sha256",
}


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _plain(value: object, label: str) -> dict[str, object]:
    try:
        decoded = json.loads(
            json.dumps(
                _thaw(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must be an object")
    return decoded


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _seal() -> dict[str, object]:
    return {
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _validate_output(output_dir: Path) -> None:
    allowed = _STATIC_FILES | _BLOCK_FILES
    unexpected = {path.name for path in output_dir.iterdir()} - allowed
    if unexpected:
        raise ValueError(
            f"D1b output contains unexpected artifacts: {sorted(unexpected)}"
        )
    if any(path.is_symlink() for path in output_dir.iterdir()):
        raise ValueError("D1b output artifacts cannot be symlinks")


def _existing_manifest(
    output_dir: Path,
    request: ResidualD1Request,
    d1a_manifest: Mapping[str, object],
) -> dict[str, object] | None:
    path = output_dir / "d1b_manifest.json"
    sidecar = path.with_suffix(".json.sha256")
    state = path.is_file(), sidecar.is_file()
    if state == (False, False):
        existing = {item.name for item in output_dir.iterdir()} - {".d1b.lock"}
        if existing:
            raise ValueError("D1b partial artifacts exist without a manifest")
        return None
    if state != (True, True):
        raise ValueError("D1b manifest artifact pair is incomplete")
    manifest = load_verified_json(path)
    if manifest.get("request_sha256") != request.digest() or manifest.get(
        "d1a_manifest_digest"
    ) != canonical_json_digest(d1a_manifest):
        raise ValueError("D1b resume request or D1a binding mismatch")
    _source_only(manifest, "D1b existing manifest")
    status = manifest.get("status")
    if status in {"failed", "deadline_reached"}:
        raise ValueError("D1b terminal manifest cannot be retried or overwritten")
    if status not in {"running", "complete", "skipped"}:
        raise ValueError("D1b existing manifest has an invalid status")
    names = {item.name for item in output_dir.iterdir()}
    if status == "running" and names & _NONRESUMABLE_FINAL_FILES:
        raise ValueError("D1b partial final evidence cannot be retried or overwritten")
    return manifest


def _row_records(rows: Sequence[ResearchResultRow]) -> list[dict[str, object]]:
    return [
        {
            **asdict(row),
            "heldout_family": D1_HELDOUT_FAMILY,
            "source_slice": "seen_family_new_instance",
            "target_calls": 0,
            "hidden_target_calls": 0,
        }
        for row in rows
    ]


def _trace_records(
    traces: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            **dict(trace),
            "heldout_family": D1_HELDOUT_FAMILY,
            "source_slice": "seen_family_new_instance",
            "target_calls": 0,
            "hidden_target_calls": 0,
        }
        for trace in traces
    ]


def _paired_reports(
    rows: Sequence[ResearchResultRow],
) -> dict[str, object]:
    comparisons = (
        ("score_greedy", "residual_ranker_bc"),
        ("score_greedy", "residual_ranker_bc_ppo"),
        ("residual_ranker_bc", "residual_ranker_bc_ppo"),
    )
    reports: dict[str, object] = {}
    for offset, (control, learned) in enumerate(comparisons):
        pair_rows = tuple(row for row in rows if row.method in {control, learned})
        reports[f"{control}_vs_{learned}"] = paired_source_statistics(
            pair_rows,
            control_method=control,
            learned_method=learned,
            bootstrap_samples=10_000,
            seed=D1_SEED + 70_000 + 10_000 * offset,
        )
    return reports


def _evaluate_d1b_source(
    output_dir: Path,
    inputs: ResidualD1BEvaluationInputs,
    *,
    deadline_check: Callable[[], None],
    progress: Callable[[str], None],
) -> dict[str, object]:
    """Evaluate score, frozen BC, and PPO together on the reserved D1b cohort."""

    if not isinstance(inputs, ResidualD1BEvaluationInputs):
        raise TypeError("D1b evaluation requires verified core inputs")
    if not isinstance(inputs.cohort, ResidualD1BEvaluationPayload):
        raise TypeError("D1b evaluation cohort payload is invalid")
    if (
        inputs.methods != D1B_METHODS
        or inputs.sample_ids != inputs.cohort.sample_ids
        or inputs.source_families != D1_SOURCE_FAMILIES
        or inputs.query_budget != 50
        or not isinstance(inputs.bc_policy, ResidualRankerPolicy)
        or not isinstance(inputs.ppo_policy, ResidualRankerPolicy)
        or inputs.bc_policy.persistent_digest() != inputs.bc_policy_digest
        or inputs.ppo_policy.persistent_digest() != inputs.ppo_policy_digest
    ):
        raise ValueError("D1b evaluation inputs violate the locked cohort")
    payload = inputs.cohort
    conditions: dict[str, object] = {}
    rows: list[ResearchResultRow] = []
    traces: list[dict[str, object]] = []
    for offset, family in enumerate(D1_SOURCE_FAMILIES):
        deadline_check()
        condition, family_rows, family_traces = evaluate_residual_policy_cohort(
            policies={
                "residual_ranker_bc": inputs.bc_policy,
                "residual_ranker_bc_ppo": inputs.ppo_policy,
            },
            victims=payload.source_victims[family],
            samples=payload.source_samples,
            indices=inputs.sample_ids,
            attack=payload.attack_config,  # type: ignore[arg-type]
            family=family,
            seed=inputs.seed + 900_000 + offset,
            heldout_family=D1_HELDOUT_FAMILY,
            source_slice="seen_family_new_instance",
            deadline_check=deadline_check,
            progress=progress,
        )
        conditions[family] = condition
        rows.extend(family_rows)
        traces.extend(family_traces)
    enriched_rows = _row_records(rows)
    enriched_traces = _trace_records(traces)
    validate_residual_source_records(
        enriched_rows,
        heldout_family=D1_HELDOUT_FAMILY,
    )
    validate_residual_source_records(
        enriched_traces,
        heldout_family=D1_HELDOUT_FAMILY,
    )
    results_sha256 = _write_verified_jsonl(
        output_dir / "source_results.jsonl",
        enriched_rows,
    )
    traces_sha256 = _write_verified_jsonl(
        output_dir / "source_query_traces.jsonl",
        enriched_traces,
    )
    raw = verify_d1_raw_evidence(
        enriched_rows,
        enriched_traces,
        expected_methods=D1B_METHODS,
    )
    verify_d1_recorded_summaries(raw, conditions)
    figures = {
        path.name: sha256_file(path)
        for path in write_d1_evidence_plots(output_dir, raw)
    }
    decision = residual_d1b_selection_decision(
        inputs.competence_gate,
        inputs.threshold_selection,
        conditions,
    )
    calls = sum(
        int(_mapping(condition, "D1b condition")["source_model_calls"])
        for condition in conditions.values()
    )
    return {
        "source_evaluation": conditions,
        "rows": rows,
        "results_sha256": results_sha256,
        "query_traces_sha256": traces_sha256,
        "raw_evidence_verification": raw,
        "paired_uncertainty": _paired_reports(rows),
        "figures": figures,
        "decision": decision,
        "source_model_calls": calls,
        **_seal(),
    }


def _calibrated_checkpoint(
    output_dir: Path,
    policy: ResidualRankerPolicy,
    *,
    request: ResidualD1Request,
    d1a_manifest_digest: str,
    source_roles_digest: str,
) -> dict[str, object]:
    path = output_dir / "residual_ranker_ppo.pt"
    sidecar = path.with_suffix(".pt.sha256")
    metadata = {
        "schema_version": 1,
        "kind": "phase2_d1b_source_only_calibrated_ppo",
        "request_sha256": request.digest(),
        "d1a_manifest_digest": d1a_manifest_digest,
        "source_roles_digest": source_roles_digest,
        "policy_digest": policy.persistent_digest(),
        "confidence_threshold": policy.confidence_threshold,
        "prior_temperature": policy.prior_temperature,
        "overrides_enabled": policy.overrides_enabled,
        **_seal(),
    }
    state = path.is_file(), sidecar.is_file()
    if state == (False, False):
        digest = save_recurrent_checkpoint(path, policy.backbone, metadata)
    elif state != (True, True) or path.is_symlink() or sidecar.is_symlink():
        raise ValueError("D1b calibrated checkpoint artifact pair is unsafe")
    else:
        digest = sha256_file(path)
    loaded, loaded_metadata = load_recurrent_checkpoint(
        path,
        next(policy.parameters()).device,
        expected_observation_dim=policy.backbone.observation_dim,
        expected_action_dim=policy.action_dim,
        expected_hidden_dim=policy.backbone.hidden_dim,
        expected_actor_mode="action_conditioned",
    )
    reconstructed = ResidualRankerPolicy(
        loaded,
        confidence_threshold=policy.confidence_threshold,
        prior_temperature=policy.prior_temperature,
        overrides_enabled=policy.overrides_enabled,
    )
    if loaded_metadata != metadata or (
        reconstructed.persistent_digest() != policy.persistent_digest()
    ):
        raise ValueError("D1b calibrated checkpoint binding mismatch")
    return {
        "name": path.name,
        "sha256": digest,
        "persistent_digest": policy.persistent_digest(),
        "metadata_sha256": canonical_json_digest(metadata),
    }


def _base_manifest(
    request: ResidualD1Request,
    d1a_manifest: Mapping[str, object],
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "name": "phase2-d1b-residual-ranker-ppo",
        "status": "running",
        "diagnostic_only": True,
        "research_valid": False,
        "publication_candidate": False,
        "request_sha256": request.digest(),
        "d1a_manifest_digest": canonical_json_digest(d1a_manifest),
        "dataset_version": dataset_version,
        "dataset_content_sha256": dataset_content_sha256,
        "runtime_environment": _plain(
            runtime_environment,
            "D1b runtime environment",
        ),
        "code_digest": code_digest(),
        "git_revision": git_revision(),
        "git_worktree_state": git_worktree_state(),
        "heldout_family": D1_HELDOUT_FAMILY,
        "source_families": list(D1_SOURCE_FAMILIES),
        "seed": D1_SEED,
        "evaluation_role": "d1b_evaluation",
        **_seal(),
    }


def _skip_manifest(
    request: ResidualD1Request,
    output_dir: Path,
    d1a_manifest: Mapping[str, object],
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
) -> dict[str, object]:
    skipped = run_residual_d1b(
        d1a_manifest,
        None,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
    )
    result = {
        **_base_manifest(
            request,
            d1a_manifest,
            dataset_version=dataset_version,
            dataset_content_sha256=dataset_content_sha256,
            runtime_environment=runtime_environment,
        ),
        **_plain(skipped.manifest, "D1b skipped manifest"),
    }
    write_verified_json(output_dir / "d1b_manifest.json", result)
    return result


def run_residual_d1b_from_datasets(
    request: ResidualD1Request,
    output_dir: Path,
    d1a_manifest: Mapping[str, object],
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
    deadline_check: Callable[[], None],
    progress: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Run conditional PPO and export the reserved three-method D1b evidence."""

    if not isinstance(request, ResidualD1Request):
        raise TypeError("D1b production runner requires a D1 request")
    if not isinstance(d1a_manifest, Mapping):
        raise TypeError("D1b requires a D1a manifest")
    if not all(callable(item) for item in (deadline_check, progress, clock)):
        raise TypeError("D1b callbacks must be callable")
    destination = Path(output_dir).resolve()
    if destination != request.output_dir.parent / "d1b":
        raise ValueError("D1b output must be the sibling study-root/d1b directory")
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "d1b_manifest.json"
    with exclusive_file_lock(destination / ".d1b.lock"):
        _validate_output(destination)
        existing = _existing_manifest(destination, request, d1a_manifest)
        decision = _mapping(d1a_manifest.get("d1_decision"), "D1a decision")
        if decision.get("passed") is not True:
            if existing is not None:
                return existing
            return _skip_manifest(
                request,
                destination,
                d1a_manifest,
                dataset_version=dataset_version,
                dataset_content_sha256=dataset_content_sha256,
                runtime_environment=runtime_environment,
            )
        started = clock()
        base = _base_manifest(
            request,
            d1a_manifest,
            dataset_version=dataset_version,
            dataset_content_sha256=dataset_content_sha256,
            runtime_environment=runtime_environment,
        )
        store: ResidualD1BBlockStore | None = None
        try:
            deadline_check()
            component_started = clock()
            context = load_d1_source_context(
                request,
                train_dataset,
                test_dataset,
                dataset_content_sha256=dataset_content_sha256,
            )
            context_seconds = clock() - component_started
            binding, protocol = _cache_binding(request, context)
            cache = load_residual_teacher_cache(
                request.output_dir,
                expected_binding=binding,
                expected_protocol=protocol,
                action_dim=context.config.attack_config().action_dim,
                observation_dim=(
                    context.config.attack_config().recurrent_observation_dim
                ),
            )
            roles = build_d1b_source_roles(context, cache)
            bundle = ResidualD1AArtifactBundle(
                request=request,
                dataset_content_sha256=dataset_content_sha256,
                observation_dim=(
                    context.config.attack_config().recurrent_observation_dim
                ),
                action_dim=context.config.attack_config().action_dim,
                hidden_dim=D1_HIDDEN_DIM,
                device=request.device,
            )
            verified = verify_d1a_artifacts(d1a_manifest, bundle)
            store = ResidualD1BBlockStore(
                ResidualD1BStoreBinding(
                    root=destination,
                    device=request.device,
                    observation_dim=bundle.observation_dim,
                    action_dim=bundle.action_dim,
                    hidden_dim=bundle.hidden_dim,
                    d1a_manifest_digest=verified.manifest_digest,
                    d1a_checkpoint_sha256=verified.checkpoint_sha256,
                    bc_policy_digest=verified.bc_policy_digest,
                    source_roles_digest=roles.digest,
                )
            )
            resume = store.load_resume_state()
            if existing is not None and existing.get("status") == "complete":
                if resume is None or resume.completed_episodes != 200:
                    raise ValueError("D1b complete manifest lacks four PPO blocks")
                verify_complete_d1b_children(destination, existing)
                return existing
            write_verified_json(manifest_path, base)
            pending_metrics: list[dict[str, object]] = []

            def train_block(*args: object, **kwargs: object) -> object:
                if pending_metrics:
                    raise ValueError("D1b has uncommitted PPO block metrics")
                output = existing_residual_ppo_block(*args, **kwargs)
                pending_metrics.append(_plain(output, "D1b PPO metrics"))
                return output

            def save_block(
                policy: object,
                metadata: Mapping[str, object],
            ) -> object:
                if len(pending_metrics) != 1 or store is None:
                    raise ValueError("D1b checkpoint lacks exact block metrics")
                metrics = pending_metrics.pop()
                return store.save_block(policy, metadata, metrics)

            def verified_d1a(
                supplied_manifest: object,
                supplied_bundle: object,
            ) -> object:
                if (
                    supplied_bundle is not bundle
                    or canonical_json_digest(supplied_manifest)
                    != verified.manifest_digest
                ):
                    raise ValueError("D1b D1a verifier input binding changed")
                return verified

            dependencies = ResidualD1BDependencies(
                verify_d1a=verified_d1a,
                clone_policy=clone_residual_policy,
                policy_digest=lambda policy: policy.persistent_digest(),
                train_ppo_block=train_block,
                save_block_checkpoint=save_block,
                load_block_checkpoint=store.load_block,
                select_threshold=select_d1b_threshold,
                apply_threshold=apply_d1b_threshold,
                evaluate_competence=evaluate_d1b_competence,
            )
            component_started = clock()
            core = run_residual_d1b(
                d1a_manifest,
                bundle,
                roles,
                dependencies,
                resume_state=resume,
                deadline_check=deadline_check,
            )
            ppo_seconds = clock() - component_started
            if pending_metrics:
                raise ValueError("D1b core returned with uncommitted PPO metrics")
            if not isinstance(core, ResidualD1BResult) or (
                core.evaluation_inputs is None
            ):
                raise ValueError("passing D1a did not produce D1b evaluation inputs")
            core_manifest = _plain(core.manifest, "D1b core manifest")
            checkpoint = _calibrated_checkpoint(
                destination,
                core.evaluation_inputs.ppo_policy,
                request=request,
                d1a_manifest_digest=verified.manifest_digest,
                source_roles_digest=roles.digest,
            )
            component_started = clock()
            evidence = _evaluate_d1b_source(
                destination,
                core.evaluation_inputs,
                deadline_check=deadline_check,
                progress=progress,
            )
            evaluation_seconds = clock() - component_started
            training_calls = int(core_manifest["source_model_calls"])
            evaluation_calls = int(evidence["source_model_calls"])
            final = {
                **base,
                **core_manifest,
                "status": "complete",
                "request_sha256": request.digest(),
                "dataset_version": dataset_version,
                "dataset_content_sha256": dataset_content_sha256,
                "runtime_environment": _plain(
                    runtime_environment,
                    "D1b runtime environment",
                ),
                "elapsed_seconds": clock() - started,
                "runtime_components_seconds": {
                    "verified_context_cache_and_d1a_loading": context_seconds,
                    "ppo_blocks_threshold_and_competence": ppo_seconds,
                    "three_method_reserved_source_evaluation": (evaluation_seconds),
                },
                "checkpoint": checkpoint,
                "ppo_blocks": d1b_block_records(destination),
                "source_evaluation": evidence["source_evaluation"],
                "paired_uncertainty": evidence["paired_uncertainty"],
                "raw_evidence_verification": evidence["raw_evidence_verification"],
                "results_sha256": evidence["results_sha256"],
                "query_traces_sha256": evidence["query_traces_sha256"],
                "figures": evidence["figures"],
                "d1_decision": evidence["decision"],
                "source_model_calls_by_phase": {
                    "ppo_training": training_calls,
                    "reserved_evaluation": evaluation_calls,
                },
                "source_model_calls": training_calls + evaluation_calls,
                "gpu_memory": _gpu_memory_record(),
                "limitations": (
                    "D1b is one fixed seed on reused visible-source development "
                    "cohorts. Point estimates and paired image-bootstrap intervals "
                    "are descriptive, not confirmatory or non-inferiority evidence."
                ),
                **_seal(),
            }
            write_verified_json(manifest_path, final)
            verify_complete_d1b_children(destination, final)
            return final
        except ResidualD1Deadline:
            completed = sum(
                (destination / f"ppo_block_{endpoint:03d}.receipt.json").is_file()
                for endpoint in D1B_BLOCK_ENDPOINTS
            )
            deadline = {
                **base,
                "status": "deadline_reached",
                "elapsed_seconds": clock() - started,
                "ppo_blocks_completed": completed,
                "ppo_episodes_completed": completed * 50,
                "failure_code": "d1b_persisted_deadline_reached",
                "gpu_memory": _gpu_memory_record(),
                **_seal(),
            }
            write_verified_json(manifest_path, deadline)
            return deadline
        except Exception as error:
            failed = {
                **base,
                "status": "failed",
                "elapsed_seconds": clock() - started,
                "failure_code": f"d1b_{type(error).__name__.lower()}",
                "gpu_memory": _gpu_memory_record(),
                **_seal(),
            }
            write_verified_json(manifest_path, failed)
            raise


__all__ = ("_evaluate_d1b_source", "run_residual_d1b_from_datasets")
