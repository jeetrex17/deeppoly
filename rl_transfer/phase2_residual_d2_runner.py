"""RTX-only execution for the preregistered D2 source-only study.

This runner deliberately evaluates only the two sealed source families.  It
collects one fresh teacher cache, fits three independently initialised
GroupDRO residual rankers, applies a conservative global threshold, and
records all six paired source cells.  No code path receives a hidden victim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import Dataset

from .artifacts import exclusive_file_lock, save_recurrent_checkpoint, sha256_file
from .cifar_manifest import git_revision
from .phase2_residual_d1_evaluation import evaluate_residual_d1
from .phase2_residual_d1_teacher import (
    D1_HIDDEN_DIM,
    D1_PRIOR_TEMPERATURE,
    D1_TRAIN_DECISIONS,
    D1_VALIDATION_DECISIONS,
    _collect_teacher_blocks,
    _step_record,
    _write_verified_jsonl,
)
from .phase2_residual_d2 import (
    D2_POLICY_SEEDS,
    D2_SOURCE_FAMILIES,
    D2SourceMetric,
    ResidualD2Request,
    residual_d2_promotion_decision,
)
from .phase2_residual_d2_source import D2SourceContext, load_d2_source_context
from .recurrent import PPOConfig, RecurrentAttackPolicy
from .residual_groupdro import fit_groupdro_residual_ranker_bc
from .residual_ranker import (
    ResidualRankerPolicy,
    evaluate_residual_ranker_examples,
    select_confidence_threshold,
)
from .verified_artifacts import write_verified_json


class ResidualD2Deadline(TimeoutError):
    """Raised between finite D2 units of source-only work."""


@dataclass(frozen=True)
class _EvaluationRequest:
    seed: int
    heldout_family: str


def _request_digest(request: ResidualD2Request) -> str:
    payload = {
        **asdict(request),
        "source_manifest": str(request.source_manifest),
        "source_root": str(request.source_root),
        "output_dir": str(request.output_dir),
        "data_root": str(request.data_root),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _metric(condition: Mapping[str, object], *, seed: int, family: str) -> D2SourceMetric:
    methods = condition.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("D2 paired condition lacks method summaries")
    baseline = methods.get("score_greedy")
    # The D1 paired evaluator is reused for its audited attack accounting and
    # therefore retains its historical method label.  In D2 that label denotes
    # the GroupDRO-trained policy passed into the evaluator.
    learned = methods.get("residual_ranker_groupdro_bc") or methods.get(
        "residual_ranker_bc"
    )
    if not isinstance(baseline, Mapping) or not isinstance(learned, Mapping):
        raise ValueError("D2 paired condition lacks a paired baseline or learner")
    def number(values: Mapping[str, object], name: str) -> float:
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"D2 paired condition has invalid {name}")
        return float(value)
    def asr(values: Mapping[str, object]) -> float:
        eligible = number(values, "eligible")
        if eligible <= 0:
            raise ValueError("D2 paired condition has no eligible samples")
        return number(values, "successes") / eligible
    return D2SourceMetric(
        seed=seed,
        family=family,
        baseline_asr=asr(baseline),
        learned_asr=asr(learned),
        baseline_query_auc=number(baseline, "asr_query_auc"),
        learned_query_auc=number(learned, "asr_query_auc"),
    )


def _family_safe_threshold(selection: Mapping[str, object]) -> tuple[float, bool, dict[str, object]]:
    candidates = selection.get("candidate_evaluations")
    if not isinstance(candidates, list):
        raise ValueError("D2 threshold selection lacks candidates")
    selected = next(
        (candidate for candidate in candidates if candidate.get("threshold") == selection.get("threshold")),
        None,
    )
    fallback = next(
        (candidate for candidate in candidates if candidate.get("selection_mode") == "always_fallback"),
        None,
    )
    if not isinstance(selected, Mapping) or not isinstance(fallback, Mapping):
        raise ValueError("D2 threshold selection lacks a verified fallback")
    by_family = selected.get("by_source_family")
    fallback_by_family = fallback.get("by_source_family")
    if not isinstance(by_family, Mapping) or not isinstance(
        fallback_by_family,
        Mapping,
    ):
        raise ValueError("D2 threshold selection lacks source-family metrics")
    safe = all(
        isinstance(by_family.get(family), Mapping)
        and isinstance(fallback_by_family.get(family), Mapping)
        and float(by_family[family]["accuracy"])
        >= float(fallback_by_family[family]["accuracy"])
        for family in D2_SOURCE_FAMILIES
    )
    chosen = selected if safe else fallback
    threshold = chosen.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("D2 threshold is invalid")
    return float(threshold), safe, dict(chosen)


def _plain_training(training: Mapping[str, object]) -> dict[str, object]:
    """Convert the immutable GroupDRO audit objects into evidence JSON."""

    state = training.get("groupdro_state")
    audits = training.get("groupdro_audits")
    if not hasattr(state, "as_dict") or not isinstance(audits, tuple):
        raise ValueError("D2 GroupDRO training lacks serializable audit evidence")
    if any(not hasattr(audit, "as_dict") for audit in audits):
        raise ValueError("D2 GroupDRO audit contains an invalid record")
    return {
        **training,
        "groupdro_state": state.as_dict(),
        "groupdro_audits": [audit.as_dict() for audit in audits],
    }


def _evaluate(
    context: D2SourceContext,
    policy: ResidualRankerPolicy,
    *,
    seed: int,
    deadline_check: Callable[[], None],
    progress: Callable[[str], None],
) -> tuple[dict[str, object], list[object], list[dict[str, object]]]:
    return evaluate_residual_d1(
        _EvaluationRequest(seed=seed, heldout_family="modern_cnn"),
        context,  # type: ignore[arg-type]
        policy,
        deadline_check=deadline_check,
        progress=progress,
    )


def run_residual_d2_from_datasets(
    request: ResidualD2Request,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
    progress: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
    deadline_seconds: float = 8 * 60 * 60.0,
) -> dict[str, object]:
    """Run D2 once on CUDA and preserve source-only provenance and evidence."""

    if request.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2 execution requires the RTX CUDA workstation")
    if not isinstance(deadline_seconds, (int, float)) or deadline_seconds <= 0:
        raise ValueError("D2 deadline must be positive")
    request.output_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(request.output_dir / ".d2.lock"):
        manifest_path = request.output_dir / "d2_manifest.json"
        if manifest_path.exists():
            raise ValueError("D2 output already has a manifest; choose a new directory")
        started = clock()
        deadline = started + float(deadline_seconds)
        def deadline_check() -> None:
            if clock() >= deadline:
                raise ResidualD2Deadline("D2 source-only deadline reached")

        request_sha256 = _request_digest(request)
        base = {
            "schema_version": 1,
            "study": "phase2_d2_source_only_groupdro",
            "status": "running",
            "request_sha256": request_sha256,
            "git_revision": git_revision(),
            "dataset_version": dataset_version,
            "dataset_content_sha256": dataset_content_sha256,
            "runtime_environment": dict(runtime_environment),
            "source_only": True,
            "hidden_target_calls": 0,
            "hidden_target_evaluation_performed": False,
            "authorizes_hidden_target_evaluation": False,
        }
        write_verified_json(manifest_path, base)
        try:
            torch.cuda.reset_peak_memory_stats()
            context = load_d2_source_context(
                request,
                train_dataset,
                test_dataset,
                dataset_content_sha256=dataset_content_sha256,
            )
            attack = context.config.attack_config()
            progress("[d2] collecting fresh GroupDRO teacher trajectories")
            train_steps, train_teacher = _collect_teacher_blocks(
                victims=context.teacher_victims,
                samples=context.train_samples,
                config=attack,
                episodes=len(context.train_samples),
                decisions=D1_TRAIN_DECISIONS,
                seed=100_223,
                role="d2_groupdro_training",
                deadline_check=deadline_check,
                progress=progress,
            )
            threshold_steps, threshold_teacher = _collect_teacher_blocks(
                victims=context.teacher_victims,
                samples=context.threshold_samples,
                config=attack,
                episodes=len(context.threshold_samples),
                decisions=D1_VALIDATION_DECISIONS,
                seed=100_227,
                role="d2_threshold_selection",
                deadline_check=deadline_check,
                progress=progress,
            )
            competence_steps, competence_teacher = _collect_teacher_blocks(
                victims=context.teacher_victims,
                samples=context.competence_samples,
                config=attack,
                episodes=len(context.competence_samples),
                decisions=D1_VALIDATION_DECISIONS,
                seed=100_229,
                role="d2_competence_gate",
                deadline_check=deadline_check,
                progress=progress,
            )
            teacher_records = [
                _step_record(step, role="groupdro_training") for step in train_steps
            ] + [
                _step_record(step, role="threshold_selection") for step in threshold_steps
            ] + [
                _step_record(step, role="competence_gate") for step in competence_steps
            ]
            teacher_sha256 = _write_verified_jsonl(
                request.output_dir / "teacher_examples.jsonl", teacher_records
            )
            seed_runs: dict[str, object] = {}
            metrics: list[D2SourceMetric] = []
            all_rows: list[dict[str, object]] = []
            all_traces: list[dict[str, object]] = []
            for policy_seed in D2_POLICY_SEEDS:
                deadline_check()
                progress(f"[d2] seed {policy_seed}: fitting 12-epoch GroupDRO BC")
                backbone = RecurrentAttackPolicy(
                    attack.recurrent_observation_dim,
                    attack.action_dim,
                    hidden_dim=D1_HIDDEN_DIM,
                    seed=policy_seed,
                    config=PPOConfig(
                        learning_rate=context.config.policy_learning_rate,
                        entropy_weight=context.config.policy_entropy_weight,
                        update_epochs=4,
                    ),
                    actor_mode="action_conditioned",
                    action_grid_size=attack.grid_size,
                ).to(torch.device("cuda"))
                training = fit_groupdro_residual_ranker_bc(
                    backbone,
                    train_steps,
                    epochs=request.bc_epochs,
                    seed=policy_seed,
                    prior_seed=policy_seed + 50_000,
                    prior_temperature=D1_PRIOR_TEMPERATURE,
                    groupdro_eta=request.groupdro_eta,
                    required_source_families=D2_SOURCE_FAMILIES,
                    deadline_check=deadline_check,
                )
                selection = select_confidence_threshold(
                    backbone,
                    threshold_steps,
                    seed=policy_seed + 50_000,
                    prior_temperature=D1_PRIOR_TEMPERATURE,
                    required_source_families=D2_SOURCE_FAMILIES,
                    deadline_check=deadline_check,
                )
                threshold, teacher_safe, selected_threshold = _family_safe_threshold(selection)
                policy = ResidualRankerPolicy(
                    backbone,
                    confidence_threshold=threshold,
                    prior_temperature=D1_PRIOR_TEMPERATURE,
                    overrides_enabled=teacher_safe and selected_threshold.get("selection_mode") != "always_fallback",
                )
                threshold_context = replace(
                    context,
                    evaluation_indices=context.threshold_indices,
                    evaluation_samples=context.threshold_samples,
                )
                threshold_conditions, _, _ = _evaluate(
                    threshold_context, policy, seed=policy_seed, deadline_check=deadline_check, progress=progress
                )
                threshold_attack_safe = all(
                    _metric(threshold_conditions[family], seed=policy_seed, family=family).non_regression
                    for family in D2_SOURCE_FAMILIES
                )
                if not threshold_attack_safe:
                    policy = ResidualRankerPolicy(
                        backbone,
                        confidence_threshold=math.inf,
                        prior_temperature=D1_PRIOR_TEMPERATURE,
                        overrides_enabled=False,
                    )
                competence = evaluate_residual_ranker_examples(
                    policy,
                    competence_steps,
                    prior_seed=policy_seed + 50_000,
                    required_source_families=D2_SOURCE_FAMILIES,
                    deadline_check=deadline_check,
                )
                conditions, rows, traces = _evaluate(
                    context, policy, seed=policy_seed, deadline_check=deadline_check, progress=progress
                )
                cells = tuple(_metric(conditions[family], seed=policy_seed, family=family) for family in D2_SOURCE_FAMILIES)
                metrics.extend(cells)
                checkpoint = request.output_dir / f"residual_ranker_groupdro_seed_{policy_seed}.pt"
                checkpoint_sha256 = save_recurrent_checkpoint(
                    checkpoint, backbone,
                    {"kind": "phase2_d2_source_only_groupdro", "policy_seed": policy_seed, "request_sha256": request_sha256, "hidden_target_calls": 0},
                )
                seed_runs[str(policy_seed)] = {
                    "training": _plain_training(training),
                    "threshold_selection": selection,
                    "selected_threshold": selected_threshold,
                    "threshold_teacher_safe": teacher_safe,
                    "threshold_attack_safe": threshold_attack_safe,
                    "competence": competence,
                    "source_evaluation": conditions,
                    "checkpoint": {"name": checkpoint.name, "sha256": checkpoint_sha256},
                }
                all_rows.extend({**asdict(row), "seed": policy_seed, "action_trace": list(row.action_trace)} for row in rows)
                all_traces.extend({**trace, "seed": policy_seed} for trace in traces)
            results_sha256 = _write_verified_jsonl(request.output_dir / "source_results.jsonl", all_rows)
            traces_sha256 = _write_verified_jsonl(request.output_dir / "source_query_traces.jsonl", all_traces)
            source_gates_passed = all(
                float(seed_runs[str(seed)]["competence"]["gated_top1_accuracy"]) >= float(seed_runs[str(seed)]["competence"]["prior_top1_accuracy"])
                and float(seed_runs[str(seed)]["competence"]["soft_cross_entropy"]) <= float(seed_runs[str(seed)]["competence"]["prior_soft_cross_entropy"])
                for seed in D2_POLICY_SEEDS
            )
            decision = residual_d2_promotion_decision(
                tuple(metrics), source_gates_passed=source_gates_passed, artifact_audits_passed=True
            )
            final = {
                **base, "status": "complete", "elapsed_seconds": clock() - started,
                "source_manifest_sha256": context.source_manifest_sha256,
                "source_roles": {role.name: list(role.sample_ids) for role in context.roles.as_tuple},
                "teacher": {"examples_sha256": teacher_sha256, "training": train_teacher, "threshold": threshold_teacher, "competence": competence_teacher},
                "seed_runs": seed_runs, "metrics": [asdict(cell) for cell in metrics],
                "d2_decision": asdict(decision), "source_gates_passed": source_gates_passed,
                "results_sha256": results_sha256, "query_traces_sha256": traces_sha256,
                "hidden_target_calls": 0, "hidden_target_evaluation_performed": False,
                "authorizes_hidden_target_evaluation": False,
                "gpu_memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            }
            write_verified_json(manifest_path, final)
            return final
        except Exception as error:
            failed = {**base, "status": "failed", "elapsed_seconds": clock() - started, "failure_code": type(error).__name__, "hidden_target_calls": 0}
            write_verified_json(manifest_path, failed)
            raise


__all__ = ("ResidualD2Deadline", "run_residual_d2_from_datasets")
