"""Stage A source-only Phase 1 checkpoint diagnostic and scheduling."""

from __future__ import annotations

import math
import time
from typing import Callable, Mapping, Sequence

from torch.utils.data import Dataset

from .phase2_temperature_evaluation import (
    _StageDeadlineReached,
    _check_deadline,
    evaluate_temperature_fold,
)
from .phase2_temperature_manifest import (
    FOLDS,
    STAGE_A_ELIGIBLE_IMAGES,
    STAGE_A_MAX_SECONDS,
    STAGE_A_POLICY_SEEDS,
    STAGE_A_RANKING_RULE,
    STAGE_A_RANKING_TIE_BAND,
    STAGE_A_TEMPERATURES,
    Phase1Selection,
    Phase1SourceFold,
    StageARequest,
    _require_mapping,
    load_phase1_source_selection,
    select_fixed_source_development_indices,
)
from .phase2_temperature_output import (
    build_stage_a_dry_run,
    portable_artifact_path,
    write_verified_jsonl,
)
from .results import ResearchResultRow
from .verified_artifacts import write_verified_json

__all__ = (
    "FOLDS",
    "STAGE_A_ELIGIBLE_IMAGES",
    "STAGE_A_MAX_SECONDS",
    "STAGE_A_POLICY_SEEDS",
    "STAGE_A_RANKING_RULE",
    "STAGE_A_RANKING_TIE_BAND",
    "STAGE_A_TEMPERATURES",
    "Phase1Selection",
    "Phase1SourceFold",
    "StageARequest",
    "build_stage_a_dry_run",
    "evaluate_temperature_fold",
    "load_phase1_source_selection",
    "run_temperature_screen_from_datasets",
    "select_fixed_source_development_indices",
    "select_stage_a_temperature",
    "write_verified_jsonl",
)


def _finite_metric(
    mapping: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> float:
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} {key} is not finite")
    return float(value)


def select_stage_a_temperature(
    fold_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the preregistered Phase 1 diagnostic ranking rule."""

    summaries = tuple(fold_summaries)
    fold_names = tuple(
        str(summary.get("heldout_family", ""))
        for summary in summaries
    )
    complete = (
        len(summaries) == len(FOLDS)
        and set(fold_names) == set(FOLDS)
        and len(set(fold_names)) == len(FOLDS)
        and all(summary.get("complete", True) is True for summary in summaries)
    )
    if not complete:
        return {
            "complete": False,
            "decision_scope": "phase1_checkpoint_diagnostic_only",
            "diagnostic_only": True,
            "applies_to_new_phase2_architecture": False,
            "authorizes_phase2_deployment_temperature": False,
            "selected_temperature": 1.0,
            "diagnostic_temperature_for_phase1_checkpoint": 1.0,
            "ranked_winner": None,
            "replaced_default": False,
            "reason": "all three complete folds are required",
            "ranking_rule": STAGE_A_RANKING_RULE,
            "ranking_tie_band": STAGE_A_RANKING_TIE_BAND,
            "ranking_tie_reference": "best_macro_asr_gain_vs_score",
        }

    aggregates: dict[float, dict[str, object]] = {}
    for temperature in STAGE_A_TEMPERATURES:
        per_fold: list[dict[str, object]] = []
        for summary in summaries:
            score = _require_mapping(
                summary.get("score_greedy"),
                label="fold score-greedy metrics",
            )
            score_asr = _finite_metric(
                score,
                "asr",
                label="score-greedy",
            )
            score_auc = _finite_metric(
                score,
                "auc",
                label="score-greedy",
            )
            temperatures = _require_mapping(
                summary.get("temperatures"),
                label="fold temperature metrics",
            )
            raw = temperatures.get(str(temperature))
            metrics = _require_mapping(
                raw,
                label=f"temperature {temperature}",
            )
            asr = _finite_metric(
                metrics,
                "asr",
                label=f"temperature {temperature}",
            )
            auc = _finite_metric(
                metrics,
                "auc",
                label=f"temperature {temperature}",
            )
            asr_gain = _finite_metric(
                metrics,
                "asr_gain_vs_score",
                label=f"temperature {temperature}",
            )
            auc_gain = _finite_metric(
                metrics,
                "auc_gain_vs_score",
                label=f"temperature {temperature}",
            )
            if (
                not math.isclose(
                    asr_gain,
                    asr - score_asr,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    auc_gain,
                    auc - score_auc,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "temperature gains do not match the paired control"
                )
            per_fold.append(
                {
                    "heldout_family": summary["heldout_family"],
                    "asr": asr,
                    "auc": auc,
                    "asr_gain_vs_score": asr_gain,
                    "auc_gain_vs_score": auc_gain,
                    "normalized_action_entropy": _finite_metric(
                        metrics,
                        "normalized_action_entropy",
                        label=f"temperature {temperature}",
                    ),
                    "frozen": metrics.get("frozen") is True,
                }
            )
        aggregates[temperature] = {
            "mean_asr": sum(float(item["asr"]) for item in per_fold)
            / len(per_fold),
            "mean_auc": sum(float(item["auc"]) for item in per_fold)
            / len(per_fold),
            "mean_asr_gain_vs_score": sum(
                float(item["asr_gain_vs_score"]) for item in per_fold
            )
            / len(per_fold),
            "mean_auc_gain_vs_score": sum(
                float(item["auc_gain_vs_score"]) for item in per_fold
            )
            / len(per_fold),
            "folds": per_fold,
        }
    best_asr_gain = max(
        float(metrics["mean_asr_gain_vs_score"])
        for metrics in aggregates.values()
    )
    tied = tuple(
        temperature
        for temperature in STAGE_A_TEMPERATURES
        if best_asr_gain
        - float(aggregates[temperature]["mean_asr_gain_vs_score"])
        <= STAGE_A_RANKING_TIE_BAND + 1e-12
    )
    winner = min(
        tied,
        key=lambda temperature: (
            -float(
                aggregates[temperature]["mean_auc_gain_vs_score"]
            ),
            temperature,
        ),
    )
    default = aggregates[1.0]
    candidate = aggregates[winner]
    entropy_and_freeze_pass = all(
        0.10
        <= float(item["normalized_action_entropy"])
        <= 0.95
        and item["frozen"] is True
        for item in candidate["folds"]
    )
    replacement_checks = {
        "macro_asr_gain_over_default_at_least_0p005": (
            float(candidate["mean_asr"])
            - float(default["mean_asr"])
            >= 0.005 - 1e-12
        ),
        "macro_auc_no_more_than_0p002_below_default": (
            float(candidate["mean_auc"])
            >= float(default["mean_auc"]) - 0.002 - 1e-12
        ),
        "fold_entropy_and_frozen_digests_pass": entropy_and_freeze_pass,
    }
    replace = (
        winner != 1.0 and all(replacement_checks.values())
    )
    return {
        "complete": True,
        "decision_scope": "phase1_checkpoint_diagnostic_only",
        "diagnostic_only": True,
        "applies_to_new_phase2_architecture": False,
        "authorizes_phase2_deployment_temperature": False,
        "ranked_winner": winner,
        "selected_temperature": winner if replace else 1.0,
        "diagnostic_temperature_for_phase1_checkpoint": (
            winner if replace else 1.0
        ),
        "replaced_default": replace,
        "replacement_checks": replacement_checks,
        "aggregates": {
            str(temperature): metrics
            for temperature, metrics in aggregates.items()
        },
        "ranking_rule": STAGE_A_RANKING_RULE,
        "ranking_tie_band": STAGE_A_RANKING_TIE_BAND,
        "ranking_tie_reference": "best_macro_asr_gain_vs_score",
    }


def _request_payload(request: StageARequest) -> dict[str, object]:
    return {
        "phase1_manifest": portable_artifact_path(
            request.phase1_manifest,
            relative_to=request.phase1_root,
        ),
        "phase1_root": portable_artifact_path(request.phase1_root),
        "output_dir": portable_artifact_path(request.output_dir),
        "data_root": portable_artifact_path(request.data_root),
        "seeds": list(request.seeds),
        "folds": list(request.folds),
        "temperatures": list(request.temperatures),
        "eligible_images_per_family": (
            request.eligible_images_per_family
        ),
        "deadline_seconds": request.deadline_seconds,
        "device": request.device,
        "download": request.download,
    }


def run_temperature_screen_from_datasets(
    request: StageARequest,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str = "in-memory",
    dataset_content_sha256: str | None = None,
    fold_evaluator: Callable[..., tuple[
        dict[str, object],
        list[ResearchResultRow],
    ]] = evaluate_temperature_fold,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run only the selected Phase 1 exact-source Stage A folds."""

    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("dataset version must be a non-empty string")
    report = progress or (lambda _message: None)
    started = clock()
    deadline = started + float(request.deadline_seconds)
    selection = load_phase1_source_selection(request)
    recorded_runtime = selection.dataset_version.split(";", 1)[0]
    if dataset_version != recorded_runtime and not (
        dataset_version == "in-memory"
        and selection.dataset_version == "in-memory"
    ):
        raise ValueError("runtime dataset version differs from Phase 1")
    if (
        selection.dataset_content_sha256 is not None
        and dataset_content_sha256
        != selection.dataset_content_sha256
    ):
        raise ValueError("CIFAR content digest differs from Phase 1")
    if selection.target_calls != 0:
        raise ValueError("Phase 1 selection violates the source-only seal")
    fold_summaries: list[dict[str, object]] = []
    result_rows: list[ResearchResultRow] = []
    status = "complete"
    for fold in selection.folds:
        try:
            _check_deadline(clock, deadline)
            summary, rows = fold_evaluator(
                fold=fold,
                request=request,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                absolute_deadline=deadline,
                clock=clock,
                progress=report,
            )
        except _StageDeadlineReached as error:
            if error.partial_fold is not None:
                if (
                    error.partial_fold.get("target_calls") != 0
                    or error.partial_fold.get(
                        "target_evaluation_performed"
                    )
                    is not False
                ):
                    raise ValueError(
                        "partial Stage A fold violated the source-only seal"
                    )
                fold_summaries.append(error.partial_fold)
                result_rows.extend(error.rows)
            status = "deadline_reached"
            break
        if (
            summary.get("target_calls", 0) != 0
            or summary.get("target_evaluation_performed", False) is not False
        ):
            raise ValueError("Stage A fold violated the source-only seal")
        fold_summaries.append(summary)
        result_rows.extend(rows)
    if (
        status == "complete"
        and len(fold_summaries) != len(selection.folds)
    ):
        raise RuntimeError("Stage A scheduler lost a selected fold")
    if status == "complete" and request.folds != FOLDS:
        status = "selected_folds_complete"

    decision = select_stage_a_temperature(fold_summaries)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = request.output_dir / "stage_a_results.jsonl"
    rows_digest = write_verified_jsonl(rows_path, result_rows)
    elapsed = max(0.0, clock() - started)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "stage": "A",
        "name": "phase1-checkpoint-diagnostic-temperature-screen",
        "status": status,
        "decision_scope": "phase1_checkpoint_diagnostic_only",
        "diagnostic_only": True,
        "applies_to_new_phase2_architecture": False,
        "authorizes_phase2_deployment_temperature": False,
        "research_valid": False,
        "publication_candidate": False,
        "exploratory_screen": True,
        "training_performed": False,
        "request": _request_payload(request),
        "dataset_version": dataset_version,
        "dataset_content_sha256": dataset_content_sha256,
        "phase1_dataset_version": selection.dataset_version,
        "phase1_manifest": portable_artifact_path(
            selection.manifest_path,
            relative_to=request.phase1_root,
        ),
        "phase1_manifest_sha256": selection.manifest_sha256,
        "split_digest": selection.split_digest,
        "source_gate_digest": selection.source_gate_digest,
        "fold_summaries": fold_summaries,
        "phase1_diagnostic_temperature_decision": decision,
        "completed_folds": sum(
            summary.get("complete", True) is True
            for summary in fold_summaries
        ),
        "partial_folds": sum(
            summary.get("complete") is False
            for summary in fold_summaries
        ),
        "partial_results_preserved": any(
            summary.get("complete") is False
            for summary in fold_summaries
        ),
        "selected_folds": len(selection.folds),
        "source_model_calls": sum(
            int(summary.get("source_model_calls", 0))
            for summary in fold_summaries
        ),
        "target_calls": 0,
        "target_evaluation_performed": False,
        "results_path": portable_artifact_path(
            rows_path,
            relative_to=request.output_dir,
        ),
        "results_sha256": rows_digest,
        "elapsed_seconds": elapsed,
        "deadline_seconds": request.deadline_seconds,
        "deadline_reached": status == "deadline_reached",
    }
    write_verified_json(request.output_dir / "stage_a.json", manifest)
    return manifest
