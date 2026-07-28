"""Source-only temperature calibration over completed Phase 2 policies."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import math
import os
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from torch.utils.data import Dataset

from .cifar_manifest import code_digest, git_revision, git_worktree_state
from .phase2_calibration_evaluation import (
    CalibrationDeadlineReached,
    evaluate_calibration_fold,
)
from .phase2_calibration_manifest import (
    CALIBRATION_MAX_SECONDS,
    CALIBRATION_POLICY_SEEDS,
    CALIBRATION_TEMPERATURES,
    FOLDS,
    Phase2CalibrationRequest,
    load_phase2_calibration_source,
)
from .phase2_temperature_output import (
    portable_artifact_path,
    write_verified_jsonl,
)
from .reproducibility import seed_everything
from .verified_artifacts import write_verified_json


def _value(record: object, key: str) -> object:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _finite(record: Mapping[str, object], key: str, *, label: str) -> float:
    value = record.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} {key} must be finite")
    return float(value)


def select_calibration_temperature(
    fold_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require one temperature to match score ASR and AUC in every fold."""

    summaries = tuple(fold_summaries)
    names = tuple(str(summary.get("heldout_family", "")) for summary in summaries)
    complete = (
        len(summaries) == len(FOLDS)
        and set(names) == set(FOLDS)
        and len(set(names)) == len(FOLDS)
        and all(summary.get("complete") is True for summary in summaries)
    )
    base = {
        "complete": complete,
        "diagnostic_only": True,
        "decision_scope": "phase2_checkpoint_calibration_diagnostic",
        "authorizes_hidden_target_evaluation": False,
    }
    if not complete:
        return {
            **base,
            "calibration_useful": False,
            "stop_temperature_only_work": True,
            "selected_temperature": None,
            "qualifying_temperatures": [],
            "aggregates": {},
            "reason": "all three complete source folds are required",
        }

    aggregates: dict[float, dict[str, object]] = {}
    qualifying: list[float] = []
    for temperature in CALIBRATION_TEMPERATURES:
        fold_metrics: list[dict[str, object]] = []
        for summary in summaries:
            score = _value(summary, "score_greedy")
            temperatures = _value(summary, "temperatures")
            if not isinstance(score, Mapping) or not isinstance(
                temperatures,
                Mapping,
            ):
                raise ValueError("calibration fold metrics are incomplete")
            learned = temperatures.get(str(temperature))
            if not isinstance(learned, Mapping):
                raise ValueError(f"temperature {temperature} metrics are missing")
            score_asr = _finite(score, "asr", label="score-greedy")
            score_auc = _finite(score, "auc", label="score-greedy")
            learned_asr = _finite(
                learned,
                "asr",
                label=f"temperature {temperature}",
            )
            learned_auc = _finite(
                learned,
                "auc",
                label=f"temperature {temperature}",
            )
            fold_metrics.append(
                {
                    "heldout_family": summary["heldout_family"],
                    "asr": learned_asr,
                    "auc": learned_auc,
                    "asr_gain_vs_score": learned_asr - score_asr,
                    "auc_gain_vs_score": learned_auc - score_auc,
                    "frozen": learned.get("frozen") is True,
                }
            )
        aggregate = {
            "mean_asr_gain_vs_score": sum(
                float(item["asr_gain_vs_score"]) for item in fold_metrics
            )
            / len(fold_metrics),
            "mean_auc_gain_vs_score": sum(
                float(item["auc_gain_vs_score"]) for item in fold_metrics
            )
            / len(fold_metrics),
            "folds": fold_metrics,
        }
        aggregates[temperature] = aggregate
        if all(
            float(item["asr_gain_vs_score"]) >= -1e-12
            and float(item["auc_gain_vs_score"]) >= -1e-12
            and item["frozen"] is True
            for item in fold_metrics
        ):
            qualifying.append(temperature)

    selected = (
        min(
            qualifying,
            key=lambda temperature: (
                -float(aggregates[temperature]["mean_asr_gain_vs_score"]),
                -float(aggregates[temperature]["mean_auc_gain_vs_score"]),
                temperature,
            ),
        )
        if qualifying
        else None
    )
    return {
        **base,
        "calibration_useful": selected is not None,
        "stop_temperature_only_work": selected is None,
        "selected_temperature": selected,
        "qualifying_temperatures": qualifying,
        "aggregates": {
            str(temperature): metrics for temperature, metrics in aggregates.items()
        },
        "reason": (
            "a frozen temperature matched score-greedy ASR and AUC in every source fold"
            if selected is not None
            else "no frozen temperature matched both controls in every fold"
        ),
    }


def build_calibration_dry_run(
    request: Phase2CalibrationRequest,
    source: object,
) -> dict[str, object]:
    if (
        _value(source, "target_calls") != 0
        or _value(source, "target_evaluation_performed") is not False
    ):
        raise ValueError("Phase 2 source evidence contains target calls")
    folds = _value(source, "folds")
    if not isinstance(folds, (tuple, list)):
        raise ValueError("Phase 2 source folds are missing")
    return {
        "schema_version": 1,
        "mode": "source_only_phase2_checkpoint_calibration_diagnostic",
        "diagnostic_only": True,
        "research_valid": False,
        "source_manifest": portable_artifact_path(
            Path(str(_value(source, "manifest_path"))),
            relative_to=request.source_root,
        ),
        "source_manifest_sha256": _value(source, "manifest_sha256"),
        "selected_policy_seeds": list(request.seeds),
        "selected_folds": list(request.folds),
        "selected_cells": len(folds),
        "temperatures": list(request.temperatures),
        "maximum_wall_clock_seconds": request.deadline_seconds,
        "training_performed": False,
        "target_calls": 0,
        "target_evaluation_available": False,
    }


def _default_fold_evaluator(**arguments):
    return evaluate_calibration_fold(**arguments)


def _request_payload(
    request: Phase2CalibrationRequest,
) -> dict[str, object]:
    return {
        "source_manifest": portable_artifact_path(
            request.source_manifest,
            relative_to=request.source_root,
        ),
        "source_root": portable_artifact_path(request.source_root),
        "output_dir": portable_artifact_path(request.output_dir),
        "data_root": portable_artifact_path(request.data_root),
        "seeds": list(request.seeds),
        "folds": list(request.folds),
        "temperatures": list(request.temperatures),
        "deadline_seconds": request.deadline_seconds,
        "device": request.device,
        "download": request.download,
    }


@contextmanager
def _exclusive_output_directory(output_dir: Path):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "calibration output is not empty; refusing to overwrite evidence"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".calibration.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError(
            "another calibration process owns the output directory"
        ) from error
    try:
        os.write(descriptor, b"phase2-calibration\n")
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _locked_output(function):
    @wraps(function)
    def locked(request, *args, **kwargs):
        with _exclusive_output_directory(request.output_dir):
            return function(request, *args, **kwargs)

    return locked


@_locked_output
def run_calibration_from_datasets(
    request: Phase2CalibrationRequest,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str = "in-memory",
    dataset_content_sha256: str | None = None,
    runtime_environment: Mapping[str, object] | None = None,
    source_loader: Callable[[Phase2CalibrationRequest], object] = (
        load_phase2_calibration_source
    ),
    fold_evaluator: Callable[
        ...,
        tuple[
            Mapping[str, object],
            Sequence[object],
            Sequence[Mapping[str, object]],
        ],
    ] = _default_fold_evaluator,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Run the bounded frozen-policy diagnostic and persist verified evidence."""

    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("dataset version must be a non-empty string")
    report = progress or (lambda _message: None)
    started = clock()
    deadline = started + request.deadline_seconds
    seed_everything(request.seeds[0])
    calibration_code_digest = code_digest()
    calibration_git_revision = git_revision()
    calibration_worktree = git_worktree_state()
    source = source_loader(request)
    if (
        _value(source, "target_calls") != 0
        or _value(source, "target_evaluation_performed") is not False
    ):
        raise ValueError("Phase 2 source evidence contains target calls")
    source_dataset_version = _value(source, "dataset_version")
    source_runtime = (
        source_dataset_version.split(";", 1)[0]
        if isinstance(source_dataset_version, str)
        else None
    )
    if dataset_version != source_runtime and not (
        dataset_version == "in-memory" and source_dataset_version == "in-memory"
    ):
        raise ValueError("runtime dataset version differs from Phase 2")
    expected_content = _value(source, "dataset_content_sha256")
    if expected_content is not None and dataset_content_sha256 != expected_content:
        raise ValueError("CIFAR content digest differs from Phase 2")
    folds = _value(source, "folds")
    if not isinstance(folds, (tuple, list)) or len(folds) != len(FOLDS):
        raise ValueError("Phase 2 calibration source grid is incomplete")

    fold_summaries: list[Mapping[str, object]] = []
    rows: list[object] = []
    traces: list[Mapping[str, object]] = []
    status = "complete"
    for fold in folds:
        if code_digest() != calibration_code_digest:
            raise RuntimeError("calibration code changed during evaluation")
        if clock() >= deadline:
            status = "deadline_reached"
            break
        try:
            summary, fold_rows, fold_traces = fold_evaluator(
                fold=fold,
                request=request,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                absolute_deadline=deadline,
                clock=clock,
                progress=report,
            )
        except CalibrationDeadlineReached as error:
            partial_fold = getattr(error, "partial_fold", None)
            if partial_fold is not None:
                fold_summaries.append(dict(partial_fold))
                rows.extend(getattr(error, "rows", ()))
                traces.extend(dict(trace) for trace in getattr(error, "traces", ()))
            status = "deadline_reached"
            break
        if (
            summary.get("target_calls") != 0
            or summary.get("target_evaluation_performed") is not False
            or summary.get("complete") is not True
            or summary.get("raw_evidence_audited") is not True
        ):
            raise ValueError(
                "calibration fold is incomplete, unaudited, or contains target calls"
            )
        fold_summaries.append(dict(summary))
        rows.extend(fold_rows)
        traces.extend(dict(trace) for trace in fold_traces)

    decision = select_calibration_temperature(fold_summaries)
    if code_digest() != calibration_code_digest:
        raise RuntimeError("calibration code changed before persistence")
    rows_path = request.output_dir / "calibration_results.jsonl"
    traces_path = request.output_dir / "calibration_query_traces.jsonl"
    rows_digest = write_verified_jsonl(rows_path, rows)
    traces_digest = write_verified_jsonl(traces_path, traces)
    elapsed = max(0.0, clock() - started)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "name": "phase2-frozen-policy-calibration-diagnostic",
        "status": status,
        "diagnostic_only": True,
        "research_valid": False,
        "publication_candidate": False,
        "exploratory_screen": True,
        "training_performed": False,
        "calibration_code_digest": calibration_code_digest,
        "calibration_git_revision": calibration_git_revision,
        "calibration_git_worktree": calibration_worktree,
        "runtime_environment": dict(runtime_environment or {}),
        "request": _request_payload(request),
        "dataset_version": dataset_version,
        "dataset_content_sha256": dataset_content_sha256,
        "source_manifest": portable_artifact_path(
            Path(str(_value(source, "manifest_path"))),
            relative_to=request.source_root,
        ),
        "source_manifest_sha256": _value(source, "manifest_sha256"),
        "fold_summaries": fold_summaries,
        "calibration_decision": decision,
        "completed_folds": sum(
            summary.get("complete") is True for summary in fold_summaries
        ),
        "partial_folds": sum(
            summary.get("complete") is False for summary in fold_summaries
        ),
        "selected_folds": len(folds),
        "source_model_calls": sum(
            int(summary.get("source_model_calls", 0)) for summary in fold_summaries
        ),
        "target_calls": 0,
        "target_evaluation_available": False,
        "target_evaluation_performed": False,
        "results_path": portable_artifact_path(
            rows_path,
            relative_to=request.output_dir,
        ),
        "results_sha256": rows_digest,
        "query_traces_path": portable_artifact_path(
            traces_path,
            relative_to=request.output_dir,
        ),
        "query_traces_sha256": traces_digest,
        "elapsed_seconds": elapsed,
        "deadline_seconds": request.deadline_seconds,
        "deadline_reached": status == "deadline_reached",
    }
    write_verified_json(
        request.output_dir / "calibration_manifest.json",
        manifest,
    )
    return manifest


__all__ = (
    "CALIBRATION_MAX_SECONDS",
    "CALIBRATION_POLICY_SEEDS",
    "CALIBRATION_TEMPERATURES",
    "FOLDS",
    "Phase2CalibrationRequest",
    "build_calibration_dry_run",
    "load_phase2_calibration_source",
    "run_calibration_from_datasets",
    "select_calibration_temperature",
)
