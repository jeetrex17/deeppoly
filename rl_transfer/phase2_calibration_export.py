"""Deterministic Git-safe export for the Phase 2 calibration diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .paths import resolve_descendant
from .phase1_export_validation import (
    digest,
    finite_number,
    nonnegative_integer,
    require_mapping,
    require_sequence,
    validate_portable_value,
    validated_output_directory,
)
from .phase2_calibration_manifest import (
    CALIBRATION_TEMPERATURES,
    FOLDS,
)
from .phase2_calibration_export_archive import raw_calibration_archive
from .phase2_calibration_export_contract import (
    validate_calibration_manifest_contract,
)
from .phase2_calibration_export_render import (
    evidence_readme,
    fold_asr_figure,
    mean_gain_figure,
    provenance,
)
from .phase2_calibration_export_validation import (
    validate_calibration_evidence,
)
from .phase2_export import (
    _atomic_write,
    _csv_text,
    _formatted_rows,
    _write_checksums,
)
from .phase2_export_validation import read_bounded_regular_file


_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_RESULTS_BYTES = 32 * 1024 * 1024
_MAX_TRACES_BYTES = 8 * 1024 * 1024
_MAX_LOG_BYTES = 16 * 1024 * 1024
_MAX_LOG_TOTAL_BYTES = 32 * 1024 * 1024
_MAX_ATTEMPT_LOGS = 8
_SOURCE_FILES = {
    "calibration_manifest.json",
    "calibration_manifest.json.sha256",
    "calibration_results.jsonl",
    "calibration_results.jsonl.sha256",
    "calibration_query_traces.jsonl",
    "calibration_query_traces.jsonl.sha256",
}
CALIBRATION_EVIDENCE_FILES = {
    "README.md",
    "PROVENANCE.md",
    "SHA256SUMS",
    "summary.json",
    "environment_summary.json",
    "temperature_summary.csv",
    "fold_metrics.csv",
    "condition_metrics.csv",
    "input_checksums.csv",
    "attempt_log_checksums.csv",
    "raw_calibration_records.tar.gz",
    "mean_gain_by_temperature.svg",
    "fold_asr_by_temperature.svg",
}


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _verified_bytes(
    source: Path,
    filename: str,
    *,
    max_bytes: int,
) -> tuple[bytes, str, bytes]:
    path = resolve_descendant(source, filename, label=filename)
    payload = read_bounded_regular_file(
        path,
        label=filename,
        max_bytes=max_bytes,
    )
    sidecar = resolve_descendant(
        source,
        f"{filename}.sha256",
        label=f"{filename} checksum",
    )
    sidecar_payload = read_bounded_regular_file(
        sidecar,
        label=f"{filename} checksum",
        max_bytes=128,
    )
    expected = sidecar_payload.decode("ascii").strip()
    actual = hashlib.sha256(payload).hexdigest()
    if digest(expected, f"{filename} checksum") != actual:
        raise ValueError(f"{filename} checksum failed")
    return payload, actual, sidecar_payload


def _jsonl_records(payload: bytes, label: str) -> list[dict[str, object]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"{label} must contain non-empty JSONL records")
    records: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} row {index} is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{label} row {index} must be an object")
        validate_portable_value(value, f"{label} row {index}")
        records.append(value)
    return records


def _method(temperature: float) -> str:
    return f"temperature_{temperature:.2f}".replace(".", "p")


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    results_sha: str,
    traces_sha: str,
) -> tuple[list[Mapping[str, object]], Mapping[str, object]]:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("diagnostic_only") is not True
        or manifest.get("research_valid") is not False
        or manifest.get("publication_candidate") is not False
        or manifest.get("exploratory_screen") is not True
        or manifest.get("training_performed") is not False
        or manifest.get("target_calls") != 0
        or manifest.get("target_evaluation_performed") is not False
        or manifest.get("target_evaluation_available") is not False
        or manifest.get("deadline_reached") is not False
        or manifest.get("completed_folds") != len(FOLDS)
        or manifest.get("selected_folds") != len(FOLDS)
        or manifest.get("results_sha256") != results_sha
        or manifest.get("query_traces_sha256") != traces_sha
    ):
        raise ValueError(
            "calibration manifest is incomplete, invalid, or has target calls"
        )
    raw_folds = require_sequence(
        manifest.get("fold_summaries"),
        "calibration fold summaries",
    )
    folds = [require_mapping(item, "calibration fold") for item in raw_folds]
    names = tuple(fold.get("heldout_family") for fold in folds)
    if (
        len(folds) != len(FOLDS)
        or set(names) != set(FOLDS)
        or any(
            fold.get("complete") is not True
            or fold.get("raw_evidence_audited") is not True
            or fold.get("temperature_one_reproduced") is not True
            or fold.get("target_calls") != 0
            or fold.get("target_evaluation_performed") is not False
            for fold in folds
        )
    ):
        raise ValueError("calibration fold summaries failed integrity validation")
    decision = require_mapping(
        manifest.get("calibration_decision"),
        "calibration decision",
    )
    if (
        decision.get("complete") is not True
        or decision.get("diagnostic_only") is not True
        or decision.get("authorizes_hidden_target_evaluation") is not False
        or not isinstance(decision.get("calibration_useful"), bool)
        or not isinstance(decision.get("stop_temperature_only_work"), bool)
    ):
        raise ValueError("calibration decision is invalid or authorizes target access")
    return folds, decision


def _fold_rows(
    folds: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in folds:
        heldout = str(fold["heldout_family"])
        score = require_mapping(fold.get("score_greedy"), "score metrics")
        score_asr = finite_number(score.get("asr"), "score ASR")
        score_auc = finite_number(score.get("auc"), "score AUC")
        temperatures = require_mapping(
            fold.get("temperatures"),
            "fold temperatures",
        )
        rows.append(
            {
                "heldout_family": heldout,
                "method": "score_greedy",
                "temperature": "",
                "asr": score_asr,
                "auc": score_auc,
                "asr_gain_vs_score": 0.0,
                "auc_gain_vs_score": 0.0,
                "frozen": True,
            }
        )
        for temperature in CALIBRATION_TEMPERATURES:
            metrics = require_mapping(
                temperatures.get(str(temperature)),
                f"temperature {temperature}",
            )
            asr = finite_number(metrics.get("asr"), "temperature ASR")
            auc = finite_number(metrics.get("auc"), "temperature AUC")
            rows.append(
                {
                    "heldout_family": heldout,
                    "method": _method(temperature),
                    "temperature": temperature,
                    "asr": asr,
                    "auc": auc,
                    "asr_gain_vs_score": asr - score_asr,
                    "auc_gain_vs_score": auc - score_auc,
                    "frozen": metrics.get("frozen") is True,
                }
            )
    return rows


def _temperature_rows(
    decision: Mapping[str, object],
) -> list[dict[str, object]]:
    aggregates = require_mapping(
        decision.get("aggregates"),
        "calibration aggregates",
    )
    qualifying = set(
        require_sequence(
            decision.get("qualifying_temperatures"),
            "qualifying temperatures",
        )
    )
    rows = []
    for temperature in CALIBRATION_TEMPERATURES:
        metrics = require_mapping(
            aggregates.get(str(temperature)),
            f"temperature {temperature} aggregate",
        )
        fold_metrics = [
            require_mapping(item, "temperature fold metrics")
            for item in require_sequence(
                metrics.get("folds"),
                "temperature folds",
            )
        ]
        if len(fold_metrics) != len(FOLDS):
            raise ValueError("temperature aggregate does not contain all folds")
        asr_gains = [
            finite_number(item.get("asr_gain_vs_score"), "fold ASR gain")
            for item in fold_metrics
        ]
        auc_gains = [
            finite_number(item.get("auc_gain_vs_score"), "fold AUC gain")
            for item in fold_metrics
        ]
        rows.append(
            {
                "temperature": temperature,
                "temperature_label": f"T={temperature:g}",
                "mean_asr_gain": finite_number(
                    metrics.get("mean_asr_gain_vs_score"),
                    "mean ASR gain",
                ),
                "mean_auc_gain": finite_number(
                    metrics.get("mean_auc_gain_vs_score"),
                    "mean AUC gain",
                ),
                "minimum_fold_asr_gain": min(asr_gains),
                "minimum_fold_auc_gain": min(auc_gains),
                "folds_observed_nonnegative_both": sum(
                    asr >= -1e-12 and auc >= -1e-12
                    for asr, auc in zip(asr_gains, auc_gains)
                ),
                "qualifies": temperature in qualifying,
            }
        )
    return rows


def _condition_rows(
    folds: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for fold in folds:
        heldout = str(fold["heldout_family"])
        conditions = require_mapping(fold.get("conditions"), "fold conditions")
        if len(conditions) != 4:
            raise ValueError("each calibration fold requires four source conditions")
        for condition in conditions.values():
            item = require_mapping(condition, "source condition")
            source_slice = str(item.get("source_slice"))
            family = str(item.get("family"))
            victims = require_sequence(item.get("victim_ids"), "condition victims")
            score = require_mapping(item.get("score_greedy"), "condition score")
            eligible = nonnegative_integer(score.get("eligible"), "score eligible")
            successes = nonnegative_integer(score.get("successes"), "score successes")
            if eligible <= 0 or successes > eligible or family == heldout:
                raise ValueError("condition score metrics are invalid")
            score_asr = successes / eligible
            score_auc = finite_number(score.get("asr_query_auc"), "condition score AUC")
            base = {
                "heldout_family": heldout,
                "source_slice": source_slice,
                "source_family": family,
                "victim_count": len(victims),
            }
            rows.append(
                {
                    **base,
                    "method": "score_greedy",
                    "temperature": "",
                    "eligible": eligible,
                    "successes": successes,
                    "asr": score_asr,
                    "auc": score_auc,
                    "asr_gain_vs_score": 0.0,
                    "auc_gain_vs_score": 0.0,
                    "frozen": True,
                }
            )
            temperatures = require_mapping(
                item.get("temperatures"),
                "condition temperatures",
            )
            for temperature in CALIBRATION_TEMPERATURES:
                metrics = require_mapping(
                    temperatures.get(str(temperature)),
                    f"condition temperature {temperature}",
                )
                rows.append(
                    {
                        **base,
                        "method": _method(temperature),
                        "temperature": temperature,
                        "eligible": nonnegative_integer(
                            metrics.get("eligible"),
                            "temperature eligible",
                        ),
                        "successes": nonnegative_integer(
                            metrics.get("successes"),
                            "temperature successes",
                        ),
                        "asr": finite_number(metrics.get("asr"), "temperature ASR"),
                        "auc": finite_number(metrics.get("auc"), "temperature AUC"),
                        "asr_gain_vs_score": finite_number(
                            metrics.get("asr_gain_vs_score"),
                            "temperature ASR gain",
                        ),
                        "auc_gain_vs_score": finite_number(
                            metrics.get("auc_gain_vs_score"),
                            "temperature AUC gain",
                        ),
                        "frozen": metrics.get("frozen") is True,
                    }
                )
    return rows


def _augment_temperature_rows(
    temperature_rows: Sequence[Mapping[str, object]],
    fold_rows: Sequence[Mapping[str, object]],
    condition_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    augmented = []
    for row in temperature_rows:
        temperature = float(row["temperature"])
        method = _method(temperature)
        fold_metrics = [item for item in fold_rows if item["method"] == method]
        condition_metrics = [
            item for item in condition_rows if item["method"] == method
        ]
        if len(fold_metrics) != len(FOLDS) or len(condition_metrics) != 12:
            raise ValueError("temperature export metrics are incomplete")
        augmented.append(
            {
                **row,
                "macro_asr": sum(float(item["asr"]) for item in fold_metrics)
                / len(fold_metrics),
                "macro_auc": sum(float(item["auc"]) for item in fold_metrics)
                / len(fold_metrics),
                "conditions_observed_nonnegative_both": sum(
                    float(item["asr_gain_vs_score"]) >= -1e-12
                    and float(item["auc_gain_vs_score"]) >= -1e-12
                    for item in condition_metrics
                ),
            }
        )
    return augmented


def _summary(
    manifest: Mapping[str, object],
    decision: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
    temperatures: Sequence[Mapping[str, object]],
    fold_metrics: Sequence[Mapping[str, object]],
    *,
    manifest_sha: str,
    results_sha: str,
    traces_sha: str,
    result_rows: int,
    query_traces: int,
) -> dict[str, object]:
    best_asr = max(temperatures, key=lambda row: float(row["mean_asr_gain"]))
    best_auc = max(temperatures, key=lambda row: float(row["mean_auc_gain"]))
    score_metrics = [row for row in fold_metrics if row.get("method") == "score_greedy"]
    if len(score_metrics) != len(FOLDS):
        raise ValueError("score-greedy fold metrics are incomplete")
    runtime = require_mapping(
        manifest.get("runtime_environment"),
        "runtime environment",
    )
    summary = {
        "schema_version": 1,
        "name": "cifar10-rtx-phase2-calibration-evidence",
        "status": manifest["status"],
        "scientific_status": "exploratory_source_only_diagnostic",
        "decision": {
            "scope": (
                "five_value_global_sampling_temperature_grid_on_frozen_"
                "seed_17_checkpoints"
            ),
            "tested_global_temperature_qualified": decision["calibration_useful"],
            "stop_tested_global_temperature_protocol": decision[
                "stop_temperature_only_work"
            ],
            "selected_temperature": decision.get("selected_temperature"),
            "qualifying_temperatures": list(
                require_sequence(
                    decision.get("qualifying_temperatures"),
                    "qualifying temperatures",
                )
            ),
            "reason": decision.get("reason"),
            "best_mean_asr_temperature": best_asr["temperature"],
            "best_mean_asr_gain": best_asr["mean_asr_gain"],
            "best_mean_auc_temperature": best_auc["temperature"],
            "best_mean_auc_gain": best_auc["mean_auc_gain"],
        },
        "score_greedy": {
            "macro_asr": sum(float(row["asr"]) for row in score_metrics)
            / len(score_metrics),
            "macro_auc": sum(float(row["auc"]) for row in score_metrics)
            / len(score_metrics),
        },
        "temperature_summary": list(temperatures),
        "folds": [
            {
                "heldout_family": fold["heldout_family"],
                "temperature_one_reproduced": fold["temperature_one_reproduced"],
                "raw_evidence_audited": fold["raw_evidence_audited"],
            }
            for fold in folds
        ],
        "runtime": {
            "elapsed_seconds": manifest["elapsed_seconds"],
            "elapsed_minutes": float(manifest["elapsed_seconds"]) / 60,
            "deadline_seconds": manifest["deadline_seconds"],
            "deadline_reached": manifest["deadline_reached"],
            "source_model_calls": manifest["source_model_calls"],
            "gpu_name": runtime.get("gpu_name"),
        },
        "target_evaluation": {
            "target_calls": 0,
            "target_evaluation_performed": False,
            "target_evaluation_authorized": False,
            "training_performed": False,
        },
        "integrity": {
            "manifest_sha256": manifest_sha,
            "results_sha256": results_sha,
            "query_traces_sha256": traces_sha,
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "calibration_code_digest": manifest["calibration_code_digest"],
            "calibration_git_revision": manifest["calibration_git_revision"],
            "result_rows": result_rows,
            "query_traces": query_traces,
            "completed_folds": len(folds),
            "temperature_one_reproduced_folds": sum(
                fold.get("temperature_one_reproduced") is True for fold in folds
            ),
        },
        "limitations": [
            "Only one development policy seed was evaluated.",
            "Each image-victim-temperature combination had one stochastic replay.",
            (
                "The same 100 source images and overlapping source victims were "
                "reused, so the 9,000 rows are dependent observations."
            ),
            (
                "Five temperatures were selected and evaluated on the same "
                "visible source cohort."
            ),
            (
                "Successes were rare, observed effects were small, and no "
                "confidence intervals were estimated; results are descriptive "
                "point estimates."
            ),
            (
                "Only visible source victims were evaluated, so this is not a "
                "transfer result."
            ),
            "No hidden target evaluation was authorized or performed.",
        ],
    }
    validate_portable_value(summary, "calibration evidence summary")
    return summary


def _input_checksums(
    payloads: Mapping[str, bytes],
) -> list[dict[str, object]]:
    return [
        {
            "filename": name,
            "bytes": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
        for name in sorted(payloads)
    ]


def _attempt_log_rows(
    attempt_logs: Sequence[Path],
) -> list[dict[str, object]]:
    if len(attempt_logs) > _MAX_ATTEMPT_LOGS:
        raise ValueError("too many calibration attempt logs")
    rows = []
    seen = set()
    total_bytes = 0
    for raw_path in attempt_logs:
        path = Path(raw_path)
        if path.name in seen:
            raise ValueError("attempt log names must be unique")
        seen.add(path.name)
        payload = read_bounded_regular_file(
            path,
            label=f"attempt log {path.name}",
            max_bytes=_MAX_LOG_BYTES,
        )
        total_bytes += len(payload)
        if total_bytes > _MAX_LOG_TOTAL_BYTES:
            raise ValueError("calibration attempt logs exceed the aggregate limit")
        rows.append(
            {
                "filename": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    validate_portable_value(rows, "calibration attempt logs")
    return rows


def _write_table(
    output: Path,
    filename: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        raise ValueError(f"{filename} cannot be empty")
    _write_text(
        output / filename,
        _csv_text(tuple(rows[0]), _formatted_rows(rows)),
    )


def export_phase2_calibration_evidence(
    source_root: Path,
    output_dir: Path,
    *,
    attempt_logs: Sequence[Path] = (),
) -> dict[str, object]:
    """Verify the completed D0 archive and write portable evidence."""

    source = Path(source_root).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("calibration source root must be a directory")
    entries = {path.name for path in source.iterdir()}
    if entries != _SOURCE_FILES or any(
        path.is_symlink() or not path.is_file() for path in source.iterdir()
    ):
        raise ValueError("calibration source file set is incomplete or unmanaged")

    manifest_bytes, manifest_sha, manifest_sidecar = _verified_bytes(
        source,
        "calibration_manifest.json",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    results_bytes, results_sha, results_sidecar = _verified_bytes(
        source,
        "calibration_results.jsonl",
        max_bytes=_MAX_RESULTS_BYTES,
    )
    traces_bytes, traces_sha, traces_sidecar = _verified_bytes(
        source,
        "calibration_query_traces.jsonl",
        max_bytes=_MAX_TRACES_BYTES,
    )
    source_payloads = {
        "calibration_manifest.json": manifest_bytes,
        "calibration_manifest.json.sha256": manifest_sidecar,
        "calibration_results.jsonl": results_bytes,
        "calibration_results.jsonl.sha256": results_sidecar,
        "calibration_query_traces.jsonl": traces_bytes,
        "calibration_query_traces.jsonl.sha256": traces_sidecar,
    }
    try:
        manifest_value = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("calibration manifest is invalid JSON") from error
    manifest = require_mapping(manifest_value, "calibration manifest")
    validate_portable_value(manifest, "calibration manifest")
    validate_calibration_manifest_contract(manifest)
    results = _jsonl_records(results_bytes, "calibration results")
    traces = _jsonl_records(traces_bytes, "calibration query traces")
    folds, decision = _validate_manifest(
        manifest,
        results_sha=results_sha,
        traces_sha=traces_sha,
    )
    source_model_calls = validate_calibration_evidence(
        folds,
        decision,
        results,
        traces,
    )
    if manifest.get("source_model_calls") != source_model_calls:
        raise ValueError("calibration source-model call count is inconsistent")
    fold_rows = _fold_rows(folds)
    condition_rows = _condition_rows(folds)
    temperature_rows = _augment_temperature_rows(
        _temperature_rows(decision),
        fold_rows,
        condition_rows,
    )
    summary = _summary(
        manifest,
        decision,
        folds,
        temperature_rows,
        fold_rows,
        manifest_sha=manifest_sha,
        results_sha=results_sha,
        traces_sha=traces_sha,
        result_rows=len(results),
        query_traces=len(traces),
    )
    environment = dict(
        require_mapping(
            manifest.get("runtime_environment"),
            "runtime environment",
        )
    )
    validate_portable_value(environment, "runtime environment")
    log_rows = _attempt_log_rows(tuple(attempt_logs))
    output = validated_output_directory(source, Path(output_dir))
    unexpected = {
        path.name
        for path in output.iterdir()
        if path.name not in CALIBRATION_EVIDENCE_FILES
    }
    if unexpected:
        raise ValueError(
            "evidence output contains unmanaged entries: "
            + ", ".join(sorted(unexpected))
        )

    _write_text(output / "README.md", evidence_readme(summary))
    _write_text(output / "PROVENANCE.md", provenance(summary))
    _atomic_write(output / "summary.json", _json_bytes(summary))
    _atomic_write(
        output / "environment_summary.json",
        _json_bytes(environment),
    )
    _write_table(
        output,
        "temperature_summary.csv",
        temperature_rows,
    )
    _write_table(output, "fold_metrics.csv", fold_rows)
    _write_table(output, "condition_metrics.csv", condition_rows)
    _write_table(
        output,
        "input_checksums.csv",
        _input_checksums(source_payloads),
    )
    _write_text(
        output / "attempt_log_checksums.csv",
        _csv_text(
            ("filename", "bytes", "sha256"),
            _formatted_rows(log_rows),
        ),
    )
    _atomic_write(
        output / "raw_calibration_records.tar.gz",
        raw_calibration_archive(source_payloads),
    )
    _write_text(
        output / "mean_gain_by_temperature.svg",
        mean_gain_figure(temperature_rows),
    )
    _write_text(
        output / "fold_asr_by_temperature.svg",
        fold_asr_figure(folds),
    )
    _write_checksums(output)
    final_entries = list(output.iterdir())
    if {path.name for path in final_entries} != CALIBRATION_EVIDENCE_FILES or any(
        path.is_symlink() or not path.is_file() for path in final_entries
    ):
        raise RuntimeError("calibration evidence output set is incomplete")
    return summary


__all__ = (
    "CALIBRATION_EVIDENCE_FILES",
    "export_phase2_calibration_evidence",
)
