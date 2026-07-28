"""Deterministic, portable evidence export for the RTX Phase 1 study."""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

from .artifacts import sha256_file
from .paths import resolve_descendant
from .phase1_environment import verified_environment_evidence
from .phase1_export_archive import write_raw_source_archive
from .phase1_export_render import (
    MANAGED_OUTPUT_NAMES as _MANAGED_OUTPUT_NAMES,
    METHOD_ORDER as _METHOD_ORDER,
    bc_figure as _bc_figure,
    family_figure as _family_figure,
    formatted_condition_rows as _formatted_condition_rows,
    formatted_rows as _formatted_rows,
    method_figure as _method_figure,
    provenance as _provenance,
    readme as _readme,
    runtime_figure as _runtime_figure,
)
from .phase1_export_validation import (
    compact_raw_runs as _compact_raw_runs,
    digest as _digest,
    finite_number as _finite_number,
    nonnegative_integer as _nonnegative_integer,
    require_mapping as _require_mapping,
    validate_portable_value as _portable_value,
    validated_output_directory as _validated_output_directory,
    verified_source_runs as _verified_source_runs,
)
from .verified_artifacts import load_verified_json


_LEARNED_METHOD = "gradient_bc_groupdro_ppo_stochastic"
_DEFAULT_CONTROL = "score_greedy"
_MAX_GZIP_BYTES = 25 * 1024 * 1024


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))


def _json_bytes(value: object, *, compact: bool = False) -> bytes:
    options: dict[str, object] = {
        "sort_keys": True,
        "allow_nan": False,
    }
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _csv_text(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _condition_rows(
    verified_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in verified_runs:
        run = _require_mapping(item.get("run"), "verified run")
        evaluation = _require_mapping(
            run.get("source_evaluation"),
            "source_evaluation",
        )
        for slice_name, families_value in evaluation.items():
            families = _require_mapping(
                families_value,
                f"{slice_name} families",
            )
            for family, methods_value in families.items():
                methods = _require_mapping(
                    methods_value,
                    f"{slice_name}/{family} methods",
                )
                for method, metrics_value in methods.items():
                    metrics = _require_mapping(
                        metrics_value,
                        f"{slice_name}/{family}/{method}",
                    )
                    eligible = _nonnegative_integer(
                        metrics.get("eligible"),
                        "eligible",
                    )
                    successes = _nonnegative_integer(
                        metrics.get("successes"),
                        "successes",
                    )
                    if successes > eligible or eligible == 0:
                        raise ValueError("invalid eligible/success count")
                    rows.append(
                        {
                            "fingerprint": str(run["fingerprint"]),
                            "seed": int(run["seed"]),
                            "omitted_target_family": str(run["target_family"]),
                            "source_slice": str(slice_name),
                            "evaluated_source_family": str(family),
                            "method": str(method),
                            "eligible": eligible,
                            "successes": successes,
                            "asr": successes / eligible,
                            "asr_query_auc": _finite_number(
                                metrics.get("asr_query_auc"),
                                "asr_query_auc",
                            ),
                            "normalized_action_entropy": _finite_number(
                                metrics.get(
                                    "normalized_action_entropy",
                                    0.0,
                                ),
                                "normalized_action_entropy",
                            ),
                            "query_budget": _nonnegative_integer(
                                metrics.get("query_budget"),
                                "query_budget",
                            ),
                        }
                    )
    rows.sort(
        key=lambda row: (
            row["omitted_target_family"],
            row["seed"],
            row["source_slice"],
            row["evaluated_source_family"],
            row["method"],
        )
    )
    return rows


def _weighted_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    eligible = sum(int(row["eligible"]) for row in rows)
    successes = sum(int(row["successes"]) for row in rows)
    if eligible <= 0:
        raise ValueError("cannot summarize an empty eligible cohort")
    return {
        "conditions": len(rows),
        "eligible": eligible,
        "successes": successes,
        "asr": sum(float(row["asr"]) for row in rows) / len(rows),
        "pooled_asr": successes / eligible,
        "asr_query_auc": sum(float(row["asr_query_auc"]) for row in rows) / len(rows),
        "eligible_weighted_asr_query_auc": sum(
            float(row["asr_query_auc"]) * int(row["eligible"]) for row in rows
        )
        / eligible,
        "normalized_action_entropy": sum(
            float(row["normalized_action_entropy"]) for row in rows
        )
        / len(rows),
        "eligible_weighted_action_entropy": sum(
            float(row["normalized_action_entropy"]) * int(row["eligible"])
            for row in rows
        )
        / eligible,
    }


def _method_summaries(
    condition_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    keys = sorted(
        {(str(row["source_slice"]), str(row["method"])) for row in condition_rows},
        key=lambda key: (
            key[0],
            _METHOD_ORDER.index(key[1])
            if key[1] in _METHOD_ORDER
            else len(_METHOD_ORDER),
            key[1],
        ),
    )
    summaries = [
        {
            "source_slice": source_slice,
            "method": method,
            **_weighted_summary(
                [
                    row
                    for row in condition_rows
                    if row["source_slice"] == source_slice and row["method"] == method
                ]
            ),
        }
        for source_slice, method in keys
    ]
    for method in sorted(
        {str(row["method"]) for row in condition_rows},
        key=lambda value: (
            _METHOD_ORDER.index(value)
            if value in _METHOD_ORDER
            else len(_METHOD_ORDER),
            value,
        ),
    ):
        summaries.append(
            {
                "source_slice": "combined_source",
                "method": method,
                **_weighted_summary(
                    [row for row in condition_rows if row["method"] == method]
                ),
            }
        )
    return summaries


def _method_for_run(
    rows: Sequence[Mapping[str, object]],
    method: str,
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["source_slice"] == "exact_source" and row["method"] == method
    ]
    if not selected:
        raise ValueError(f"exact_source method is missing: {method}")
    return _weighted_summary(selected)


def _bc_diagnostics(run: Mapping[str, object]) -> dict[str, float | bool]:
    policy = _require_mapping(run.get("policy"), "policy")
    training = _require_mapping(policy.get("training"), "policy training")
    bc = _require_mapping(training.get("behavior_cloning"), "behavior cloning")
    fit = _require_mapping(bc.get("fit"), "behavior cloning fit")
    validation = _require_mapping(
        bc.get("validation"),
        "behavior cloning validation",
    )
    gate = _require_mapping(bc.get("gate"), "behavior cloning gate")
    return {
        "train_accuracy": _finite_number(
            fit.get("final_accuracy"),
            "BC train accuracy",
        ),
        "validation_accuracy": _finite_number(
            validation.get("accuracy"),
            "BC validation accuracy",
        ),
        "validation_majority_accuracy": _finite_number(
            validation.get("majority_accuracy"),
            "BC validation majority accuracy",
        ),
        "uniform_accuracy": _finite_number(
            validation.get("uniform_accuracy"),
            "BC uniform accuracy",
        ),
        "validation_nll": _finite_number(
            validation.get("nll"),
            "BC validation NLL",
        ),
        "validation_frequency_nll": _finite_number(
            validation.get("frequency_nll"),
            "BC validation frequency NLL",
        ),
        "accepted_steps": _finite_number(
            fit.get("accepted_steps"),
            "BC accepted steps",
        ),
        "elapsed_seconds": _finite_number(
            bc.get("elapsed_seconds"),
            "BC elapsed seconds",
        ),
        "gate_passed": gate.get("passed") is True,
    }


def _mean(records: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records)


def _run_rows(
    verified_runs: Sequence[Mapping[str, object]],
    conditions: Sequence[Mapping[str, object]],
    control: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    bc_records: list[dict[str, object]] = []
    for item in verified_runs:
        run = _require_mapping(item.get("run"), "verified run")
        fingerprint = str(run["fingerprint"])
        selected = [row for row in conditions if row["fingerprint"] == fingerprint]
        learned = _method_for_run(selected, _LEARNED_METHOD)
        baseline = _method_for_run(selected, control)
        bc = _bc_diagnostics(run)
        bc_records.append({"fingerprint": fingerprint, **bc})
        training = _require_mapping(
            _require_mapping(run.get("policy"), "policy").get("training"),
            "policy training",
        )
        rows.append(
            {
                "fingerprint": fingerprint,
                "seed": int(run["seed"]),
                "omitted_target_family": str(run["target_family"]),
                "status": str(run["status"]),
                "exact_source_asr": learned["asr"],
                "exact_source_auc": learned["asr_query_auc"],
                f"{control}_asr": baseline["asr"],
                f"{control}_auc": baseline["asr_query_auc"],
                f"asr_gain_vs_{control}": float(learned["asr"])
                - float(baseline["asr"]),
                f"auc_gain_vs_{control}": float(learned["asr_query_auc"])
                - float(baseline["asr_query_auc"]),
                "bc_train_accuracy": bc["train_accuracy"],
                "bc_validation_accuracy": bc["validation_accuracy"],
                "bc_majority_accuracy": bc["validation_majority_accuracy"],
                "bc_validation_nll": bc["validation_nll"],
                "bc_frequency_nll": bc["validation_frequency_nll"],
                "bc_gate_passed": bc["gate_passed"],
                "trained_episodes": _nonnegative_integer(
                    training.get("trained_episodes"),
                    "trained episodes",
                ),
                "source_calls": _nonnegative_integer(
                    training.get("source_calls"),
                    "source calls",
                ),
                "elapsed_seconds": _finite_number(
                    run.get("elapsed_seconds"),
                    "run elapsed seconds",
                ),
                "source_evaluation_elapsed_seconds": _finite_number(
                    run.get("source_evaluation_elapsed_seconds"),
                    "source evaluation elapsed seconds",
                ),
            }
        )
    return rows, bc_records


def _family_summaries(
    conditions: Sequence[Mapping[str, object]],
    control: str,
) -> list[dict[str, object]]:
    families = sorted({str(row["omitted_target_family"]) for row in conditions})
    summaries: list[dict[str, object]] = []
    for family in families:
        rows = [row for row in conditions if row["omitted_target_family"] == family]
        learned = _weighted_summary(
            [row for row in rows if row["method"] == _LEARNED_METHOD]
        )
        baseline = _weighted_summary([row for row in rows if row["method"] == control])
        summaries.append(
            {
                "omitted_target_family": family,
                "learned_asr": learned["asr"],
                "learned_auc": learned["asr_query_auc"],
                "control_asr": baseline["asr"],
                "control_auc": baseline["asr_query_auc"],
                "asr_gain": float(learned["asr"]) - float(baseline["asr"]),
                "auc_gain": float(learned["asr_query_auc"])
                - float(baseline["asr_query_auc"]),
            }
        )
    return summaries


def _comparison_counts(
    conditions: Sequence[Mapping[str, object]],
    control: str,
) -> dict[str, int]:
    keys = sorted(
        {
            (
                str(row["fingerprint"]),
                str(row["source_slice"]),
                str(row["evaluated_source_family"]),
            )
            for row in conditions
            if row["method"] == _LEARNED_METHOD
        }
    )
    counts = {"wins": 0, "ties": 0, "losses": 0}
    for fingerprint, source_slice, family in keys:
        cell = [
            row
            for row in conditions
            if row["fingerprint"] == fingerprint
            and row["source_slice"] == source_slice
            and row["evaluated_source_family"] == family
        ]
        learned = _method_for_cell(cell, _LEARNED_METHOD)
        baseline = _method_for_cell(cell, control)
        difference = float(learned["asr"]) - float(baseline["asr"])
        if difference > 1e-12:
            counts["wins"] += 1
        elif difference < -1e-12:
            counts["losses"] += 1
        else:
            counts["ties"] += 1
    return counts


def _method_for_cell(
    rows: Sequence[Mapping[str, object]],
    method: str,
) -> Mapping[str, object]:
    selected = [row for row in rows if row["method"] == method]
    if len(selected) != 1:
        raise ValueError(f"cell must contain exactly one {method} record")
    return selected[0]


def _summary(
    study: Mapping[str, object],
    study_manifest_sha: str,
    verified_runs: Sequence[Mapping[str, object]],
    condition_rows: Sequence[Mapping[str, object]],
    method_summaries: Sequence[Mapping[str, object]],
    run_rows: Sequence[Mapping[str, object]],
    bc_records: Sequence[Mapping[str, object]],
    family_summaries: Sequence[Mapping[str, object]],
    control: str,
) -> dict[str, object]:
    exact = {
        str(record["method"]): {
            key: record[key]
            for key in (
                "conditions",
                "eligible",
                "successes",
                "asr",
                "pooled_asr",
                "asr_query_auc",
                "eligible_weighted_asr_query_auc",
                "normalized_action_entropy",
                "eligible_weighted_action_entropy",
            )
        }
        for record in method_summaries
        if record["source_slice"] == "exact_source"
    }
    new_instance = {
        str(record["method"]): {
            key: record[key]
            for key in (
                "conditions",
                "eligible",
                "successes",
                "asr",
                "pooled_asr",
                "asr_query_auc",
                "eligible_weighted_asr_query_auc",
                "normalized_action_entropy",
                "eligible_weighted_action_entropy",
            )
        }
        for record in method_summaries
        if record["source_slice"] == "seen_family_new_instance"
    }
    combined = {
        str(record["method"]): {
            key: record[key]
            for key in (
                "conditions",
                "eligible",
                "successes",
                "asr",
                "pooled_asr",
                "asr_query_auc",
                "eligible_weighted_asr_query_auc",
                "normalized_action_entropy",
                "eligible_weighted_action_entropy",
            )
        }
        for record in method_summaries
        if record["source_slice"] == "combined_source"
    }
    gate = _require_mapping(
        study.get("source_competence_gate"),
        "source competence gate",
    )
    runtime_environment = _require_mapping(
        study.get("runtime_environment"),
        "runtime environment",
    )
    environment = {
        key: runtime_environment[key]
        for key in (
            "cuda_device_name",
            "cuda_total_memory_bytes",
            "cuda_runtime",
            "cudnn_version",
            "git_revision",
            "python",
            "torch",
            "platform",
            "determinism",
        )
        if key in runtime_environment
    }
    result = {
        "schema_version": 1,
        "study_name": str(study["name"]),
        "study_schema_version": int(study["schema_version"]),
        "status": str(study["status"]),
        "research_valid": study.get("research_valid") is True,
        "publication_candidate": study.get("publication_candidate") is True,
        "target_evaluation": {
            "target_calls": int(study["target_calls"]),
            "held_out_attack_evaluation_calls": int(study["target_calls"]),
            "target_evaluation_performed": False,
            "hidden_target_attack_cohort_opened": False,
            "all_family_victim_clean_validation_performed": True,
        },
        "protocol": {
            "design": "leave-one-family-out source-attack Phase 1",
            "policy_runs": len(verified_runs),
            "primary_control": control,
            "query_budget_including_initialization": 50,
            "source_gate_passed": gate.get("passed") is True,
            "grid_complete": gate.get("grid_complete") is True,
            "completed_runs": int(gate.get("completed_runs", 0)),
            "expected_runs": int(gate.get("expected_runs", 0)),
        },
        "outcomes": {
            "exact_source": exact,
            "seen_family_new_instance": new_instance,
            "combined_source": combined,
            "lofo_split_combined_source": list(family_summaries),
            f"hybrid_vs_{control}_cell_counts": _comparison_counts(
                condition_rows,
                control,
            ),
            f"hybrid_vs_{control}_run_mean_asr_gain": _mean(
                run_rows,
                f"asr_gain_vs_{control}",
            ),
            f"hybrid_vs_{control}_run_mean_auc_gain": _mean(
                run_rows,
                f"auc_gain_vs_{control}",
            ),
        },
        "behavior_cloning": {
            "run_count": len(bc_records),
            "gate_pass_count": sum(
                bool(record["gate_passed"]) for record in bc_records
            ),
            "train_accuracy": _mean(bc_records, "train_accuracy"),
            "validation_accuracy": _mean(
                bc_records,
                "validation_accuracy",
            ),
            "validation_majority_accuracy": _mean(
                bc_records,
                "validation_majority_accuracy",
            ),
            "uniform_accuracy": _mean(bc_records, "uniform_accuracy"),
            "validation_nll": _mean(bc_records, "validation_nll"),
            "validation_frequency_nll": _mean(
                bc_records,
                "validation_frequency_nll",
            ),
            "accepted_steps": _mean(bc_records, "accepted_steps"),
        },
        "runtime": {
            "source_phase_seconds": _finite_number(
                study.get("source_phase_elapsed_seconds"),
                "source phase elapsed seconds",
            ),
            "source_phase_hours": _finite_number(
                study.get("source_phase_elapsed_seconds"),
                "source phase elapsed seconds",
            )
            / 3600,
            "mean_run_seconds": _mean(run_rows, "elapsed_seconds"),
            "mean_source_evaluation_seconds": _mean(
                run_rows,
                "source_evaluation_elapsed_seconds",
            ),
        },
        "environment": environment,
        "integrity": {
            "study_manifest_sha256": study_manifest_sha,
            "study_code_digest": _digest(
                study.get("study_code_digest"),
                "study code digest",
            ),
            "verified_runs": len(verified_runs),
            "verified_results_files": len(verified_runs),
            "verified_trace_files": len(verified_runs),
            "all_source_audits_passed": True,
        },
        "interpretation": {
            "source_phase_result": "negative",
            "target_transfer_claim_supported": False,
            "primary_failure": (
                "strict source competence and behavior-cloning gates failed"
            ),
        },
    }
    _portable_value(result, "summary")
    return result


def _write_bundle_checksums(output: Path) -> None:
    paths = sorted(
        (
            path
            for path in output.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        key=lambda path: path.name,
    )
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    _write_text(output / "SHA256SUMS", "\n".join(lines) + "\n")


def export_phase1_evidence(
    source_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Verify Phase 1 artifacts and write a compact Git-safe evidence bundle."""

    study_root = source_root.resolve(strict=True)
    if not study_root.is_dir():
        raise ValueError("source root must be a directory")
    output = _validated_output_directory(study_root, output_dir)
    unexpected_outputs = {
        path.name for path in output.iterdir() if path.name not in _MANAGED_OUTPUT_NAMES
    }
    if unexpected_outputs:
        raise ValueError(
            "evidence output contains unmanaged entries: "
            + ", ".join(sorted(unexpected_outputs))
        )
    study_manifest_path = resolve_descendant(
        study_root,
        "study_manifest.json",
        label="study manifest",
    )
    study = load_verified_json(study_manifest_path)
    if study.get("target_evaluation_performed") is not False:
        raise ValueError("Phase 1 evidence must be target-free")
    if _nonnegative_integer(study.get("target_calls"), "target_calls") != 0:
        raise ValueError("Phase 1 evidence must be target-free")
    if study.get("status") not in {
        "source_complete",
        "source_learning_failed",
    }:
        raise ValueError("study is not a completed Phase 1 source study")

    dependency_freeze, environment_evidence = verified_environment_evidence(
        study_root,
        study,
    )
    verified_runs, input_checksums = _verified_source_runs(study_root, study)
    conditions = _condition_rows(verified_runs)
    method_summaries = _method_summaries(conditions)
    config = _require_mapping(study.get("config"), "study config")
    control_value = config.get("primary_control", _DEFAULT_CONTROL)
    if not isinstance(control_value, str) or not control_value:
        raise ValueError("primary control must be a nonempty string")
    control = control_value
    run_rows, bc_records = _run_rows(
        verified_runs,
        conditions,
        control,
    )
    family_summaries = _family_summaries(conditions, control)
    summary = _summary(
        study,
        sha256_file(study_manifest_path),
        verified_runs,
        conditions,
        method_summaries,
        run_rows,
        bc_records,
        family_summaries,
        control,
    )
    summary["environment_evidence"] = environment_evidence
    _portable_value(summary, "summary")
    compact_raw = {
        "schema_version": 1,
        "study_manifest_sha256": summary["integrity"]["study_manifest_sha256"],
        "study_status": study["status"],
        "target_calls": 0,
        "target_evaluation_performed": False,
        "environment_evidence": environment_evidence,
        "input_checksums": input_checksums,
        "runs": _compact_raw_runs(verified_runs),
    }
    _portable_value(compact_raw, "compact raw evidence")
    compressed_raw = gzip.compress(
        _json_bytes(compact_raw, compact=True),
        compresslevel=9,
        mtime=0,
    )
    if len(compressed_raw) > _MAX_GZIP_BYTES:
        raise ValueError("compact evidence exceeds the Git-safe size limit")

    _write_text(output / "dependency_freeze.txt", dependency_freeze)
    _atomic_write(
        output / "environment_summary.json",
        _json_bytes(environment_evidence),
    )
    _atomic_write(output / "summary.json", _json_bytes(summary))
    _write_text(
        output / "run_summary.csv",
        _csv_text(tuple(run_rows[0]), _formatted_rows(run_rows)),
    )
    _write_text(
        output / "condition_metrics.csv",
        _csv_text(
            tuple(conditions[0]),
            _formatted_condition_rows(conditions),
        ),
    )
    _write_text(
        output / "method_summary.csv",
        _csv_text(
            tuple(method_summaries[0]),
            _formatted_rows(method_summaries),
        ),
    )
    _write_text(
        output / "input_checksums.csv",
        _csv_text(tuple(input_checksums[0]), input_checksums),
    )
    _atomic_write(
        output / "raw_compact_evidence.json.gz",
        compressed_raw,
    )
    write_raw_source_archive(
        verified_runs,
        output / "raw_source_records.tar.gz",
    )
    _write_text(output / "README.md", _readme(summary))
    _write_text(output / "PROVENANCE.md", _provenance(summary))
    _write_text(
        output / "method_performance.svg",
        _method_figure(method_summaries),
    )
    _write_text(
        output / "heldout_family_asr.svg",
        _family_figure(family_summaries),
    )
    _write_text(
        output / "bc_diagnostics.svg",
        _bc_figure(summary["behavior_cloning"]),
    )
    _write_text(
        output / "runtime.svg",
        _runtime_figure(run_rows),
    )
    _write_bundle_checksums(output)
    return summary
