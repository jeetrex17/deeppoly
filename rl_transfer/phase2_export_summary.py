"""Compact summaries and provenance records for Phase 2 evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .phase1_export_validation import (
    digest,
    require_mapping,
    require_sequence,
    validate_portable_value,
)
from .phase2_export_tables import LEARNED_METHOD, mean
from .phase2_export_validation import bounded_regular_file_digest
from .phase2_promotion import SCREEN_CONTROL


_MAX_ATTEMPT_LOGS = 100
_MAX_ATTEMPT_LOG_BYTES = 64 * 1024 * 1024


def attempt_log_rows(source_root: Path) -> list[dict[str, object]]:
    directory = source_root / "attempt_logs"
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("attempt log directory must be a regular directory")
    paths = sorted(directory.glob("*.log"))
    if len(paths) > _MAX_ATTEMPT_LOGS:
        raise ValueError("attempt log count exceeds the export limit")
    rows = []
    for path in paths:
        size = path.stat(follow_symlinks=False).st_size
        checksum = bounded_regular_file_digest(
            path,
            label="attempt log",
            max_bytes=_MAX_ATTEMPT_LOG_BYTES,
        )
        rows.append(
            {
                "filename": path.name,
                "bytes": size,
                "sha256": checksum,
            }
        )
    return rows


def _safe_victims(run: Mapping[str, object]) -> dict[str, object]:
    instances = require_mapping(
        run.get("victim_instances"),
        "victim instances",
    )
    return {
        str(family): [
            {
                key: record[key]
                for key in (
                    "victim_id",
                    "instance_index",
                    "checkpoint_sha256",
                    "source_validation_accuracy",
                    "training_seed",
                    "resumed",
                )
                if key in record
            }
            for value in require_sequence(records, "victim records")
            for record in [require_mapping(value, "victim record")]
        ]
        for family, records in instances.items()
    }


def compact_runs(
    verified_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    result = []
    for item in verified_runs:
        run = require_mapping(item.get("run"), "verified run")
        policy = require_mapping(run.get("policy"), "policy")
        runtime = require_mapping(run.get("runtime"), "runtime")
        result.append(
            {
                key: run[key]
                for key in (
                    "schema_version",
                    "name",
                    "fingerprint",
                    "status",
                    "seed",
                    "target_family",
                    "source_families",
                    "research_valid",
                    "target_calls",
                    "target_evaluation_performed",
                    "validation_roles_disjoint",
                    "config_digest",
                    "split_digest",
                    "data_role_digests",
                    "victim_bank_digest",
                    "victim_cache_digest",
                    "victim_accuracy_gate",
                    "victim_access_audit",
                    "source_competence_gate",
                    "source_evaluation",
                    "source_evaluation_audits",
                )
                if key in run
            }
            | {
                "runtime": {
                    key: runtime[key]
                    for key in (
                        "code_digest",
                        "git_revision",
                        "python",
                        "torch",
                        "cuda_runtime",
                        "cudnn_version",
                        "cuda_device_name",
                        "determinism",
                    )
                    if key in runtime
                },
                "policy_checkpoint_sha256": policy.get(
                    "checkpoint_sha256"
                ),
                "policy_persistent_digest": policy.get(
                    "persistent_digest"
                ),
                "policy_training": policy.get("training"),
                "victim_instances": _safe_victims(run),
            }
        )
    return result


def environment_summary(
    screen: Mapping[str, object],
    verified_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    screen_environment = require_mapping(
        screen.get("runtime_environment"),
        "runtime environment",
    )
    first_run = require_mapping(
        require_mapping(verified_runs[0].get("run"), "run").get(
            "runtime"
        ),
        "run runtime",
    )
    keys = (
        "git_revision",
        "python",
        "torch",
        "cuda_runtime",
        "cudnn_version",
        "cuda_device_name",
        "cuda_total_memory_bytes",
        "determinism",
        "platform",
    )
    result = {
        key: first_run[key]
        for key in keys
        if key in first_run
    }
    result.update(
        {
            key: screen_environment[key]
            for key in (
                "nvidia_driver",
                "pip_freeze_sha256",
                "requirements_sha256",
            )
            if key in screen_environment
        }
    )
    return result


def build_summary(
    *,
    screen: Mapping[str, object],
    screen_manifest_sha: str,
    sidecar_count: int,
    verified_runs: Sequence[Mapping[str, object]],
    input_checksums: Sequence[Mapping[str, object]],
    method_rows: Sequence[Mapping[str, object]],
    fold_rows: Sequence[Mapping[str, object]],
    bc_rows: Sequence[Mapping[str, object]],
    environment: Mapping[str, object],
) -> dict[str, object]:
    decision = dict(
        require_mapping(
            screen.get("screen_promotion_decision"),
            "screen promotion decision",
        )
    )
    if (
        decision.get("grid_complete") is not True
        or decision.get("completed_cells") != len(verified_runs)
        or decision.get("expected_cells") != len(verified_runs)
    ):
        raise ValueError("Phase 2 source grid is incomplete")
    total_seconds = sum(
        float(row["recorded_total_seconds"]) for row in fold_rows
    )
    result = {
        "schema_version": 1,
        "study_name": str(screen["name"]),
        "status": str(screen["status"]),
        "research_valid": screen.get("research_valid") is True,
        "publication_candidate": (
            screen.get("publication_candidate") is True
        ),
        "target_evaluation": {
            "target_calls": 0,
            "target_evaluation_performed": False,
            "hidden_target_attack_cohort_opened": False,
        },
        "protocol": {
            "design": (
                "three-fold leave-one-family-out source-only Stage B"
            ),
            "policy_seed_count": 1,
            "policy_seeds": sorted(
                {
                    int(
                        require_mapping(item["run"], "run")["seed"]
                    )
                    for item in verified_runs
                }
            ),
            "folds": len(verified_runs),
            "source_conditions": int(decision["condition_count"]),
            "primary_control": SCREEN_CONTROL,
            "learned_method": LEARNED_METHOD,
            "query_budget_including_initialization": 50,
        },
        "promotion": decision,
        "outcomes": {
            "method_summary": list(method_rows),
            "fold_summary": list(fold_rows),
        },
        "behavior_cloning": {
            "folds": len(bc_rows),
            "gate_passes": sum(
                bool(row["gate_passed"]) for row in bc_rows
            ),
            "mean_top5_gain_over_validation_oracle": mean(
                bc_rows,
                "top5_gain",
            ),
            "mean_soft_ce_improvement_over_validation_oracle": mean(
                bc_rows,
                "soft_ce_improvement",
            ),
        },
        "runtime": {
            "recorded_component_seconds": total_seconds,
            "recorded_component_minutes": total_seconds / 60,
            "folds": [
                {
                    key: row[key]
                    for key in (
                        "omitted_target_family",
                        "bc_seconds",
                        "ppo_seconds",
                        "source_evaluation_seconds",
                        "recorded_total_seconds",
                    )
                }
                for row in fold_rows
            ],
            "definition": (
                "sum of manifest BC, PPO-block, and source-evaluation "
                "component timers"
            ),
        },
        "environment": dict(environment),
        "integrity": {
            "screen_manifest_sha256": screen_manifest_sha,
            "study_code_digest": digest(
                screen.get("study_code_digest"),
                "study code digest",
            ),
            "protocol_sha256": digest(
                screen.get("protocol_sha256"),
                "protocol checksum",
            ),
            "verified_runs": len(verified_runs),
            "verified_sidecars": sidecar_count,
            "verified_results_files": len(input_checksums),
            "verified_trace_files": len(input_checksums),
            "all_source_audits_passed": True,
        },
        "interpretation": {
            "source_screen_result": (
                "positive" if decision.get("passed") is True else "negative"
            ),
            "target_transfer_claim_supported": False,
            "stage_c_authorized": decision.get("passed") is True,
            "target_evaluation_authorized": False,
        },
    }
    validate_portable_value(result, "Phase 2 summary")
    return result
