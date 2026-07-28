"""Persisted-evidence validation and resume binding for Phase 2."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Mapping

from .artifacts import sha256_file
from .cifar_config import MacPilotConfig
from .paths import resolve_descendant
from .phase2_config import Phase2ScreenConfig
from .phase2_promotion import validate_source_run_semantics
from .verified_artifacts import load_verified_json


def _portable_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} is missing")
    path = Path(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.parts in {(), (".",)}
        or re.match(r"^[A-Za-z]:", value) is not None
    ):
        raise ValueError(f"{label} must be a portable relative path")
    return path


def _json_record(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _config_record(config: Phase2ScreenConfig) -> dict[str, object]:
    record = _json_record(asdict(config))
    if not isinstance(record, dict):
        raise RuntimeError("Phase 2 configuration serialization failed")
    return record


def validate_source_run_artifacts(
    run: Mapping[str, object],
    *,
    derived_config: MacPilotConfig,
    run_output_dir: Path,
) -> None:
    """Recheck persisted cell evidence and checksums before promotion."""

    validate_source_run_semantics(
        run,
        family=derived_config.target_family,
        seed=derived_config.seed,
    )
    if run.get("config") != _json_record(asdict(derived_config)):
        raise ValueError("source run configuration mismatch")
    raw_run_dir = run.get("run_dir")
    portable_run_dir = _portable_path(
        raw_run_dir,
        label="source run directory",
    )
    run_dir = resolve_descendant(
        run_output_dir,
        portable_run_dir,
        label="Phase 2 source run directory",
    )
    persisted = load_verified_json(run_dir / "manifest.json")
    for key in (
        "fingerprint",
        "config_digest",
        "split_digest",
        "seed",
        "target_family",
        "target_evaluation_performed",
        "target_calls",
        "victim_access_audit",
    ):
        if persisted.get(key) != run.get(key):
            raise ValueError(
                f"persisted source manifest mismatch: {key}"
            )
    source_cache = load_verified_json(
        run_dir / "source_evaluation.json"
    )
    if source_cache.get("source_evaluation") != _json_record(
        run.get("source_evaluation")
    ):
        raise ValueError("persisted source evaluation mismatch")
    runtime = run.get("runtime")
    policy = run.get("policy")
    source_families = run.get("source_families")
    victim_instances = run.get("victim_instances")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(policy, Mapping)
        or not isinstance(source_families, list)
        or not isinstance(victim_instances, Mapping)
    ):
        raise ValueError("source evidence binding inputs are missing")
    if (
        set(victim_instances) != set(source_families)
        or derived_config.target_family in victim_instances
    ):
        raise ValueError(
            "held-out-family victim records must be absent"
        )
    source_victim_checkpoints: dict[str, object] = {}
    for family in source_families:
        instances = victim_instances.get(family)
        if not isinstance(instances, list):
            raise ValueError("source victim checkpoint records are missing")
        for instance in instances:
            if not isinstance(instance, Mapping):
                raise ValueError("source victim checkpoint record is invalid")
            victim_id = instance.get("victim_id")
            checkpoint_sha = instance.get("checkpoint_sha256")
            if (
                not isinstance(victim_id, str)
                or not isinstance(checkpoint_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha)
                is None
            ):
                raise ValueError("source victim ID is invalid")
            checkpoint_path = resolve_descendant(
                run_output_dir,
                _portable_path(
                    instance.get("checkpoint"),
                    label="source victim checkpoint",
                ),
                label="Phase 2 source victim checkpoint",
            )
            checkpoint_sidecar = checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".sha256"
            )
            if (
                not checkpoint_path.is_file()
                or not checkpoint_sidecar.is_file()
                or checkpoint_sidecar.read_text().strip()
                != checkpoint_sha
                or sha256_file(checkpoint_path) != checkpoint_sha
            ):
                raise ValueError(
                    "source victim checkpoint checksum failed"
                )
            source_victim_checkpoints[victim_id] = checkpoint_sha
    expected_binding = {
        "config_digest": run.get("config_digest"),
        "code_digest": runtime.get("code_digest"),
        "split_digest": run.get("split_digest"),
        "data_role_digests": run.get("data_role_digests"),
        "policy_checkpoints": policy.get("checkpoints"),
        "source_victim_checkpoints": source_victim_checkpoints,
    }
    if source_cache.get("binding") != expected_binding:
        raise ValueError("persisted source evidence binding mismatch")
    results_path = run_dir / "source_results.jsonl"
    traces_path = run_dir / "source_query_traces.jsonl"
    if (
        source_cache.get("results_sha256")
        != sha256_file(results_path)
        or source_cache.get("query_traces_sha256")
        != sha256_file(traces_path)
    ):
        raise ValueError("source row or query-trace checksum mismatch")
    checkpoint_value = (
        policy.get("checkpoint")
        if isinstance(policy, Mapping)
        else None
    )
    checkpoint_digest = (
        policy.get("checkpoint_sha256")
        if isinstance(policy, Mapping)
        else None
    )
    if (
        not isinstance(checkpoint_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_digest) is None
    ):
        raise ValueError("source policy checkpoint identity is invalid")
    checkpoint = resolve_descendant(
        run_dir,
        _portable_path(
            checkpoint_value,
            label="source policy checkpoint",
        ),
        label="Phase 2 policy checkpoint",
    )
    checkpoint_records = policy.get("checkpoints")
    if not isinstance(checkpoint_records, Mapping):
        raise ValueError("source policy checkpoint records are missing")
    for record in checkpoint_records.values():
        if not isinstance(record, Mapping):
            raise ValueError("source policy checkpoint record is invalid")
        resolve_descendant(
            run_dir,
            _portable_path(
                record.get("path"),
                label="source policy checkpoint record",
            ),
            label="Phase 2 policy checkpoint record",
        )
    sidecar = checkpoint.with_suffix(
        checkpoint.suffix + ".sha256"
    )
    if (
        not checkpoint.is_file()
        or not sidecar.is_file()
        or sidecar.read_text().strip() != checkpoint_digest
        or sha256_file(checkpoint) != checkpoint_digest
    ):
        raise ValueError("source policy checkpoint checksum failed")


def load_resumable_screen_manifest(
    path: Path,
    *,
    config: Phase2ScreenConfig,
    base_config_digest: str,
    code_digest: str,
    dataset_version: str,
    protocol_sha256: str,
) -> dict[str, object] | None:
    if not path.exists() and not path.with_suffix(
        path.suffix + ".sha256"
    ).exists():
        return None
    manifest = load_verified_json(path)
    expected_config = _config_record(config)
    mismatches = (
        manifest.get("schema_version") != 1,
        manifest.get("status")
        not in {
            "screen_running",
            "screen_deadline_reached",
            "screen_complete",
            "screen_failed",
        },
        manifest.get("research_valid") is not False,
        manifest.get("publication_candidate") is not False,
        manifest.get("config") != expected_config,
        manifest.get("base_config_digest") != base_config_digest,
        manifest.get("study_code_digest") != code_digest,
        manifest.get("dataset_version") != dataset_version,
        manifest.get("protocol_sha256") != protocol_sha256,
        manifest.get("target_evaluation_performed") is not False,
        manifest.get("target_calls") != 0,
    )
    if any(mismatches):
        raise ValueError(
            "resumable Phase 2 manifest does not match this locked run"
        )
    runs = manifest.get("source_runs")
    if not isinstance(runs, list) or any(
        not isinstance(run, dict) for run in runs
    ):
        raise ValueError("resumable Phase 2 source runs are invalid")
    return manifest
