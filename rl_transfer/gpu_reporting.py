"""Integrity checks for notebook-side study reporting and export."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping

from .artifacts import sha256_file
from .paths import resolve_descendant
from .verified_artifacts import load_verified_json


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TIMESTAMP_PATTERN = re.compile(r"[0-9]{8}_[0-9]{6}")


def _validated_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def load_verified_study_manifest(
    manifest_path: Path,
    study_dir: Path,
) -> dict[str, object]:
    """Load a manifest only from the study directory and verify its sidecar."""

    safe_path = resolve_descendant(
        study_dir,
        manifest_path,
        label="study manifest",
    )
    return load_verified_json(safe_path)


def resolve_verified_result_rows(
    run_dir: Path,
    runs_root: Path,
    study_dir: Path,
) -> Path:
    """Resolve target rows and verify them against the target cache."""

    safe_runs_root = resolve_descendant(
        study_dir,
        runs_root,
        label="study run root",
    )
    safe_run_dir = resolve_descendant(
        safe_runs_root,
        run_dir,
        label="manifest run directory",
    )
    target_cache_path = resolve_descendant(
        safe_run_dir,
        "target_evaluation.json",
        label="target evaluation cache",
    )
    target_cache = load_verified_json(target_cache_path)
    result_path = resolve_descendant(
        safe_run_dir,
        "results.jsonl",
        label="target result rows",
    )
    expected_results_sha = _validated_sha256(
        target_cache.get("results_sha256"),
        label="target result checksum",
    )
    if (
        not result_path.is_file()
        or sha256_file(result_path) != expected_results_sha
    ):
        raise ValueError("target result rows failed checksum verification")
    return result_path


def create_timestamped_export_directory(
    study_dir: Path,
    report_root: Path,
    timestamp: str,
) -> Path:
    """Create one non-overwriting export directory inside the study."""

    if _TIMESTAMP_PATTERN.fullmatch(timestamp) is None:
        raise ValueError("export timestamp must use YYYYMMDD_HHMMSS")
    safe_report_root = resolve_descendant(
        study_dir,
        report_root,
        label="paper artifact root",
    )
    export_dir = resolve_descendant(
        safe_report_root,
        safe_report_root / timestamp,
        label="paper artifact export",
    )
    if export_dir.exists():
        raise FileExistsError(
            "refusing to overwrite an existing artifact directory"
        )
    export_dir.mkdir(parents=True, exist_ok=False)
    return resolve_descendant(
        study_dir,
        export_dir,
        label="created paper artifact export",
    )


def load_verified_runtime_freeze(
    study: Mapping[str, object],
    study_dir: Path,
    repository: Path,
) -> tuple[Path, str]:
    """Read the run-start package snapshot after path and hash checks."""

    runtime = study.get("runtime_environment")
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime environment record is missing")
    recorded_path = runtime.get("pip_freeze_path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ValueError("runtime package snapshot path is missing")
    raw_path = Path(recorded_path)
    candidate = raw_path if raw_path.is_absolute() else repository / raw_path
    runtime_freeze = resolve_descendant(
        study_dir,
        candidate,
        label="run-start package snapshot",
    )
    if not runtime_freeze.is_file():
        raise ValueError("run-start package snapshot is missing")
    expected_freeze_sha = _validated_sha256(
        runtime.get("pip_freeze_sha256"),
        label="runtime package snapshot checksum",
    )
    freeze_bytes = runtime_freeze.read_bytes()
    if hashlib.sha256(freeze_bytes).hexdigest() != expected_freeze_sha:
        raise ValueError(
            "run-start package snapshot failed checksum verification"
        )
    return runtime_freeze, freeze_bytes.decode("utf-8")
