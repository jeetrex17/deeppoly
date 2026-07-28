"""Bounded provenance validation for the Phase 2 evidence exporter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping

from .paths import resolve_descendant
from .phase1_export_validation import (
    digest,
    require_mapping,
    require_sequence,
    validate_portable_value,
)
from .phase2_promotion import validate_source_run_semantics


MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_DEPENDENCY_FREEZE_BYTES = 2 * 1024 * 1024
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_SOURCE_RECORD_BYTES = 25 * 1024 * 1024
MAX_SIDECARS = 10_000
_READ_CHUNK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PINNED_REQUIREMENT = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:\[[A-Za-z0-9_,.-]+\])?"
    r"==[A-Za-z0-9][A-Za-z0-9_.+!-]*"
)
_EDITABLE_GITHUB_REQUIREMENT = re.compile(
    r"-e git\+https://github\.com/"
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git@"
    r"[0-9a-f]{40}#egg=[A-Za-z0-9_.-]+"
)


def _open_regular_file(path: Path, *, label: str) -> tuple[int, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a regular file") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ValueError(f"{label} must be a regular file")
        return descriptor, details.st_size
    except Exception:
        os.close(descriptor)
        raise


def read_bounded_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    """Read one regular file through a no-follow descriptor with a limit."""

    descriptor, recorded_size = _open_regular_file(path, label=label)
    try:
        if recorded_size > max_bytes:
            raise ValueError(f"{label} exceeds the permitted size")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, max_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"{label} exceeds the permitted size")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_bounded_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> str:
    descriptor, recorded_size = _open_regular_file(path, label=label)
    try:
        if recorded_size > max_bytes:
            raise ValueError(f"{label} exceeds the permitted size")
        checksum = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"{label} exceeds the permitted size")
            checksum.update(chunk)
        return checksum.hexdigest()
    finally:
        os.close(descriptor)


def bounded_regular_file_digest(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> str:
    """Hash one regular no-follow file while enforcing a byte limit."""

    return _sha256_bounded_regular_file(
        path,
        label=label,
        max_bytes=max_bytes,
    )


def _sidecar_digest(path: Path, *, label: str) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        value = read_bounded_regular_file(
            sidecar,
            label=f"{label} checksum sidecar",
            max_bytes=128,
        ).decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} checksum sidecar is invalid") from error
    return digest(value, f"{label} checksum sidecar")


def _verified_artifact_digest(
    path: Path,
    *,
    expected: str,
    label: str,
    max_bytes: int,
) -> str:
    sidecar = _sidecar_digest(path, label=label)
    actual = _sha256_bounded_regular_file(
        path,
        label=label,
        max_bytes=max_bytes,
    )
    if sidecar != expected or actual != expected:
        raise ValueError(f"{label} checksum failed")
    return actual


def load_bounded_verified_json_with_digest(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, object], str]:
    """Load one bounded JSON object and return its verified byte digest."""

    payload = read_bounded_regular_file(
        path,
        label=label,
        max_bytes=MAX_JSON_BYTES,
    )
    expected = _sidecar_digest(path, label=label)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} checksum failed")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value, actual


def load_bounded_verified_json(
    path: Path,
    *,
    label: str,
) -> dict[str, object]:
    """Load one checksum-bound JSON object without an unbounded read."""

    value, _ = load_bounded_verified_json_with_digest(
        path,
        label=label,
    )
    return value


def canonical_dependency_freeze(value: str) -> str:
    """Return only strict, credential-free, reproducible requirement pins."""

    records = [
        line.strip()
        for line in value.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not records or any(
        _PINNED_REQUIREMENT.fullmatch(line) is None
        and _EDITABLE_GITHUB_REQUIREMENT.fullmatch(line) is None
        for line in records
    ):
        raise ValueError(
            "dependency freeze contains an unapproved requirement form"
        )
    return "".join(f"{record}\n" for record in records)


def verified_dependency_freeze(
    source_root: Path,
    screen_environment: Mapping[str, object],
) -> str:
    freeze_path = resolve_descendant(
        source_root,
        "pip_freeze.txt",
        label="Phase 2 dependency freeze",
    )
    payload = read_bounded_regular_file(
        freeze_path,
        label="Phase 2 dependency freeze",
        max_bytes=MAX_DEPENDENCY_FREEZE_BYTES,
    )
    expected = digest(
        screen_environment.get("pip_freeze_sha256"),
        "dependency freeze checksum",
    )
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("dependency freeze checksum failed")
    try:
        value = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("dependency freeze is not UTF-8") from error
    return canonical_dependency_freeze(value)


def verify_all_sidecars(source_root: Path) -> int:
    sidecars = sorted(source_root.rglob("*.sha256"))
    if not sidecars or len(sidecars) > MAX_SIDECARS:
        raise ValueError("Phase 2 archive has an invalid sidecar count")
    for sidecar in sidecars:
        artifact = sidecar.with_suffix("")
        expected = _sidecar_digest(artifact, label=artifact.name)
        actual = _sha256_bounded_regular_file(
            artifact,
            label=artifact.name,
            max_bytes=MAX_ARTIFACT_BYTES,
        )
        if actual != expected:
            raise ValueError(
                f"checksum sidecar failed: {artifact.name}"
            )
    return len(sidecars)


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


def _bind_policy_checkpoint(
    run_dir: Path,
    policy: Mapping[str, object],
) -> str:
    expected = digest(
        policy.get("checkpoint_sha256"),
        "policy checkpoint checksum",
    )
    checkpoint = resolve_descendant(
        run_dir,
        _portable_path(
            policy.get("checkpoint"),
            label="policy checkpoint",
        ),
        label="Phase 2 policy checkpoint",
    )
    _verified_artifact_digest(
        checkpoint,
        expected=expected,
        label="policy checkpoint",
        max_bytes=MAX_CHECKPOINT_BYTES,
    )
    records = require_mapping(
        policy.get("checkpoints"),
        "policy checkpoint records",
    )
    if not records:
        raise ValueError("policy checkpoint records cannot be empty")
    main_bound = False
    for value in records.values():
        record = require_mapping(value, "policy checkpoint record")
        record_digest = digest(
            record.get("sha256"),
            "policy checkpoint record checksum",
        )
        record_path = resolve_descendant(
            run_dir,
            _portable_path(
                record.get("path"),
                label="policy checkpoint record",
            ),
            label="Phase 2 policy checkpoint record",
        )
        _verified_artifact_digest(
            record_path,
            expected=record_digest,
            label="policy checkpoint",
            max_bytes=MAX_CHECKPOINT_BYTES,
        )
        if record_path == checkpoint:
            main_bound = record_digest == expected
    if not main_bound:
        raise ValueError("policy checkpoint records do not bind main policy")
    return expected


def _bind_victim_checkpoints(
    source_root: Path,
    run: Mapping[str, object],
) -> None:
    families = require_sequence(
        run.get("source_families"),
        "source families",
    )
    instances = require_mapping(
        run.get("victim_instances"),
        "victim instances",
    )
    if set(instances) != set(families):
        raise ValueError("victim checkpoint families are incomplete")
    for family in families:
        records = require_sequence(
            instances.get(family),
            "victim instance records",
        )
        if not records:
            raise ValueError("victim checkpoint records cannot be empty")
        for value in records:
            record = require_mapping(value, "victim checkpoint record")
            expected = digest(
                record.get("checkpoint_sha256"),
                "victim checkpoint checksum",
            )
            checkpoint = resolve_descendant(
                source_root,
                _portable_path(
                    record.get("checkpoint"),
                    label="victim checkpoint",
                ),
                label="Phase 2 victim checkpoint",
            )
            _verified_artifact_digest(
                checkpoint,
                expected=expected,
                label="victim checkpoint",
                max_bytes=MAX_CHECKPOINT_BYTES,
            )


def _verify_jsonl_payload(payload: bytes, *, label: str) -> int:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    rows = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label} line {line_number} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{label} rows must be objects")
        validate_portable_value(value, label)
        rows += 1
    if rows == 0:
        raise ValueError(f"{label} cannot be empty")
    return rows


def verified_runs(
    source_root: Path,
    screen: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Bind screen runs, checkpoints, and raw records to verified bytes."""

    runs_value = require_sequence(screen.get("source_runs"), "source_runs")
    if not runs_value:
        raise ValueError("source_runs cannot be empty")
    runs_root = resolve_descendant(
        source_root,
        "runs",
        label="Phase 2 run root",
    )
    verified: list[dict[str, object]] = []
    checksums: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, embedded_value in enumerate(runs_value):
        embedded = dict(
            require_mapping(embedded_value, f"source_runs[{index}]")
        )
        fingerprint = digest(
            embedded.get("fingerprint"),
            f"source_runs[{index}] fingerprint",
        )
        if fingerprint in seen:
            raise ValueError("source run fingerprints must be unique")
        seen.add(fingerprint)
        run_dir = resolve_descendant(
            runs_root,
            fingerprint[:12],
            label="Phase 2 source run",
        )
        manifest_path = resolve_descendant(
            run_dir,
            "manifest.json",
            label="source run manifest",
        )
        run, manifest_sha = load_bounded_verified_json_with_digest(
            manifest_path,
            label="source run manifest",
        )
        if run != embedded:
            raise ValueError(
                "source run manifest disagrees with screen manifest"
            )
        family = run.get("target_family")
        seed = run.get("seed")
        if (
            not isinstance(family, str)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            raise ValueError("source run identity is invalid")
        validate_source_run_semantics(run, family=family, seed=seed)
        policy = require_mapping(run.get("policy"), "policy")
        checkpoint_sha = _bind_policy_checkpoint(run_dir, policy)
        _bind_victim_checkpoints(runs_root, run)
        evaluation_path = resolve_descendant(
            run_dir,
            "source_evaluation.json",
            label="source evaluation",
        )
        evaluation, evaluation_sha = (
            load_bounded_verified_json_with_digest(
            evaluation_path,
            label="source evaluation",
            )
        )
        if evaluation.get("source_evaluation") != run.get(
            "source_evaluation"
        ):
            raise ValueError(
                "source evaluation disagrees with run manifest"
            )
        raw_records: dict[str, bytes] = {}
        row_counts: dict[str, int] = {}
        for name, label in (
            ("source_results.jsonl", "source results"),
            ("source_query_traces.jsonl", "source query traces"),
        ):
            path = resolve_descendant(run_dir, name, label=label)
            payload = read_bounded_regular_file(
                path,
                label=label,
                max_bytes=MAX_SOURCE_RECORD_BYTES,
            )
            expected = digest(
                evaluation.get(
                    "results_sha256"
                    if name == "source_results.jsonl"
                    else "query_traces_sha256"
                ),
                f"{label} checksum",
            )
            if hashlib.sha256(payload).hexdigest() != expected:
                failure_label = (
                    "source result rows"
                    if name == "source_results.jsonl"
                    else "source query traces"
                )
                raise ValueError(
                    f"{failure_label} failed checksum verification"
                )
            raw_records[name] = payload
            row_counts[name] = _verify_jsonl_payload(
                payload,
                label=label,
            )
        verified.append(
            {
                "fingerprint": fingerprint,
                "run": run,
                "evaluation": evaluation,
                "run_dir": run_dir,
                "raw_records": raw_records,
            }
        )
        checksums.append(
            {
                "fingerprint": fingerprint,
                "omitted_target_family": family,
                "seed": seed,
                "run_manifest_sha256": manifest_sha,
                "source_evaluation_sha256": evaluation_sha,
                "source_results_sha256": hashlib.sha256(
                    raw_records["source_results.jsonl"]
                ).hexdigest(),
                "source_query_traces_sha256": hashlib.sha256(
                    raw_records["source_query_traces.jsonl"]
                ).hexdigest(),
                "policy_checkpoint_sha256": checkpoint_sha,
                "source_result_rows": row_counts[
                    "source_results.jsonl"
                ],
                "source_query_trace_rows": row_counts[
                    "source_query_traces.jsonl"
                ],
            }
        )
    verified.sort(
        key=lambda item: (
            str(require_mapping(item["run"], "run")["target_family"]),
            int(require_mapping(item["run"], "run")["seed"]),
        )
    )
    checksums.sort(
        key=lambda row: (
            str(row["omitted_target_family"]),
            int(row["seed"]),
        )
    )
    return verified, checksums
