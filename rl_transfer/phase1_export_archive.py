"""Deterministic archive of checksum-verified Phase 1 source records."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import BinaryIO, Mapping, Sequence

from .paths import resolve_descendant
from .phase1_export_validation import (
    digest,
    require_mapping,
    require_sequence,
    validate_portable_value,
)


_MAX_LINE_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 50 * 1024 * 1024


class _HashingReader:
    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self._hasher = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        self._hasher.update(chunk)
        return chunk

    @property
    def hexdigest(self) -> str:
        return self._hasher.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _references_family(value: object, family: str) -> bool:
    family_key = family.casefold()
    if isinstance(value, Mapping):
        return any(
            _references_family(key, family) or _references_family(item, family)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_references_family(item, family) for item in value)
    return isinstance(value, str) and family_key in value.casefold()


def _validate_jsonl(
    path: Path,
    label: str,
    *,
    family_field: str,
    held_out_family: str,
    source_families: frozenset[str],
) -> None:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{label} contains an empty JSONL record")
            if len(line) > _MAX_LINE_BYTES:
                raise ValueError(f"{label} contains an oversized JSONL record")
            try:
                record = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{label} contains invalid JSON at line {line_number}"
                ) from error
            row = require_mapping(record, f"{label} line {line_number}")
            if _references_family(row, held_out_family):
                raise ValueError(
                    f"{label} line {line_number} references the held-out target family"
                )
            row_family = row.get(family_field)
            if not isinstance(row_family, str) or row_family not in source_families:
                raise ValueError(
                    f"{label} line {line_number} does not identify "
                    "a registered source family"
                )
            validate_portable_value(
                row,
                f"{label} line {line_number}",
            )


def _add_verified_file(
    archive: tarfile.TarFile,
    path: Path,
    member_name: str,
    expected_sha256: str,
) -> None:
    info = tarfile.TarInfo(member_name)
    info.size = path.stat().st_size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with path.open("rb") as handle:
        reader = _HashingReader(handle)
        archive.addfile(info, reader)
    if reader.hexdigest != expected_sha256:
        raise ValueError(f"raw source record changed during export: {member_name}")


def write_raw_source_archive(
    verified_runs: Sequence[Mapping[str, object]],
    destination: Path,
) -> None:
    """Write exactly two verified JSONL files per run with fixed metadata."""

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                with tarfile.open(
                    fileobj=gzip_handle,
                    mode="w|",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for item in sorted(
                        verified_runs,
                        key=lambda value: str(
                            require_mapping(
                                value.get("run"),
                                "verified run",
                            )["fingerprint"]
                        ),
                    ):
                        run = require_mapping(item.get("run"), "verified run")
                        cache = require_mapping(
                            item.get("cache"),
                            "verified source cache",
                        )
                        run_dir_value = item.get("run_dir")
                        if not isinstance(run_dir_value, Path):
                            raise ValueError("verified run directory is missing")
                        fingerprint = str(run["fingerprint"])
                        held_out_family = run.get("target_family")
                        if not isinstance(held_out_family, str) or not held_out_family:
                            raise ValueError("held-out target family is missing")
                        source_family_values = require_sequence(
                            run.get("source_families"),
                            "source families",
                        )
                        if any(
                            not isinstance(value, str) or not value
                            for value in source_family_values
                        ):
                            raise ValueError("source family names are malformed")
                        source_families = frozenset(source_family_values)
                        if (
                            not source_families
                            or len(source_families) != len(source_family_values)
                            or held_out_family in source_families
                        ):
                            raise ValueError(
                                "source families violate the LOFO boundary"
                            )
                        for filename, checksum_key, family_field in (
                            (
                                "source_results.jsonl",
                                "results_sha256",
                                "victim_family",
                            ),
                            (
                                "source_query_traces.jsonl",
                                "query_traces_sha256",
                                "family",
                            ),
                        ):
                            source = resolve_descendant(
                                run_dir_value,
                                filename,
                                label="raw source record",
                            )
                            _validate_jsonl(
                                source,
                                filename,
                                family_field=family_field,
                                held_out_family=held_out_family,
                                source_families=source_families,
                            )
                            _add_verified_file(
                                archive,
                                source,
                                f"runs/{fingerprint[:12]}/{filename}",
                                digest(
                                    cache.get(checksum_key),
                                    f"{filename} checksum",
                                ),
                            )
        if temporary.stat().st_size > _MAX_ARCHIVE_BYTES:
            raise ValueError("raw source archive exceeds the Git-safe size limit")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
