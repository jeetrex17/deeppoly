"""Deterministic raw-record archive creation for Phase 2 evidence."""

from __future__ import annotations

import gzip
import io
import tarfile
from typing import Mapping, Sequence


_MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024


def raw_source_archive(
    verified_runs: Sequence[Mapping[str, object]],
) -> bytes:
    """Archive the same bounded bytes that passed checksum validation."""

    tar_buffer = io.BytesIO()
    total = 0
    with tarfile.open(
        fileobj=tar_buffer,
        mode="w",
        format=tarfile.PAX_FORMAT,
    ) as archive:
        for item in verified_runs:
            fingerprint = str(item["fingerprint"])
            raw_records = item.get("raw_records")
            if not isinstance(raw_records, Mapping):
                raise ValueError("verified raw source records are missing")
            for name in (
                "source_results.jsonl",
                "source_query_traces.jsonl",
            ):
                payload = raw_records.get(name)
                if not isinstance(payload, bytes):
                    raise ValueError("verified raw source record is invalid")
                total += len(payload)
                if total > _MAX_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        "raw Phase 2 records exceed the archive limit"
                    )
                info = tarfile.TarInfo(
                    f"runs/{fingerprint[:12]}/{name}"
                )
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
    compressed = io.BytesIO()
    with gzip.GzipFile(
        fileobj=compressed,
        mode="wb",
        compresslevel=9,
        mtime=0,
        filename="",
    ) as handle:
        handle.write(tar_buffer.getvalue())
    return compressed.getvalue()
