"""Deterministic raw-record archive for calibration evidence."""

from __future__ import annotations

import gzip
import io
from pathlib import PurePath
import tarfile
from typing import Mapping


def _validated_payloads(
    payloads: Mapping[str, bytes],
) -> tuple[tuple[str, bytes], ...]:
    validated = []
    for name, payload in payloads.items():
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or PurePath(name).name != name
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("archive member name must be a portable basename")
        if not isinstance(payload, bytes):
            raise TypeError("archive member payload must be bytes")
        validated.append((name, payload))
    return tuple(sorted(validated))


def raw_calibration_archive(payloads: Mapping[str, bytes]) -> bytes:
    """Archive the exact in-memory bytes that passed checksum validation."""

    members = _validated_payloads(payloads)
    tar_buffer = io.BytesIO()
    with tarfile.open(
        fileobj=tar_buffer,
        mode="w",
        format=tarfile.PAX_FORMAT,
    ) as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
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


__all__ = ("raw_calibration_archive",)
