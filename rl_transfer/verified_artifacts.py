"""Checksum-verified JSON artifacts with repository-safe sidecars."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import sha256_file
from .paths import resolve_descendant


def write_verified_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Atomically replace JSON plus its SHA-256 sidecar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolve_descendant(
        path.parent,
        path.with_suffix(path.suffix + ".tmp"),
        label="JSON temporary file",
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    temporary.replace(path)
    checksum = resolve_descendant(
        path.parent,
        path.with_suffix(path.suffix + ".sha256"),
        label="JSON checksum",
    )
    checksum_temporary = resolve_descendant(
        path.parent,
        checksum.with_suffix(checksum.suffix + ".tmp"),
        label="JSON checksum temporary file",
    )
    checksum_temporary.write_text(sha256_file(path) + "\n")
    checksum_temporary.replace(checksum)


def load_verified_json(path: Path) -> dict[str, object]:
    """Load JSON only after its SHA-256 sidecar verifies."""

    checksum = resolve_descendant(
        path.parent,
        path.with_suffix(path.suffix + ".sha256"),
        label="JSON checksum",
    )
    if not path.is_file() or not checksum.is_file():
        raise ValueError("verified JSON artifact is incomplete")
    expected = checksum.read_text().strip()
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError("verified JSON artifact checksum failed")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("verified JSON artifact must contain an object")
    return payload
