"""CIFAR run-manifest provenance and verified JSON writes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from .artifacts import sha256_file
from .paths import resolve_descendant


def git_revision() -> str:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def code_digest() -> str:
    hasher = hashlib.sha256()
    package_root = Path(__file__).parent
    for path in sorted(package_root.glob("*.py")):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def git_worktree_state() -> dict[str, object]:
    try:
        result = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"dirty": None, "status_sha256": None}
    status = result.stdout
    return {
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
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
