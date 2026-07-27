"""Repository-contained path resolution for locked study entry points."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def resolve_descendant(
    root: str | Path,
    value: str | Path,
    *,
    label: str,
) -> Path:
    """Resolve a path and reject symlink escapes from an approved root."""

    approved = Path(root).resolve()
    raw = Path(value)
    candidate = (
        raw
        if raw.is_absolute()
        else approved / raw
    ).resolve()
    try:
        candidate.relative_to(approved)
    except ValueError as error:
        raise ValueError(
            f"{label} must resolve within its approved root"
        ) from error
    return candidate


def resolve_within_repository(
    value: str | Path,
    *,
    allowed_directory: str | Path,
    label: str,
) -> Path:
    repository = REPOSITORY_ROOT.resolve()
    allowed_value = Path(allowed_directory)
    if (
        allowed_value.is_absolute()
        or ".." in allowed_value.parts
        or not allowed_value.parts
    ):
        raise ValueError("allowed_directory must be repository-relative")
    allowed = (repository / allowed_value).resolve()
    try:
        allowed.relative_to(repository)
    except ValueError as error:
        raise ValueError(
            f"{allowed_directory} escapes the repository through a symlink"
        ) from error
    raw = Path(value)
    candidate = raw if raw.is_absolute() else repository / raw
    return resolve_descendant(allowed, candidate, label=label)
