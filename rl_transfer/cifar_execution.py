"""Fail-closed victim selection, cache preflight, and path rendering."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping, Sequence

import torch

from .artifacts import sha256_file
from .paths import resolve_descendant


CIFAR_VICTIM_FAMILIES = (
    "classical_cnn",
    "modern_cnn",
    "transformer",
)


def selected_victim_families(
    target_family: str,
    *,
    source_victims_only: bool,
) -> tuple[str, ...]:
    if not source_victims_only:
        return CIFAR_VICTIM_FAMILIES
    return tuple(
        family
        for family in CIFAR_VICTIM_FAMILIES
        if family != target_family
    )


def validate_victim_population(
    victim_population: Mapping[
        str,
        Sequence[tuple[str, torch.nn.Module]],
    ],
    instance_counts: Mapping[str, int],
) -> tuple[str, ...]:
    """Validate the model bank before any checkpoint or fitting call."""

    if set(victim_population) != set(instance_counts):
        raise ValueError(
            "victim population does not exactly match selected families"
        )
    victim_ids: list[str] = []
    for family in instance_counts:
        instances = victim_population.get(family)
        if (
            not isinstance(instances, Sequence)
            or len(instances) != instance_counts[family]
        ):
            raise ValueError(
                f"victim population count mismatch for {family}"
            )
        for entry in instances:
            if (
                not isinstance(entry, Sequence)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not entry[0]
            ):
                raise ValueError("victim population entry is invalid")
            victim_ids.append(entry[0])
    if len(victim_ids) != len(set(victim_ids)):
        raise ValueError("victim population contains duplicate IDs")
    if len(victim_ids) != sum(instance_counts.values()):
        raise ValueError("victim population total count mismatch")
    return tuple(victim_ids)


def preflight_cache_only_victims(
    victim_cache_dir: Path,
    victim_ids: Sequence[str],
) -> None:
    """Fail atomically when any selected cached victim is unavailable."""

    errors: list[str] = []
    for victim_id in victim_ids:
        checkpoint_path = resolve_descendant(
            victim_cache_dir,
            f"{victim_id}.pt",
            label="victim checkpoint",
        )
        checksum_path = resolve_descendant(
            victim_cache_dir,
            f"{victim_id}.pt.sha256",
            label="victim checkpoint checksum",
        )
        if (
            checkpoint_path.is_symlink()
            or checksum_path.is_symlink()
            or not checkpoint_path.is_file()
            or not checksum_path.is_file()
        ):
            errors.append(f"{victim_id}: checkpoint or checksum is missing")
            continue
        expected = checksum_path.read_text().strip()
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or sha256_file(checkpoint_path) != expected
        ):
            errors.append(f"{victim_id}: checkpoint checksum failed")
    if errors:
        raise ValueError(
            "cache-only victim preflight failed before loading or fitting: "
            + "; ".join(errors)
        )


def portable_descendant(root: Path, path: Path, *, label: str) -> str:
    resolved = resolve_descendant(root, path, label=label)
    relative = resolved.relative_to(root.resolve())
    if not relative.parts:
        raise ValueError(f"{label} cannot identify the root itself")
    return relative.as_posix()


def portable_checkpoint_records(
    checkpoints: Mapping[str, Mapping[str, object]],
    *,
    run_dir: Path,
) -> dict[str, dict[str, object]]:
    portable: dict[str, dict[str, object]] = {}
    for name, record in checkpoints.items():
        path_value = record.get("path")
        if not isinstance(path_value, str):
            raise ValueError("policy checkpoint path is missing")
        portable[name] = {
            **record,
            "path": portable_descendant(
                run_dir,
                Path(path_value),
                label="portable policy checkpoint",
            ),
        }
    return portable
