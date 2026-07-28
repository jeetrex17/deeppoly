"""Verified Phase 1 victim-cache reuse for Phase 2 screening."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .artifacts import sha256_file
from .cifar_config import MacPilotConfig
from .cifar_data import build_cifar_split
from .cifar_victim_cache import (
    victim_cache_digest as _victim_cache_digest,
    victim_code_digest as _victim_code_digest,
)
from .phase2_config import Phase2ScreenConfig


MAX_STUDY_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_VICTIM_CHECKPOINT_BYTES = 64 * 1024 * 1024
EXPECTED_VICTIM_FAMILIES = (
    "classical_cnn",
    "modern_cnn",
    "transformer",
)
EXPECTED_INSTANCES_PER_FAMILY = 3
EXPECTED_VICTIM_CHECKPOINTS = (
    len(EXPECTED_VICTIM_FAMILIES) * EXPECTED_INSTANCES_PER_FAMILY
)


def _bounded_bytes(path: Path, maximum: int, *, label: str) -> bytes:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 0 < path.stat().st_size <= maximum
    ):
        raise ValueError(f"{label} exceeds the safe size or is unavailable")
    with path.open("rb") as handle:
        payload = handle.read(maximum + 1)
    if not 0 < len(payload) <= maximum:
        raise ValueError(f"{label} exceeds the safe size or is unavailable")
    return payload


def _sidecar_value(path: Path, *, label: str) -> str:
    try:
        return _bounded_bytes(path, 256, label=label).decode(
            "ascii"
        ).strip()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not ASCII") from error


def _contract_fingerprint(contract: Mapping[str, object]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _verified_checkpoint(
    path: Path,
    expected_fingerprint: str,
    expected_sha256: str,
) -> str:
    checkpoint_bytes = _bounded_bytes(
        path,
        MAX_VICTIM_CHECKPOINT_BYTES,
        label="victim cache checkpoint",
    )
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_digest = _sidecar_value(
        sidecar,
        label="victim cache checksum sidecar",
    )
    if (
        sidecar_digest != expected_sha256
        or hashlib.sha256(checkpoint_bytes).hexdigest()
        != expected_sha256
    ):
        raise ValueError("victim cache checkpoint checksum failed")
    payload = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("metadata"), Mapping)
    ):
        raise ValueError("victim cache checkpoint schema is invalid")
    metadata = payload["metadata"]
    contract = metadata.get("cache_contract")
    fingerprint = metadata.get("fingerprint")
    if (
        not isinstance(contract, Mapping)
        or not isinstance(fingerprint, str)
        or fingerprint != expected_fingerprint
        or _contract_fingerprint(contract) != fingerprint
    ):
        raise ValueError("victim cache contract fingerprint mismatch")
    return fingerprint


def _pinned_manifest_allowlist(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_cache_fingerprint: str,
) -> dict[str, str]:
    manifest_bytes = _bounded_bytes(
        manifest_path,
        MAX_STUDY_MANIFEST_BYTES,
        label="pinned Phase 1 study manifest",
    )
    if (
        len(manifest_bytes) > MAX_STUDY_MANIFEST_BYTES
        or re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256)
        is None
        or re.fullmatch(r"[0-9a-f]{64}", expected_cache_fingerprint)
        is None
    ):
        raise ValueError("pinned Phase 1 digest is invalid")
    sidecar = manifest_path.with_suffix(
        manifest_path.suffix + ".sha256"
    )
    if (
        _sidecar_value(
            sidecar,
            label="pinned Phase 1 manifest checksum",
        )
        != expected_manifest_sha256
        or hashlib.sha256(manifest_bytes).hexdigest()
        != expected_manifest_sha256
    ):
        raise ValueError("pinned Phase 1 manifest checksum failed")
    payload = json.loads(manifest_bytes)
    runs = payload.get("source_runs") if isinstance(payload, Mapping) else None
    if (
        not isinstance(runs, list)
        or not 1 <= len(runs) <= 100
        or any(not isinstance(run, Mapping) for run in runs)
    ):
        raise ValueError("pinned Phase 1 source-run records are invalid")
    matching = [
        run
        for run in runs
        if run.get("victim_cache_digest") == expected_cache_fingerprint
    ]
    if not matching:
        raise ValueError(
            "pinned Phase 1 manifest lacks the required victim cache"
        )
    canonical: dict[str, str] | None = None
    for run in matching:
        instances = run.get("victim_instances")
        if (
            not isinstance(instances, Mapping)
            or set(instances) != set(EXPECTED_VICTIM_FAMILIES)
        ):
            raise ValueError(
                "pinned Phase 1 victim-family allowlist is invalid"
            )
        current: dict[str, str] = {}
        for family in EXPECTED_VICTIM_FAMILIES:
            family_instances = instances[family]
            if (
                not isinstance(family_instances, list)
                or len(family_instances)
                != EXPECTED_INSTANCES_PER_FAMILY
            ):
                raise ValueError(
                    "pinned Phase 1 victim count is invalid"
                )
            for instance in family_instances:
                if not isinstance(instance, Mapping):
                    raise ValueError(
                        "pinned Phase 1 victim record is invalid"
                    )
                victim_id = instance.get("victim_id")
                checkpoint_sha = instance.get("checkpoint_sha256")
                checkpoint_value = instance.get("checkpoint")
                if (
                    not isinstance(victim_id, str)
                    or re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}",
                        victim_id,
                    )
                    is None
                    or not isinstance(checkpoint_sha, str)
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        checkpoint_sha,
                    )
                    is None
                    or not isinstance(checkpoint_value, str)
                    or Path(checkpoint_value).name != f"{victim_id}.pt"
                    or victim_id in current
                ):
                    raise ValueError(
                        "pinned Phase 1 checkpoint identity is invalid"
                    )
                current[victim_id] = checkpoint_sha
        if len(current) != EXPECTED_VICTIM_CHECKPOINTS:
            raise ValueError("pinned Phase 1 victim count is invalid")
        if canonical is None:
            canonical = current
        elif current != canonical:
            raise ValueError(
                "pinned Phase 1 runs disagree on victim identities"
            )
    if canonical is None:
        raise ValueError("pinned Phase 1 victim allowlist is empty")
    return canonical


def _materialize_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> str:
    if sha256_file(source) != expected_sha256:
        raise ValueError(
            "victim cache source changed before materialization"
        )
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or sha256_file(destination) != expected_sha256
        ):
            raise ValueError(
                "existing victim cache destination conflicts with source"
            )
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )
    if temporary.exists():
        raise ValueError(
            "stale victim-cache materialization is present"
        )
    try:
        shutil.copy2(source, temporary)
        if (
            sha256_file(temporary) != expected_sha256
            or sha256_file(source) != expected_sha256
        ):
            raise ValueError(
                "victim cache source changed during materialization"
            )
        os.replace(temporary, destination)
        if sha256_file(destination) != expected_sha256:
            raise ValueError(
                "victim cache destination checksum failed"
            )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return "atomic_copy"


def mirror_verified_victim_cache(
    source_root: Path,
    destination_root: Path,
    *,
    study_manifest_path: Path,
    expected_study_manifest_sha256: str,
    expected_cache_fingerprint: str,
) -> dict[str, object]:
    """Materialize only victims authenticated by the pinned Phase 1 study."""

    if (
        source_root.is_symlink()
        or not source_root.is_dir()
        or source_root.resolve() == destination_root.resolve()
    ):
        raise ValueError("verified victim cache source is unavailable")
    allowlist = _pinned_manifest_allowlist(
        study_manifest_path,
        expected_manifest_sha256=expected_study_manifest_sha256,
        expected_cache_fingerprint=expected_cache_fingerprint,
    )
    cache_directory = source_root / expected_cache_fingerprint[:12]
    if cache_directory.is_symlink() or not cache_directory.is_dir():
        raise ValueError("required victim cache fingerprint is unavailable")
    checkpoints = tuple(sorted(cache_directory.glob("*.pt")))
    sidecars = tuple(sorted(cache_directory.glob("*.pt.sha256")))
    expected_checkpoint_names = {
        f"{victim_id}.pt" for victim_id in allowlist
    }
    expected_sidecar_names = {
        f"{name}.sha256" for name in expected_checkpoint_names
    }
    if (
        len(checkpoints) != EXPECTED_VICTIM_CHECKPOINTS
        or {path.name for path in checkpoints}
        != expected_checkpoint_names
        or {path.name for path in sidecars} != expected_sidecar_names
    ):
        raise ValueError(
            "victim cache has missing or extra checkpoint artifacts"
        )
    materializations: set[str] = set()
    destination_directory = (
        destination_root / expected_cache_fingerprint[:12]
    )
    for checkpoint in checkpoints:
        victim_id = checkpoint.stem
        _verified_checkpoint(
            checkpoint,
            expected_cache_fingerprint,
            allowlist[victim_id],
        )
        sidecar = checkpoint.with_suffix(
            checkpoint.suffix + ".sha256"
        )
        materializations.add(
            _materialize_verified_file(
                checkpoint,
                destination_directory / checkpoint.name,
                expected_sha256=allowlist[victim_id],
            )
        )
        materializations.add(
            _materialize_verified_file(
                sidecar,
                destination_directory / sidecar.name,
                expected_sha256=hashlib.sha256(
                    (allowlist[victim_id] + "\n").encode("utf-8")
                ).hexdigest(),
            )
        )
    return {
        "all_verified": True,
        "authentication": "pinned_phase1_study_manifest",
        "study_manifest_sha256": expected_study_manifest_sha256,
        "checkpoint_count": len(allowlist),
        "cache_fingerprints": [expected_cache_fingerprint],
        "checkpoint_sha256_by_victim": dict(sorted(allowlist.items())),
        "materialization": sorted(materializations),
    }


def expected_victim_cache_fingerprint(
    config: Phase2ScreenConfig,
    base: MacPilotConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str,
) -> str:
    """Compute the exact cache key before a cell can start training."""

    targets = getattr(train_dataset, "targets", None)
    test_targets = getattr(test_dataset, "targets", None)
    if not isinstance(targets, Sequence) or not isinstance(
        test_targets,
        Sequence,
    ):
        raise ValueError(
            "CIFAR train and test targets are required for cache preflight"
        )
    split = build_cifar_split(
        targets,
        test_targets,
        base.victim_train_images,
        base.policy_train_images,
        base.source_validation_images,
        base.outer_test_images,
        config.split_seed,
    )
    fingerprint_config = replace(
        base,
        seed=config.seeds[0],
        split_seed=config.split_seed,
        victim_seed=config.victim_seed,
    )
    return _victim_cache_digest(
        fingerprint_config,
        split.digest,
        dataset_version,
        _victim_code_digest(),
        "cuda",
    )
