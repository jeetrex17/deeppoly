"""Atomic Stage A result output and dry-run descriptions."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

from .artifacts import sha256_file
from .paths import REPOSITORY_ROOT
from .phase2_temperature_manifest import (
    Phase1Selection,
    STAGE_A_RANKING_RULE,
    STAGE_A_RANKING_TIE_BAND,
    StageARequest,
)


def portable_artifact_path(
    path: Path,
    *,
    relative_to: Path | None = None,
) -> str:
    """Render a path without persisting host-specific absolute prefixes."""

    raw = Path(path)
    resolved = (
        raw.resolve()
        if raw.is_absolute()
        else (REPOSITORY_ROOT / raw).resolve()
    )
    anchors = (REPOSITORY_ROOT, relative_to)
    for anchor in anchors:
        if anchor is None:
            continue
        try:
            relative = resolved.relative_to(Path(anchor).resolve())
        except ValueError:
            continue
        rendered = relative.as_posix()
        return rendered if rendered else "."
    fallback = resolved.name
    if not fallback or fallback in {".", ".."}:
        raise ValueError("artifact path cannot be rendered portably")
    return fallback


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_verified_jsonl(
    path: Path,
    rows: Iterable[object],
) -> str:
    """Atomically write JSONL and its SHA-256 sidecar."""

    encoded: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            payload = dict(row)
        elif is_dataclass(row) and not isinstance(row, type):
            payload = asdict(row)
        else:
            raise TypeError("verified JSONL rows must be mappings or dataclasses")
        encoded.append(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
    _atomic_text(path, "".join(encoded))
    digest = sha256_file(path)
    _atomic_text(path.with_suffix(path.suffix + ".sha256"), digest + "\n")
    return digest


def build_stage_a_dry_run(
    request: StageARequest,
    selection: Phase1Selection,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "source_only_phase1_checkpoint_temperature_diagnostic",
        "decision_scope": "phase1_checkpoint_diagnostic_only",
        "applies_to_new_phase2_architecture": False,
        "authorizes_phase2_deployment_temperature": False,
        "research_valid": False,
        "phase1_manifest": portable_artifact_path(
            selection.manifest_path,
            relative_to=request.phase1_root,
        ),
        "phase1_manifest_sha256": selection.manifest_sha256,
        "phase1_dataset_version": selection.dataset_version,
        "phase1_dataset_content_sha256": (
            selection.dataset_content_sha256
        ),
        "selected_policy_seeds": list(request.seeds),
        "selected_folds": list(request.folds),
        "selected_cells": len(selection.folds),
        "temperatures": list(request.temperatures),
        "ranking_rule": STAGE_A_RANKING_RULE,
        "ranking_tie_band": STAGE_A_RANKING_TIE_BAND,
        "ranking_tie_reference": "best_macro_asr_gain_vs_score",
        "eligible_images_per_source_family": (
            request.eligible_images_per_family
        ),
        "query_budget_including_initialization": 50,
        "maximum_wall_clock_seconds": request.deadline_seconds,
        "target_calls": 0,
        "target_evaluation_available": False,
        "training_performed": False,
    }
