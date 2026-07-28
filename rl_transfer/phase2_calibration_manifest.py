"""Verified inputs for the frozen Phase 2 calibration diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Mapping

from .artifacts import sha256_file
from .cifar_config import MacPilotConfig
from .paths import resolve_descendant
from .phase2_validation import validate_source_run_artifacts
from .verified_artifacts import load_verified_json


FOLDS = ("classical_cnn", "modern_cnn", "transformer")
CALIBRATION_TEMPERATURES = (0.25, 0.5, 0.75, 1.0, 1.5)
CALIBRATION_POLICY_SEEDS = (17,)
CALIBRATION_MAX_SECONDS = 900.0
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Phase2CalibrationRequest:
    """Locked source-only replay request for completed Phase 2 policies."""

    source_manifest: Path
    source_root: Path
    output_dir: Path
    data_root: Path
    seeds: tuple[int, ...] = CALIBRATION_POLICY_SEEDS
    folds: tuple[str, ...] = FOLDS
    temperatures: tuple[float, ...] = CALIBRATION_TEMPERATURES
    deadline_seconds: float = CALIBRATION_MAX_SECONDS
    device: str = "cuda"
    download: bool = False

    def __post_init__(self) -> None:
        source_root = Path(self.source_root).resolve()
        source_manifest = resolve_descendant(
            source_root,
            Path(self.source_manifest),
            label="Phase 2 source manifest",
        )
        output_dir = Path(self.output_dir).resolve()
        data_root = Path(self.data_root).resolve()
        seeds = tuple(self.seeds)
        folds = tuple(self.folds)
        temperatures = tuple(float(value) for value in self.temperatures)

        if seeds != CALIBRATION_POLICY_SEEDS:
            raise ValueError("calibration is locked to policy seed 17")
        if folds != FOLDS:
            raise ValueError("calibration requires all three ordered folds")
        if temperatures != CALIBRATION_TEMPERATURES:
            raise ValueError("calibration temperatures are locked")
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or not 0 < float(self.deadline_seconds) <= CALIBRATION_MAX_SECONDS
        ):
            raise ValueError("calibration deadline must be in (0, 900] seconds")
        if self.device != "cuda":
            raise ValueError("calibration requires the designated CUDA device")
        if not isinstance(self.download, bool):
            raise ValueError("download must be boolean")
        if (
            output_dir == source_root
            or source_root in output_dir.parents
            or output_dir in source_root.parents
        ):
            raise ValueError(
                "calibration output must not overlap the Phase 2 source tree"
            )
        if self.download and (
            data_root == source_root
            or source_root in data_root.parents
            or data_root in source_root.parents
            or data_root == output_dir
            or output_dir in data_root.parents
            or data_root in output_dir.parents
        ):
            raise ValueError(
                "download data root must not overlap sealed artifact trees"
            )

        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "source_manifest", source_manifest)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "data_root", data_root)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(
            self,
            "deadline_seconds",
            float(self.deadline_seconds),
        )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _portable_run_directory(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Phase 2 run directory is missing")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.name != value:
        raise ValueError("Phase 2 run directory must be a portable name")
    return path


def _content_digest(dataset_version: str) -> str | None:
    match = re.search(r"(?:^|;)content-sha256=([0-9a-f]{64})(?:;|$)", dataset_version)
    return match.group(1) if match is not None else None


def load_phase2_calibration_source(
    request: Phase2CalibrationRequest,
) -> dict[str, object]:
    """Verify completed Phase 2 evidence and expose source-only fold records."""

    study = load_verified_json(request.source_manifest)
    if (
        study.get("schema_version") != 1
        or study.get("status") != "screen_complete"
        or study.get("research_valid") is not False
        or study.get("target_calls") != 0
        or study.get("target_evaluation_performed") is not False
    ):
        raise ValueError("Phase 2 source manifest violates the calibration seal")
    dataset_version = study.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("Phase 2 dataset version is missing")
    raw_runs = study.get("source_runs")
    if not isinstance(raw_runs, list):
        raise ValueError("Phase 2 source runs are missing")

    indexed: dict[tuple[int, str], Mapping[str, object]] = {}
    for raw_run in raw_runs:
        run = _mapping(raw_run, label="Phase 2 source run")
        seed = run.get("seed")
        heldout = run.get("target_family")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(heldout, str)
            or (seed, heldout) in indexed
        ):
            raise ValueError("Phase 2 source grid is invalid")
        indexed[(seed, heldout)] = run

    runs_root = resolve_descendant(
        request.source_root,
        "runs",
        label="Phase 2 runs root",
    )
    selected: list[dict[str, object]] = []
    for seed in request.seeds:
        for heldout in request.folds:
            run = indexed.get((seed, heldout))
            if run is None:
                raise ValueError("Phase 2 source grid is incomplete")
            config = MacPilotConfig(
                **dict(
                    _mapping(
                        run.get("config"),
                        label="Phase 2 run config",
                    )
                )
            )
            validate_source_run_artifacts(
                run,
                derived_config=config,
                run_output_dir=runs_root,
            )
            run_dir = resolve_descendant(
                runs_root,
                _portable_run_directory(run.get("run_dir")),
                label="Phase 2 source run",
            )
            policy = _mapping(run.get("policy"), label="Phase 2 policy")
            checkpoint_name = policy.get("checkpoint")
            if (
                not isinstance(checkpoint_name, str)
                or Path(checkpoint_name).name != checkpoint_name
            ):
                raise ValueError("Phase 2 policy path is invalid")
            checkpoint_path = resolve_descendant(
                run_dir,
                checkpoint_name,
                label="Phase 2 policy checkpoint",
            )
            checkpoint_digest = policy.get("checkpoint_sha256")
            persistent_digest = policy.get("persistent_digest")
            if (
                not isinstance(checkpoint_digest, str)
                or _DIGEST.fullmatch(checkpoint_digest) is None
                or sha256_file(checkpoint_path) != checkpoint_digest
                or not isinstance(persistent_digest, str)
                or _DIGEST.fullmatch(persistent_digest) is None
            ):
                raise ValueError("Phase 2 policy identity is invalid")
            source_families = tuple(run.get("source_families", ()))
            if (
                len(source_families) != 2
                or heldout in source_families
                or set(source_families) != set(FOLDS) - {heldout}
            ):
                raise ValueError("Phase 2 source-family seal is invalid")
            selected.append(
                {
                    "seed": seed,
                    "heldout_family": heldout,
                    "source_families": source_families,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_sha256": checkpoint_digest,
                    "persistent_digest": persistent_digest,
                    "score_rows_path": run_dir / "source_results.jsonl",
                    "run_dir": run_dir,
                    "runs_root": runs_root,
                    "run_manifest": dict(run),
                }
            )

    return {
        "manifest_path": request.source_manifest,
        "manifest_sha256": sha256_file(request.source_manifest),
        "dataset_version": dataset_version,
        "dataset_content_sha256": _content_digest(dataset_version),
        "target_calls": 0,
        "target_evaluation_performed": False,
        "folds": tuple(selected),
    }


__all__ = (
    "CALIBRATION_MAX_SECONDS",
    "CALIBRATION_POLICY_SEEDS",
    "CALIBRATION_TEMPERATURES",
    "FOLDS",
    "Phase2CalibrationRequest",
    "load_phase2_calibration_source",
)
