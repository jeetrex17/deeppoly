"""Immutable data models and canonical JSON helpers for D1 teacher caches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType

from .imitation import BehaviorCloneStep
from .phase2_residual_d1 import ResidualCacheBinding


RESIDUAL_TEACHER_EXAMPLES_NAME = "teacher_ranker_examples.jsonl"
RESIDUAL_TEACHER_METADATA_NAME = "teacher_ranker_manifest.json"
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical(value: object, label: str) -> bytes:
    try:
        return json.dumps(
            _thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    decoded = json.loads(_canonical(value, label))
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must be an object")
    return decoded


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class ResidualTeacherCachePaths:
    """Fixed files forming one committed cache."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def examples(self) -> Path:
        return self.root / RESIDUAL_TEACHER_EXAMPLES_NAME

    @property
    def examples_checksum(self) -> Path:
        return self.examples.with_suffix(self.examples.suffix + ".sha256")

    @property
    def metadata(self) -> Path:
        return self.root / RESIDUAL_TEACHER_METADATA_NAME

    @property
    def metadata_checksum(self) -> Path:
        return self.metadata.with_suffix(self.metadata.suffix + ".sha256")

    @property
    def lock(self) -> Path:
        return self.root / ".teacher-cache.lock"

    @property
    def artifacts(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.examples,
            self.examples_checksum,
            self.metadata,
            self.metadata_checksum,
        )


@dataclass(frozen=True)
class ResidualTeacherCache:
    """Immutable teacher steps, role metrics, and source identity."""

    binding: ResidualCacheBinding
    protocol: Mapping[str, object]
    heldout_family: str
    source_families: tuple[str, ...]
    train_steps: tuple[BehaviorCloneStep, ...]
    threshold_steps: tuple[BehaviorCloneStep, ...]
    competence_steps: tuple[BehaviorCloneStep, ...]
    role_metrics: Mapping[str, Mapping[str, object]]
    examples_sha256: str | None = None
    metadata_sha256: str | None = None
    reused: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ResidualCacheBinding):
            raise TypeError("teacher cache requires ResidualCacheBinding")
        protocol = _mapping(self.protocol, "teacher cache protocol")
        metrics = _mapping(self.role_metrics, "teacher cache role metrics")
        try:
            families = tuple(self.source_families)
            roles = tuple(
                tuple(steps)
                for steps in (
                    self.train_steps,
                    self.threshold_steps,
                    self.competence_steps,
                )
            )
        except TypeError as error:
            raise TypeError("teacher cache roles must be finite sequences") from error
        if any(
            not isinstance(step, BehaviorCloneStep) for steps in roles for step in steps
        ):
            raise TypeError("teacher cache values must be BehaviorCloneStep")
        for label, value in (
            ("examples_sha256", self.examples_sha256),
            ("metadata_sha256", self.metadata_sha256),
        ):
            if value is not None:
                _digest(value, label)
        if not isinstance(self.reused, bool):
            raise TypeError("teacher cache reused flag must be boolean")
        object.__setattr__(self, "protocol", _freeze(protocol))
        object.__setattr__(self, "role_metrics", _freeze(metrics))
        object.__setattr__(self, "source_families", families)
        object.__setattr__(self, "train_steps", roles[0])
        object.__setattr__(self, "threshold_steps", roles[1])
        object.__setattr__(self, "competence_steps", roles[2])

    def steps_by_role(self) -> Mapping[str, tuple[BehaviorCloneStep, ...]]:
        return MappingProxyType(
            {
                "train": self.train_steps,
                "threshold_selection": self.threshold_steps,
                "competence_gate": self.competence_steps,
            }
        )
