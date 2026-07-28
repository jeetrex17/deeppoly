"""Validated contract for short, source-only Phase 2 screening."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


FAMILIES = ("classical_cnn", "modern_cnn", "transformer")


def _safe_repository_path(
    value: str,
    *,
    label: str,
    required_prefix: tuple[str, ...],
) -> None:
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[: len(required_prefix)] != required_prefix
    ):
        rendered = "/".join(required_prefix)
        raise ValueError(
            f"{label} must be a safe path within {rendered}"
        )


@dataclass(frozen=True)
class Phase2ScreenConfig:
    """Prespecified, diagnostic source screen that cannot unlock targets."""

    schema_version: int
    name: str
    research_valid: bool
    base_config: str
    output_dir: str
    device: str
    seeds: tuple[int, ...]
    target_families: tuple[str, ...]
    resume: bool
    split_seed: int
    victim_seed: int
    victim_cache_source: str
    victim_study_manifest: str
    victim_study_manifest_sha256: str
    require_verified_victim_cache: bool
    max_wall_clock_minutes: int
    estimated_minutes_per_cell: float
    minimum_mean_bc_accuracy_gain: float
    minimum_mean_bc_nll_improvement: float
    minimum_mean_score_asr_gain: float
    minimum_mean_score_auc_gain: float
    minimum_positive_condition_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(
            self,
            "target_families",
            tuple(self.target_families),
        )
        if self.schema_version != 1:
            raise ValueError("Phase 2 screen requires schema version 1")
        if self.research_valid is not False:
            raise ValueError(
                "a diagnostic source screen cannot be research-valid"
            )
        if (
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
                self.name,
            )
            is None
        ):
            raise ValueError("name must be a safe filename component")
        _safe_repository_path(
            self.base_config,
            label="base_config",
            required_prefix=("configs", "rl_transfer"),
        )
        _safe_repository_path(
            self.output_dir,
            label="output_dir",
            required_prefix=("output", "rl_transfer"),
        )
        _safe_repository_path(
            self.victim_cache_source,
            label="victim_cache_source",
            required_prefix=("output", "rl_transfer"),
        )
        _safe_repository_path(
            self.victim_study_manifest,
            label="victim_study_manifest",
            required_prefix=("output", "rl_transfer"),
        )
        if (
            re.fullmatch(
                r"[0-9a-f]{64}",
                self.victim_study_manifest_sha256,
            )
            is None
        ):
            raise ValueError(
                "victim_study_manifest_sha256 must be lowercase SHA-256"
            )
        if self.device != "cuda":
            raise ValueError("Phase 2 GPU screening requires explicit CUDA")
        if (
            not 1 <= len(self.seeds) <= 3
            or len(self.seeds) != len(set(self.seeds))
            or any(
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or seed < 0
                for seed in self.seeds
            )
        ):
            raise ValueError(
                "Phase 2 requires one to three unique non-negative seeds"
            )
        if self.target_families != FAMILIES:
            raise ValueError(
                "all three LOFO families are required in fixed order"
            )
        if self.resume is not True:
            raise ValueError("Phase 2 cells must be resumable")
        for label, value in (
            ("split_seed", self.split_seed),
            ("victim_seed", self.victim_seed),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(
                    f"{label} must be a non-negative integer"
                )
        if self.victim_seed in self.seeds:
            raise ValueError(
                "victim seed must be distinct from policy seeds"
            )
        if self.require_verified_victim_cache is not True:
            raise ValueError(
                "the verified Phase 1 victim cache is mandatory"
            )
        if (
            not isinstance(self.max_wall_clock_minutes, int)
            or isinstance(self.max_wall_clock_minutes, bool)
            or not 10 <= self.max_wall_clock_minutes <= 240
        ):
            raise ValueError(
                "wall-clock budget must be between 10 and 240 minutes"
            )
        if (
            isinstance(self.estimated_minutes_per_cell, bool)
            or not isinstance(
                self.estimated_minutes_per_cell,
                (int, float),
            )
            or not math.isfinite(self.estimated_minutes_per_cell)
            or not 1 <= self.estimated_minutes_per_cell <= 60
        ):
            raise ValueError(
                "estimated cell runtime must be between 1 and 60 minutes"
            )
        self._validate_screen_thresholds()

    def _validate_screen_thresholds(self) -> None:
        nonnegative = (
            self.minimum_mean_bc_accuracy_gain,
            self.minimum_mean_bc_nll_improvement,
            self.minimum_mean_score_asr_gain,
            self.minimum_mean_score_auc_gain,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in nonnegative
        ):
            raise ValueError(
                "screening effect thresholds must be finite in [0, 1]"
            )
        if (
            isinstance(self.minimum_positive_condition_fraction, bool)
            or not isinstance(
                self.minimum_positive_condition_fraction,
                (int, float),
            )
            or not math.isfinite(
                self.minimum_positive_condition_fraction
            )
            or not 0.5
            <= self.minimum_positive_condition_fraction
            <= 1
        ):
            raise ValueError(
                "positive-condition fraction must be in [0.5, 1]"
            )

    @classmethod
    def from_json(cls, path: Path) -> "Phase2ScreenConfig":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Phase 2 config must contain a JSON object")
        return cls(**payload)
