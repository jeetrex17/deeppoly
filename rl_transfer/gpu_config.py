"""Strict configuration contract for the confirmatory RTX study."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


FAMILIES = ("classical_cnn", "modern_cnn", "transformer")


def _safe_relative_path(value: str, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be a safe repository-relative path")


@dataclass(frozen=True)
class RTXPublicationConfig:
    schema_version: int
    name: str
    research_valid: bool
    base_config: str
    output_dir: str
    device: str
    seeds: tuple[int, ...]
    target_families: tuple[str, ...]
    require_source_gate: bool
    source_holdout_instances_per_family: int
    target_instances_per_family: int
    resume: bool
    split_seed: int
    victim_seed: int
    replicate_unit: str
    primary_control: str
    primary_metric: str
    require_clean_worktree: bool
    minimum_seeds: int = 10
    minimum_source_asr_gain: float = 0.05
    minimum_source_auc_gain: float = 0.02
    minimum_target_asr_gain: float = 0.03
    minimum_target_auc_gain: float = 0.01
    bootstrap_samples: int = 10_000
    permutation_samples: int = 10_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "target_families", tuple(self.target_families))
        if self.schema_version != 1:
            raise ValueError("RTX publication config requires schema version 1")
        if self.research_valid is not False:
            raise ValueError("research_valid remains false until evidence gates pass")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.name) is None:
            raise ValueError("name must be a safe filename component")
        _safe_relative_path(self.base_config, "base_config")
        _safe_relative_path(self.output_dir, "output_dir")
        if Path(self.base_config).parts[:2] != ("configs", "rl_transfer"):
            raise ValueError(
                "base_config must remain within configs/rl_transfer"
            )
        if Path(self.output_dir).parts[:2] != ("output", "rl_transfer"):
            raise ValueError(
                "output_dir must remain within output/rl_transfer"
            )
        if self.device != "cuda":
            raise ValueError("the RTX publication profile requires explicit CUDA")
        if (
            len(self.seeds) < 10
            or len(self.seeds) > 20
            or len(self.seeds) != len(set(self.seeds))
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in self.seeds
            )
        ):
            raise ValueError(
                "10 to 20 unique non-negative policy seeds are required"
            )
        if self.minimum_seeds < 10 or len(self.seeds) < self.minimum_seeds:
            raise ValueError(
                "minimum_seeds must be at least ten and fit the seed grid"
            )
        if tuple(self.target_families) != FAMILIES:
            raise ValueError("all three LOFO target families are required in fixed order")
        if self.require_source_gate is not True:
            raise ValueError("the source competence gate cannot be disabled")
        if (
            not isinstance(self.source_holdout_instances_per_family, int)
            or isinstance(self.source_holdout_instances_per_family, bool)
            or self.source_holdout_instances_per_family < 1
        ):
            raise ValueError("at least one unseen source-family instance is required")
        if (
            not isinstance(self.target_instances_per_family, int)
            or isinstance(self.target_instances_per_family, bool)
            or self.target_instances_per_family < 3
        ):
            raise ValueError(
                "at least three target instances per family are required"
            )
        if self.resume is not True:
            raise ValueError("the long RTX study must be resumable")
        for label, value in (
            ("split_seed", self.split_seed),
            ("victim_seed", self.victim_seed),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{label} must be a non-negative integer")
        if self.victim_seed in self.seeds:
            raise ValueError(
                "the fixed victim seed must be distinct from policy seeds"
            )
        if self.replicate_unit != "policy_seed":
            raise ValueError(
                "the confirmatory replicate unit must be policy_seed"
            )
        if self.primary_control != "score_greedy":
            raise ValueError(
                "the primary control must be score_greedy"
            )
        if self.primary_metric != "asr_at_50":
            raise ValueError("the primary endpoint must be ASR at 50 calls")
        if self.require_clean_worktree is not True:
            raise ValueError("the locked run requires a clean worktree")
        gains = (
            self.minimum_source_asr_gain,
            self.minimum_source_auc_gain,
            self.minimum_target_asr_gain,
            self.minimum_target_auc_gain,
        )
        if any(not math.isfinite(value) or not 0 < value <= 1 for value in gains):
            raise ValueError("practical effect thresholds must be in (0, 1]")
        if (
            not 2_000 <= self.bootstrap_samples <= 100_000
            or not 1_000 <= self.permutation_samples <= 100_000
        ):
            raise ValueError("publication statistics require enough resamples")

    @classmethod
    def from_json(cls, path: Path) -> "RTXPublicationConfig":
        return cls(**json.loads(path.read_text()))
