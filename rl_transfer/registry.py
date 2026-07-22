from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class VictimSpec:
    victim_id: str
    family: str
    checkpoint_source: str
    preprocessing: str
    clean_accuracy: float
    license: str
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.victim_id or not self.family:
            raise ValueError("victim_id and family are required")
        if not 0 <= self.clean_accuracy <= 1:
            raise ValueError("clean_accuracy must be in [0, 1]")


@dataclass(frozen=True)
class ExperimentSplit:
    source_train: tuple[str, ...]
    source_validation: tuple[str, ...]
    outer_test: tuple[str, ...]


class VictimRegistry:
    def __init__(self, victims: Iterable[VictimSpec]) -> None:
        self.victims = tuple(victims)
        ids = tuple(victim.victim_id for victim in self.victims)
        if len(ids) != len(set(ids)):
            raise ValueError("victim IDs must be unique")
        self._by_id = {victim.victim_id: victim for victim in self.victims}

    def validate_split(self, split: ExperimentSplit) -> None:
        groups = (set(split.source_train), set(split.source_validation), set(split.outer_test))
        if any(not group for group in groups):
            raise ValueError("source train, source validation, and outer test must be non-empty")
        unknown = set.union(*groups) - set(self._by_id)
        if unknown:
            raise ValueError(f"unknown victims in split: {sorted(unknown)}")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("victim IDs must be disjoint across split roles")
        train_families = {self._by_id[item].family for item in groups[0]}
        validation_families = {self._by_id[item].family for item in groups[1]}
        inner_overlap = train_families & validation_families
        if inner_overlap:
            raise ValueError(f"source-validation family leaked into source training: {sorted(inner_overlap)}")
        source_families = train_families | validation_families
        test_families = {self._by_id[item].family for item in groups[2]}
        overlap = source_families & test_families
        if overlap:
            raise ValueError(f"outer-test family leaked into source selection: {sorted(overlap)}")

    def digest(self) -> str:
        payload = json.dumps([asdict(victim) for victim in self.victims], sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, path: Path) -> "VictimRegistry":
        payload = json.loads(path.read_text())
        return cls(VictimSpec(**item) for item in payload["victims"])
