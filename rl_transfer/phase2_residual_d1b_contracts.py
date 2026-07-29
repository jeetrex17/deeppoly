"""Immutable contracts and validation helpers for source-only D1b."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType

from .phase2_residual_d1 import (
    D1_HELDOUT_FAMILY,
    D1_MAX_PPO_EPISODES,
    D1_SOURCE_FAMILIES,
    validate_source_only_payload as _source_only,
)


D1B_BLOCK_EPISODES = 50
D1B_TOTAL_EPISODES = D1_MAX_PPO_EPISODES
D1B_BLOCK_ENDPOINTS = (50, 100, 150, 200)
D1B_METHODS = (
    "score_greedy",
    "residual_ranker_bc",
    "residual_ranker_bc_ppo",
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")


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
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error


def _sha(value: object, label: str) -> str:
    return hashlib.sha256(_canonical(value, label)).hexdigest()


def _checkpoint_id(policy_digest: str, metadata_digest: str) -> str:
    return f"d1b-{policy_digest}-{metadata_digest}"


def _object(value: object, label: str) -> dict[str, object]:
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


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _zero(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
        raise ValueError(f"{label} must be zero for source-only D1b")


def _false(value: object, label: str) -> None:
    if value is not False:
        raise ValueError(f"{label} must be false for source-only D1b")


def _seal() -> dict[str, object]:
    return {
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


@dataclass(frozen=True)
class ResidualD1BTrainingPayload:
    source_victims: object
    source_samples: object
    attack_config: object

    def __post_init__(self) -> None:
        if isinstance(self.source_victims, Mapping) and set(self.source_victims) != set(
            D1_SOURCE_FAMILIES
        ):
            raise ValueError("D1b training payload contains non-source victims")
        _source_only(self.source_samples, "D1b PPO samples")
        _source_only(self.attack_config, "D1b PPO attack config")


@dataclass(frozen=True)
class ResidualD1BSourceRole:
    name: str
    sample_ids: tuple[int, ...]
    payload: object
    source_families: tuple[str, ...] = D1_SOURCE_FAMILIES
    hidden_target_calls: int = 0

    def __post_init__(self) -> None:
        _zero(self.hidden_target_calls, f"D1b {self.name} hidden target calls")
        sizes = {
            "ppo_training": 200,
            "threshold_selection": 50,
            "competence_gate": 50,
            "d1b_evaluation": 50,
        }
        expected = sizes.get(self.name)
        try:
            sample_ids = tuple(self.sample_ids)
            families = tuple(self.source_families)
        except TypeError as error:
            raise TypeError("D1b source role identities must be sequences") from error
        bad_ids = any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in sample_ids
        )
        if (
            expected is None
            or len(sample_ids) != expected
            or bad_ids
            or len(sample_ids) != len(set(sample_ids))
        ):
            raise ValueError(f"D1b {self.name} role has invalid IDs or size")
        if families != D1_SOURCE_FAMILIES:
            raise ValueError("D1b roles accept only locked source families")
        _source_only(self.payload, f"D1b {self.name} payload")
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "source_families", families)

    @property
    def sample_ids_sha256(self) -> str:
        return _sha(self.sample_ids, f"D1b {self.name} IDs")


@dataclass(frozen=True)
class ResidualD1BSourceRoles:
    ppo_training: ResidualD1BSourceRole
    threshold_selection: ResidualD1BSourceRole
    competence_gate: ResidualD1BSourceRole
    evaluation: ResidualD1BSourceRole

    def __post_init__(self) -> None:
        roles = self.as_tuple
        if any(not isinstance(role, ResidualD1BSourceRole) for role in roles):
            raise TypeError("D1b roles must use ResidualD1BSourceRole")
        if tuple(role.name for role in roles) != (
            "ppo_training",
            "threshold_selection",
            "competence_gate",
            "d1b_evaluation",
        ):
            raise ValueError("D1b source roles do not match the protocol")
        if any(
            set(left.sample_ids) & set(right.sample_ids)
            for offset, left in enumerate(roles)
            for right in roles[offset + 1 :]
        ):
            raise ValueError("D1b source roles must be pairwise disjoint")

    @property
    def as_tuple(self) -> tuple[ResidualD1BSourceRole, ...]:
        return (
            self.ppo_training,
            self.threshold_selection,
            self.competence_gate,
            self.evaluation,
        )

    @property
    def d1b_evaluation(self) -> ResidualD1BSourceRole:
        return self.evaluation

    @property
    def digest(self) -> str:
        return _sha(
            {
                role.name: {
                    "sample_ids": role.sample_ids,
                    "source_families": role.source_families,
                }
                for role in self.as_tuple
            },
            "D1b source roles",
        )


@dataclass(frozen=True)
class VerifiedD1AArtifacts:
    bc_policy: object
    manifest_digest: str
    checkpoint_sha256: str
    bc_policy_digest: str
    hidden_target_calls: int = 0

    def __post_init__(self) -> None:
        _zero(self.hidden_target_calls, "verified D1a hidden target calls")
        if self.bc_policy is None:
            raise ValueError("verified D1a BC policy cannot be absent")
        _digest(self.manifest_digest, "verified D1a manifest digest")
        _digest(self.checkpoint_sha256, "verified D1a checkpoint digest")
        _digest(self.bc_policy_digest, "verified D1a BC policy digest")


@dataclass(frozen=True)
class ResidualD1BCheckpointReceipt:
    reference: str
    policy_digest: str
    metadata_digest: str
    hidden_target_calls: int = 0

    def __post_init__(self) -> None:
        _zero(self.hidden_target_calls, "D1b receipt hidden target calls")
        if (
            not isinstance(self.reference, str)
            or ".." in self.reference
            or _REFERENCE.fullmatch(self.reference) is None
        ):
            raise ValueError("D1b checkpoint reference is not a safe opaque ID")
        _digest(self.policy_digest, "D1b checkpoint policy digest")
        _digest(self.metadata_digest, "D1b checkpoint metadata digest")


@dataclass(frozen=True)
class ResidualD1BLoadedCheckpoint:
    policy: object
    metadata: Mapping[str, object]
    hidden_target_calls: int = 0

    def __post_init__(self) -> None:
        _zero(self.hidden_target_calls, "loaded D1b hidden target calls")
        if self.policy is None:
            raise ValueError("loaded D1b policy cannot be absent")
        metadata = _object(self.metadata, "loaded D1b checkpoint metadata")
        _source_only(metadata, "loaded D1b checkpoint metadata")
        object.__setattr__(self, "metadata", _freeze(metadata))


def _family_state(
    raw_weights: object,
    raw_offsets: object,
) -> tuple[dict[str, object], dict[str, object]]:
    weights = _object(raw_weights, "D1b family weights")
    offsets = _object(raw_offsets, "D1b instance offsets")
    if set(weights) != set(D1_SOURCE_FAMILIES) or set(offsets) != set(
        D1_SOURCE_FAMILIES
    ):
        raise ValueError("D1b state must contain exactly source families")
    if (
        sum(
            _number(weights[family], f"D1b {family} weight")
            for family in D1_SOURCE_FAMILIES
        )
        <= 0
    ):
        raise ValueError("D1b family weights need positive mass")
    for family in D1_SOURCE_FAMILIES:
        _integer(offsets[family], f"D1b {family} instance offset")
    return weights, offsets


def _block_output(
    value: object,
    episode_offset: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    metrics = _object(value, "D1b PPO block metrics")
    _source_only(metrics, "D1b PPO block metrics")
    _zero(metrics.get("hidden_target_calls"), "D1b block hidden target calls")
    if (
        metrics.get("episodes") != D1B_BLOCK_EPISODES
        or metrics.get("episode_offset") != episode_offset
        or metrics.get("next_episode_offset") != episode_offset + D1B_BLOCK_EPISODES
    ):
        raise ValueError("D1b PPO metrics violate a 50-episode boundary")
    trained = _integer(metrics.get("trained_episodes"), "trained episodes")
    if trained > D1B_BLOCK_EPISODES:
        raise ValueError("D1b trained episodes exceed the block")
    _integer(metrics.get("source_calls"), "D1b source calls")
    by_family = _object(
        metrics.get("source_calls_by_family"),
        "D1b source calls by family",
    )
    if set(by_family) != set(D1_SOURCE_FAMILIES):
        raise ValueError("D1b metrics contain non-source families")
    for family in D1_SOURCE_FAMILIES:
        _integer(by_family[family], f"D1b {family} source calls")
    by_victim = _object(
        metrics.get("source_calls_by_victim"),
        "D1b source calls by victim",
    )
    for victim, calls in by_victim.items():
        normalized = victim.casefold().replace("-", "_")
        if (
            not victim
            or "target" in normalized
            or "heldout" in normalized
            or D1_HELDOUT_FAMILY in normalized
        ):
            raise ValueError("D1b PPO metrics contain a held-out/target victim")
        _integer(calls, f"D1b {victim} source calls")
    if metrics["source_calls"] != sum(
        int(item) for item in by_family.values()
    ) or metrics["source_calls"] != sum(int(item) for item in by_victim.values()):
        raise ValueError("D1b source-call totals are inconsistent")
    if "schedule" in metrics:
        schedule = metrics["schedule"]
        if not isinstance(schedule, (list, tuple)) or any(
            family not in D1_SOURCE_FAMILIES for family in schedule
        ):
            raise ValueError("D1b schedule contains a held-out family")
    weights, offsets = _family_state(
        metrics.get("family_weights"),
        metrics.get("instance_offsets"),
    )
    return metrics, weights, offsets


@dataclass(frozen=True)
class ResidualD1BBlockState:
    block_index: int
    episode_offset: int
    metrics: Mapping[str, object]
    family_weights: Mapping[str, float]
    instance_offsets: Mapping[str, int]
    policy_digest: str
    checkpoint: ResidualD1BCheckpointReceipt
    checkpoint_metadata: Mapping[str, object]

    @property
    def episodes_completed(self) -> int:
        return self.block_index * D1B_BLOCK_EPISODES


@dataclass(frozen=True)
class ResidualD1BResumeState:
    d1a_manifest_digest: str
    d1a_checkpoint_sha256: str
    bc_policy_digest: str
    source_roles_digest: str
    blocks: tuple[ResidualD1BBlockState, ...] = ()

    @property
    def completed_episodes(self) -> int:
        return len(self.blocks) * D1B_BLOCK_EPISODES


@dataclass(frozen=True)
class ResidualD1BEvaluationInputs:
    cohort: object
    sample_ids: tuple[int, ...]
    source_families: tuple[str, ...]
    methods: tuple[str, ...]
    seed: int
    prior_seed: int
    query_budget: int
    bc_policy: object
    ppo_policy: object
    bc_policy_digest: str
    ppo_policy_digest: str
    threshold_selection: Mapping[str, object]
    competence_gate: Mapping[str, object]


@dataclass(frozen=True)
class ResidualD1BResult:
    manifest: Mapping[str, object]
    resume_state: ResidualD1BResumeState | None
    evaluation_inputs: ResidualD1BEvaluationInputs | None


@dataclass(frozen=True)
class ResidualD1BDependencies:
    verify_d1a: Callable[..., object]
    clone_policy: Callable[[object], object]
    policy_digest: Callable[[object], str]
    train_ppo_block: Callable[..., object]
    save_block_checkpoint: Callable[..., object]
    load_block_checkpoint: Callable[..., object]
    select_threshold: Callable[..., object]
    apply_threshold: Callable[..., object]
    evaluate_competence: Callable[..., object]

    def __post_init__(self) -> None:
        if any(
            not callable(value)
            for value in (
                self.verify_d1a,
                self.clone_policy,
                self.policy_digest,
                self.train_ppo_block,
                self.save_block_checkpoint,
                self.load_block_checkpoint,
                self.select_threshold,
                self.apply_threshold,
                self.evaluate_competence,
            )
        ):
            raise TypeError("all D1b dependencies must be callable")
