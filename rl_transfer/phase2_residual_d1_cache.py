"""Verified, reusable source-teacher caches for the Phase 2 D1 diagnostic."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .artifacts import exclusive_file_lock
from .imitation import BehaviorCloneStep
from .phase2_residual_d1 import (
    D1_HELDOUT_FAMILY,
    D1_SOURCE_FAMILIES,
    ResidualCacheBinding,
    validate_residual_cache_binding,
    validate_source_only_payload,
)
from .phase2_residual_d1_cache_models import (
    ResidualTeacherCache,
    ResidualTeacherCachePaths,
    _canonical,
    _digest,
    _mapping,
    _thaw,
)


RESIDUAL_TEACHER_CACHE_SCHEMA_VERSION = 3
RESIDUAL_TEACHER_CACHE_NAME = "phase2-d1-fresh-source-gradient-teacher"
RESIDUAL_TEACHER_CACHE_ROLES = ("train", "threshold_selection", "competence_gate")
_MAX_EXAMPLES_BYTES = 256 * 1024 * 1024
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_METADATA_KEYS = frozenset(
    """schema_version name binding protocol heldout_family source_families roles
    examples_sha256 cache_reused target_calls target_evaluation_available
    hidden_target_calls hidden_target_evaluation_performed
    authorizes_hidden_target_evaluation""".split()
)
_PROTOCOL_KEYS = frozenset(
    """schema request_sha256 code_digest train_decisions validation_decisions bc_epochs
    soft_temperature prior_temperature operator_digest role_indices_sha256
    teacher_victim_ids evaluation_victim_ids victim_cache_digest""".split()
)
_ROLE_INDEX_KEYS = frozenset(
    (
        *RESIDUAL_TEACHER_CACHE_ROLES,
        "source_holdout_evaluation",
        "source_ppo_evaluation",
    )
)
_ROLE_METRIC_KEYS = frozenset(
    """role episodes decisions_per_episode steps accepted_steps source_calls
    gradient_evaluations scheduled_episodes_by_family source_calls_by_family
    scheduled_episodes_by_victim source_calls_by_victim victim_diagnostics
    target_calls hidden_target_calls hidden_target_evaluation_performed
    authorizes_hidden_target_evaluation""".split()
)
_RECORD_KEYS = frozenset(
    """role source_family observation action accepted trajectory_id step_index
    action_distribution target_calls hidden_target_calls
    hidden_target_evaluation_performed authorizes_hidden_target_evaluation""".split()
)


def _exact(value: Mapping[str, object], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} does not match the exact cache schema")


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _positive(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be positive and finite")
    return float(value)


def _victim_ids(
    value: object, families: tuple[str, ...], label: str
) -> dict[str, list[str]]:
    raw = _mapping(value, label)
    if set(raw) != set(families):
        raise ValueError(f"{label} must contain exactly the source families")
    result: dict[str, list[str]] = {}
    flattened: list[str] = []
    for family in families:
        identifiers = raw[family]
        if (
            not isinstance(identifiers, list)
            or not identifiers
            or any(not isinstance(item, str) or not item for item in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError(f"{label} contains invalid victim IDs")
        result[family] = list(identifiers)
        flattened.extend(identifiers)
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"{label} victim IDs must be globally unique")
    return result


def _protocol(value: object, families: tuple[str, ...]) -> dict[str, object]:
    result = _mapping(value, "teacher cache protocol")
    _exact(result, _PROTOCOL_KEYS, "teacher cache protocol")
    if result["schema"] != "phase2-d1-residual-teacher-v2":
        raise ValueError("teacher cache protocol schema is unsupported")
    for key in (
        "request_sha256",
        "code_digest",
        "operator_digest",
        "victim_cache_digest",
    ):
        _digest(result[key], f"protocol {key}")
    for key in ("train_decisions", "validation_decisions", "bc_epochs"):
        _integer(result[key], f"protocol {key}", 1)
    for key in ("soft_temperature", "prior_temperature"):
        _positive(result[key], f"protocol {key}")
    indices = _mapping(result["role_indices_sha256"], "protocol role indices")
    _exact(indices, _ROLE_INDEX_KEYS, "protocol role indices")
    for role, identity in indices.items():
        _digest(identity, f"protocol {role} indices")
    teachers = _victim_ids(
        result["teacher_victim_ids"], families, "protocol teacher victims"
    )
    evaluations = _victim_ids(
        result["evaluation_victim_ids"], families, "protocol evaluation victims"
    )
    teacher_set = {item for values in teachers.values() for item in values}
    evaluation_set = {item for values in evaluations.values() for item in values}
    if teacher_set & evaluation_set:
        raise ValueError("teacher and evaluation victim identities overlap")
    return result


def residual_teacher_protocol_digest(protocol: Mapping[str, object]) -> str:
    """Return the canonical protocol identity stored in the cache binding."""

    value = _protocol(protocol, D1_SOURCE_FAMILIES)
    return hashlib.sha256(_canonical(value, "teacher cache protocol")).hexdigest()


def _trajectory_family(trajectory_id: str) -> str:
    marker = "bc-gradient-source:"
    if trajectory_id.count(marker) != 1:
        raise ValueError("teacher cache trajectory lacks source-family provenance")
    family, separator, remainder = trajectory_id.split(marker, 1)[1].partition(":")
    if not separator or not remainder or family not in D1_SOURCE_FAMILIES:
        raise ValueError(
            "teacher cache trajectory has invalid source-family provenance"
        )
    return family


def _step_record(step: BehaviorCloneStep, role: str) -> dict[str, object]:
    return {
        "role": role,
        "source_family": _trajectory_family(step.trajectory_id),
        "observation": list(step.observation),
        "action": step.action,
        "accepted": step.accepted,
        "trajectory_id": step.trajectory_id,
        "step_index": step.step_index,
        "action_distribution": list(step.action_distribution or ()),
        "target_calls": 0,
        "hidden_target_calls": 0,
        "hidden_target_evaluation_performed": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _parse_record(
    value: object, action_dim: int, observation_dim: int
) -> tuple[str, BehaviorCloneStep]:
    record = _mapping(value, "teacher cache record")
    _exact(record, _RECORD_KEYS, "teacher cache record")
    validate_source_only_payload(record, "teacher cache record")
    role, family = record["role"], record["source_family"]
    if not isinstance(role, str) or role not in RESIDUAL_TEACHER_CACHE_ROLES:
        raise ValueError("teacher cache record has an invalid role")
    if (
        not isinstance(family, str)
        or family not in D1_SOURCE_FAMILIES
        or family == D1_HELDOUT_FAMILY
    ):
        raise ValueError("teacher cache record is not source-family only")
    trajectory = record["trajectory_id"]
    if (
        not isinstance(trajectory, str)
        or not trajectory.startswith(f"d1-{role}-block-")
        or _trajectory_family(trajectory) != family
    ):
        raise ValueError("teacher cache role or source provenance mismatch")
    observation = record["observation"]
    if (
        not isinstance(observation, list)
        or len(observation) != observation_dim
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in observation
        )
    ):
        raise ValueError("teacher cache observation is invalid or non-finite")
    action = _integer(record["action"], "teacher action")
    if action >= action_dim:
        raise ValueError("teacher cache action is outside the action bounds")
    if not isinstance(record["accepted"], bool):
        raise ValueError("teacher cache accepted flag must be boolean")
    step_index = _integer(record["step_index"], "teacher step index")
    distribution = record["action_distribution"]
    if (
        not isinstance(distribution, list)
        or len(distribution) != action_dim
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0
            for item in distribution
        )
    ):
        raise ValueError("teacher cache soft distribution is invalid")
    total = math.fsum(float(item) for item in distribution)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("teacher cache soft distribution must sum to one")
    if float(distribution[action]) < max(map(float, distribution)) - 1e-12:
        raise ValueError("teacher action must maximize the soft distribution")
    if record["target_calls"] != 0 or isinstance(record["target_calls"], bool):
        raise ValueError("teacher cache must record zero target calls")
    if (
        record["hidden_target_calls"] != 0
        or isinstance(record["hidden_target_calls"], bool)
        or record["hidden_target_evaluation_performed"] is not False
        or record["authorizes_hidden_target_evaluation"] is not False
    ):
        raise ValueError("teacher cache must carry the hidden-target zero seal")
    return role, BehaviorCloneStep(
        observation,
        action,
        record["accepted"],
        trajectory_id=trajectory,
        step_index=step_index,
        action_distribution=distribution,
    )


def _int_map(value: object, keys: Sequence[str], label: str) -> dict[str, int]:
    raw = _mapping(value, label)
    if set(raw) != set(keys):
        raise ValueError(f"{label} has the wrong identities")
    return {key: _integer(raw[key], f"{label} {key}") for key in keys}


def _metrics(
    value: object,
    role: str,
    steps: Sequence[BehaviorCloneStep],
    protocol: Mapping[str, object],
    families: tuple[str, ...],
) -> dict[str, object]:
    result = _mapping(value, f"{role} metrics")
    _exact(result, _ROLE_METRIC_KEYS, f"{role} metrics")
    validate_source_only_payload(result, f"{role} metrics")
    if result["role"] != role:
        raise ValueError(f"{role} metrics contain the wrong role")
    episodes = _integer(result["episodes"], f"{role} episodes", 1)
    decisions = _integer(result["decisions_per_episode"], f"{role} decisions", 1)
    expected_decisions = int(
        protocol["train_decisions" if role == "train" else "validation_decisions"]
    )
    if decisions != expected_decisions:
        raise ValueError(f"{role} decision count violates the protocol")
    count = _integer(result["steps"], f"{role} steps")
    accepted = _integer(result["accepted_steps"], f"{role} accepted steps")
    calls = _integer(result["source_calls"], f"{role} source calls")
    gradients = _integer(result["gradient_evaluations"], f"{role} gradients")
    if (
        count != len(steps)
        or accepted != sum(step.accepted for step in steps)
        or gradients != len(steps)
        or count > episodes * decisions
    ):
        raise ValueError(f"{role} cache step or accepted count mismatch")
    if result["target_calls"] != 0 or isinstance(result["target_calls"], bool):
        raise ValueError(f"{role} metrics must record zero target calls")
    if (
        result["hidden_target_calls"] != 0
        or isinstance(result["hidden_target_calls"], bool)
        or result["hidden_target_evaluation_performed"] is not False
        or result["authorizes_hidden_target_evaluation"] is not False
    ):
        raise ValueError(f"{role} metrics must carry the hidden-target zero seal")
    teachers = _victim_ids(protocol["teacher_victim_ids"], families, "teacher victims")
    victim_ids = tuple(item for family in families for item in teachers[family])
    steps_by_family = {
        family: sum(_trajectory_family(step.trajectory_id) == family for step in steps)
        for family in families
    }
    scheduled_families = _int_map(
        result["scheduled_episodes_by_family"], families, f"{role} scheduled family"
    )
    calls_families = _int_map(
        result["source_calls_by_family"], families, f"{role} calls family"
    )
    scheduled_victims = _int_map(
        result["scheduled_episodes_by_victim"], victim_ids, f"{role} scheduled victim"
    )
    calls_victims = _int_map(
        result["source_calls_by_victim"], victim_ids, f"{role} calls victim"
    )
    if (
        sum(scheduled_families.values()) != episodes
        or sum(scheduled_victims.values()) != episodes
        or sum(calls_families.values()) != calls
        or sum(calls_victims.values()) != calls
        or any(
            scheduled_families[family]
            != sum(scheduled_victims[victim] for victim in teachers[family])
            or calls_families[family]
            != sum(calls_victims[victim] for victim in teachers[family])
            or calls_families[family]
            != scheduled_families[family] + steps_by_family[family]
            for family in families
        )
    ):
        raise ValueError(f"{role} source-family or victim count totals mismatch")
    diagnostics = _mapping(result["victim_diagnostics"], f"{role} diagnostics")
    if set(diagnostics) != set(victim_ids):
        raise ValueError(f"{role} victim diagnostic identities mismatch")
    for victim_id in victim_ids:
        item = _mapping(diagnostics[victim_id], f"{role} victim diagnostic")
        if set(item) != {"scheduled_episodes", "source_calls"}:
            raise ValueError(f"{role} victim diagnostic schema mismatch")
        if (
            _integer(item["scheduled_episodes"], f"{role} victim episodes")
            != scheduled_victims[victim_id]
            or _integer(item["source_calls"], f"{role} victim calls")
            != calls_victims[victim_id]
        ):
            raise ValueError(f"{role} victim diagnostic count mismatch")
    return result


def _cache_records(
    cache: ResidualTeacherCache, action_dim: int, observation_dim: int
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, dict[str, object]]]:
    _integer(action_dim, "teacher cache action dimension", 1)
    _integer(observation_dim, "teacher cache observation dimension", 1)
    if (
        cache.heldout_family != D1_HELDOUT_FAMILY
        or cache.source_families != D1_SOURCE_FAMILIES
    ):
        raise ValueError("teacher cache violates the source-family-only seal")
    protocol = _protocol(cache.protocol, cache.source_families)
    if cache.binding.victim_cache_digest != protocol[
        "victim_cache_digest"
    ] or cache.binding.request_sha256 != residual_teacher_protocol_digest(protocol):
        raise ValueError("teacher cache binding does not match its protocol")
    raw_metrics = _mapping(cache.role_metrics, "teacher cache role metrics")
    _exact(raw_metrics, frozenset(RESIDUAL_TEACHER_CACHE_ROLES), "teacher cache roles")
    records, seen, normalized = [], set(), {}
    for role, steps in cache.steps_by_role().items():
        for step in steps:
            record = _step_record(step, role)
            parsed_role, parsed = _parse_record(record, action_dim, observation_dim)
            identity = parsed.trajectory_id, parsed.step_index
            if identity in seen:
                raise ValueError(
                    "teacher cache has a duplicate trajectory-step identity"
                )
            seen.add(identity)
            if parsed_role != role:
                raise ValueError("teacher cache record role mismatch")
            records.append(record)
        normalized[role] = _metrics(
            raw_metrics[role], role, steps, protocol, cache.source_families
        )
    return records, protocol, normalized


def _examples(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        _canonical(record, "teacher cache record") + b"\n" for record in records
    )


def _metadata(
    cache: ResidualTeacherCache,
    protocol: Mapping[str, object],
    role_metrics: Mapping[str, Mapping[str, object]],
    examples_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": RESIDUAL_TEACHER_CACHE_SCHEMA_VERSION,
        "name": RESIDUAL_TEACHER_CACHE_NAME,
        "binding": asdict(cache.binding),
        "protocol": _thaw(protocol),
        "heldout_family": cache.heldout_family,
        "source_families": list(cache.source_families),
        "roles": _thaw(role_metrics),
        "examples_sha256": examples_sha256,
        "cache_reused": False,
        "target_calls": 0,
        "target_evaluation_available": False,
        "hidden_target_calls": 0,
        "hidden_target_evaluation_performed": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _metadata_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            _thaw(value), indent=2, sort_keys=True, allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("teacher cache metadata is not finite JSON") from error


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _state(paths: ResidualTeacherCachePaths) -> str:
    present = tuple(path.exists() for path in paths.artifacts)
    return "absent" if not any(present) else "complete" if all(present) else "partial"


def _write(
    paths: ResidualTeacherCachePaths,
    cache: ResidualTeacherCache,
    action_dim: int,
    observation_dim: int,
) -> ResidualTeacherCache:
    state = _state(paths)
    if state == "partial":
        raise ValueError("teacher cache artifact pair is partial or incomplete")
    if state == "complete":
        raise ValueError("teacher cache already exists; verify and reuse it")
    records, protocol, metrics = _cache_records(cache, action_dim, observation_dim)
    examples = _examples(records)
    examples_sha = hashlib.sha256(examples).hexdigest()
    metadata = _metadata_bytes(_metadata(cache, protocol, metrics, examples_sha))
    metadata_sha = hashlib.sha256(metadata).hexdigest()
    written: list[Path] = []
    try:
        payloads = (
            (paths.examples, examples),
            (paths.examples_checksum, f"{examples_sha}\n".encode()),
            (paths.metadata, metadata),
            (paths.metadata_checksum, f"{metadata_sha}\n".encode()),
        )
        for path, content in payloads:
            _atomic_write(path, content)
            written.append(path)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return replace(
        cache,
        examples_sha256=examples_sha,
        metadata_sha256=metadata_sha,
        reused=False,
    )


def write_residual_teacher_cache(
    root: Path,
    cache: ResidualTeacherCache,
    *,
    action_dim: int,
    observation_dim: int,
) -> ResidualTeacherCache:
    """Atomically create a cache, refusing overwrite or partial state."""

    if not isinstance(cache, ResidualTeacherCache):
        raise TypeError("cache must be ResidualTeacherCache")
    paths = ResidualTeacherCachePaths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(paths.lock):
        return _write(paths, cache, action_dim, observation_dim)


def _verified(
    path: Path, sidecar: Path, label: str, maximum_bytes: int
) -> tuple[bytes, str]:
    if (
        not path.is_file()
        or path.is_symlink()
        or not sidecar.is_file()
        or sidecar.is_symlink()
        or not 0 <= path.stat().st_size <= maximum_bytes
    ):
        raise ValueError(f"{label} cache artifact is incomplete or unsafe")
    checksum = sidecar.read_bytes()
    if len(checksum) != 65 or checksum[-1:] != b"\n":
        raise ValueError(f"{label} checksum sidecar is malformed")
    try:
        expected = checksum[:-1].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} checksum sidecar is malformed") from error
    _digest(expected, f"{label} checksum")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"{label} cache checksum verification failed")
    return content, expected


def _load_metadata(
    content: bytes,
    expected_binding: ResidualCacheBinding,
    expected_protocol: Mapping[str, object],
    examples_sha: str,
) -> tuple[ResidualCacheBinding, dict[str, object], dict[str, object]]:
    try:
        decoded = json.loads(content.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("teacher cache metadata is invalid JSON") from error
    metadata = _mapping(decoded, "teacher cache metadata")
    _exact(metadata, _METADATA_KEYS, "teacher cache metadata")
    validate_source_only_payload(metadata, "teacher cache metadata")
    source_families = metadata["source_families"]
    if (
        metadata["schema_version"] != RESIDUAL_TEACHER_CACHE_SCHEMA_VERSION
        or metadata["name"] != RESIDUAL_TEACHER_CACHE_NAME
        or metadata["heldout_family"] != D1_HELDOUT_FAMILY
        or not isinstance(source_families, list)
        or tuple(source_families) != D1_SOURCE_FAMILIES
        or metadata["cache_reused"] is not False
        or isinstance(metadata["target_calls"], bool)
        or metadata["target_calls"] != 0
        or metadata["target_evaluation_available"] is not False
        or isinstance(metadata["hidden_target_calls"], bool)
        or metadata["hidden_target_calls"] != 0
        or metadata["hidden_target_evaluation_performed"] is not False
        or metadata["authorizes_hidden_target_evaluation"] is not False
    ):
        raise ValueError("teacher cache metadata violates the source-only schema")
    if _digest(metadata["examples_sha256"], "teacher examples") != examples_sha:
        raise ValueError("teacher examples checksum does not match metadata")
    binding_raw = _mapping(metadata["binding"], "teacher cache binding")
    _exact(
        binding_raw,
        frozenset(ResidualCacheBinding.__dataclass_fields__),
        "teacher cache binding",
    )
    try:
        binding = ResidualCacheBinding(**binding_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("teacher cache binding schema is invalid") from error
    validate_residual_cache_binding(expected_binding, binding)
    protocol = _protocol(metadata["protocol"], D1_SOURCE_FAMILIES)
    expected = _protocol(expected_protocol, D1_SOURCE_FAMILIES)
    if _canonical(protocol, "teacher protocol") != _canonical(
        expected, "expected protocol"
    ):
        raise ValueError("teacher cache protocol mismatch")
    if binding.victim_cache_digest != protocol[
        "victim_cache_digest"
    ] or binding.request_sha256 != residual_teacher_protocol_digest(protocol):
        raise ValueError("teacher cache binding and protocol identity mismatch")
    roles = _mapping(metadata["roles"], "teacher cache roles")
    _exact(roles, frozenset(RESIDUAL_TEACHER_CACHE_ROLES), "teacher cache roles")
    return binding, protocol, roles


def _load_examples(
    content: bytes, action_dim: int, observation_dim: int
) -> dict[str, tuple[BehaviorCloneStep, ...]]:
    if not content or not content.endswith(b"\n"):
        raise ValueError("teacher cache JSONL is not canonical")
    grouped: dict[str, list[BehaviorCloneStep]] = {
        role: [] for role in RESIDUAL_TEACHER_CACHE_ROLES
    }
    seen: set[tuple[str, int]] = set()
    prior_role = 0
    for line in content.splitlines(keepends=True):
        try:
            decoded = json.loads(line.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("teacher cache JSONL contains invalid JSON") from error
        if line != _canonical(decoded, "teacher cache record") + b"\n":
            raise ValueError("teacher cache JSONL is not canonical")
        role, step = _parse_record(decoded, action_dim, observation_dim)
        role_index = RESIDUAL_TEACHER_CACHE_ROLES.index(role)
        if role_index < prior_role:
            raise ValueError("teacher cache role blocks are out of order")
        prior_role = role_index
        identity = step.trajectory_id, step.step_index
        if identity in seen:
            raise ValueError("teacher cache has a duplicate trajectory-step identity")
        seen.add(identity)
        grouped[role].append(step)
    return {role: tuple(grouped[role]) for role in RESIDUAL_TEACHER_CACHE_ROLES}


def _load(
    paths: ResidualTeacherCachePaths,
    expected_binding: ResidualCacheBinding,
    expected_protocol: Mapping[str, object],
    action_dim: int,
    observation_dim: int,
) -> ResidualTeacherCache:
    state = _state(paths)
    if state == "absent":
        raise FileNotFoundError("teacher cache does not exist")
    if state == "partial":
        raise ValueError("teacher cache artifact pair is partial or incomplete")
    examples, examples_sha = _verified(
        paths.examples, paths.examples_checksum, "teacher examples", _MAX_EXAMPLES_BYTES
    )
    metadata, metadata_sha = _verified(
        paths.metadata, paths.metadata_checksum, "teacher metadata", _MAX_METADATA_BYTES
    )
    binding, protocol, metrics = _load_metadata(
        metadata, expected_binding, expected_protocol, examples_sha
    )
    grouped = _load_examples(examples, action_dim, observation_dim)
    for role in RESIDUAL_TEACHER_CACHE_ROLES:
        _metrics(metrics[role], role, grouped[role], protocol, D1_SOURCE_FAMILIES)
    return ResidualTeacherCache(
        binding=binding,
        protocol=protocol,
        heldout_family=D1_HELDOUT_FAMILY,
        source_families=D1_SOURCE_FAMILIES,
        train_steps=grouped["train"],
        threshold_steps=grouped["threshold_selection"],
        competence_steps=grouped["competence_gate"],
        role_metrics=metrics,
        examples_sha256=examples_sha,
        metadata_sha256=metadata_sha,
        reused=True,
    )


def load_residual_teacher_cache(
    root: Path,
    *,
    expected_binding: ResidualCacheBinding,
    expected_protocol: Mapping[str, object],
    action_dim: int,
    observation_dim: int,
) -> ResidualTeacherCache:
    """Load only a complete, checksum-bound, exact-schema source cache."""

    if not isinstance(expected_binding, ResidualCacheBinding):
        raise TypeError("expected binding must be ResidualCacheBinding")
    return _load(
        ResidualTeacherCachePaths(root),
        expected_binding,
        expected_protocol,
        action_dim,
        observation_dim,
    )


def load_or_create_residual_teacher_cache(
    root: Path,
    *,
    expected_binding: ResidualCacheBinding,
    expected_protocol: Mapping[str, object],
    action_dim: int,
    observation_dim: int,
    create: Callable[[], ResidualTeacherCache],
) -> ResidualTeacherCache:
    """Reuse a verified cache, or create it once under an exclusive lock."""

    if not isinstance(expected_binding, ResidualCacheBinding):
        raise TypeError("expected binding must be ResidualCacheBinding")
    if not callable(create):
        raise TypeError("teacher cache factory must be callable")
    paths = ResidualTeacherCachePaths(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(paths.lock):
        state = _state(paths)
        if state == "partial":
            raise ValueError("teacher cache artifact pair is partial or incomplete")
        if state == "complete":
            return _load(
                paths, expected_binding, expected_protocol, action_dim, observation_dim
            )
        created = create()
        if not isinstance(created, ResidualTeacherCache):
            raise TypeError("teacher cache factory must return ResidualTeacherCache")
        validate_residual_cache_binding(expected_binding, created.binding)
        expected = _protocol(expected_protocol, D1_SOURCE_FAMILIES)
        actual = _protocol(created.protocol, D1_SOURCE_FAMILIES)
        if _canonical(expected, "expected protocol") != _canonical(
            actual, "created protocol"
        ):
            raise ValueError("created teacher cache protocol mismatch")
        return _write(paths, created, action_dim, observation_dim)
