"""Verified policy cloning and resumable block artifacts for D1b."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import torch

from .artifacts import (
    load_recurrent_checkpoint,
    save_recurrent_checkpoint,
    sha256_file,
)
from .phase2_residual_d1 import (
    D1_SOURCE_FAMILIES,
    validate_source_only_payload as _source_only,
)
from .phase2_residual_d1b import (
    D1B_BLOCK_ENDPOINTS,
    D1B_BLOCK_EPISODES,
    ResidualD1BBlockState,
    ResidualD1BCheckpointReceipt,
    ResidualD1BLoadedCheckpoint,
    ResidualD1BResumeState,
)
from .recurrent import RecurrentAttackPolicy
from .residual_ranker import ResidualRankerPolicy
from .verified_artifacts import load_verified_json, write_verified_json


_DIGEST = re.compile(r"[0-9a-f]{64}")
_REFERENCE = re.compile(r"d1b-([0-9a-f]{64})-([0-9a-f]{64})")
_CORE_FIELDS = {
    "schema_version",
    "name",
    "block_index",
    "episode_offset",
    "episodes",
    "episodes_completed",
    "d1a_manifest_digest",
    "d1a_checkpoint_sha256",
    "bc_policy_digest",
    "source_roles_digest",
    "ppo_policy_digest",
    "ppo_metrics_digest",
    "parent_checkpoint",
    "family_weights",
    "instance_offsets",
    "target_calls",
    "hidden_target_calls",
    "target_evaluation_performed",
    "hidden_target_evaluation_performed",
    "target_evaluation_available",
    "authorizes_hidden_target_evaluation",
}


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _plain(value: object, label: str = "JSON value") -> object:
    try:
        return json.loads(
            json.dumps(
                _thaw(value),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error


def canonical_json_digest(value: object) -> str:
    """Return the compact canonical JSON SHA-256 used by D1b receipts."""

    encoded = json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        or (positive and float(value) <= 0)
    ):
        raise ValueError(f"{label} must be finite and valid")
    return float(value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    decoded = _plain(value, label)
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must be an object")
    return decoded


def _seal() -> dict[str, object]:
    return {
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


def clone_residual_policy(policy: ResidualRankerPolicy) -> ResidualRankerPolicy:
    """Clone model and optimizer state without sharing mutable policy state."""

    if not isinstance(policy, ResidualRankerPolicy):
        raise TypeError("D1b can clone only a residual-ranker policy")
    backbone = policy.backbone
    device = next(backbone.parameters()).device
    cloned_backbone = RecurrentAttackPolicy(
        backbone.observation_dim,
        backbone.action_dim,
        hidden_dim=backbone.hidden_dim,
        seed=0,
        config=backbone.config,
        actor_mode=backbone.actor_mode,
        action_grid_size=backbone.action_grid_size,
    ).to(device)
    cloned_backbone.load_state_dict(copy.deepcopy(backbone.state_dict()))
    cloned_backbone.optimizer.load_state_dict(
        copy.deepcopy(backbone.optimizer.state_dict())
    )
    clone = ResidualRankerPolicy(
        cloned_backbone,
        confidence_threshold=policy.confidence_threshold,
        prior_temperature=policy.prior_temperature,
        overrides_enabled=policy.overrides_enabled,
    )
    if clone is policy or clone.backbone is backbone:
        raise ValueError("D1b residual clone shares mutable state")
    if clone.persistent_digest() != policy.persistent_digest():
        raise ValueError("D1b residual clone does not preserve policy identity")
    return clone


@dataclass(frozen=True)
class ResidualD1BStoreBinding:
    """Immutable identity and architecture binding for all PPO blocks."""

    root: Path
    device: str | torch.device
    observation_dim: int
    action_dim: int
    hidden_dim: int
    d1a_manifest_digest: str
    d1a_checkpoint_sha256: str
    bc_policy_digest: str
    source_roles_digest: str

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        dimensions = (
            self.observation_dim,
            self.action_dim,
            self.hidden_dim,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in dimensions
        ):
            raise ValueError("D1b store dimensions must be positive integers")
        for label in (
            "d1a_manifest_digest",
            "d1a_checkpoint_sha256",
            "bc_policy_digest",
            "source_roles_digest",
        ):
            _digest(getattr(self, label), f"D1b store {label}")
        object.__setattr__(self, "root", root)

    @property
    def identity(self) -> dict[str, object]:
        return {
            "d1a_manifest_digest": self.d1a_manifest_digest,
            "d1a_checkpoint_sha256": self.d1a_checkpoint_sha256,
            "bc_policy_digest": self.bc_policy_digest,
            "source_roles_digest": self.source_roles_digest,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }


class ResidualD1BBlockStore:
    """Persist and verify a contiguous four-block D1b PPO checkpoint chain."""

    def __init__(self, binding: ResidualD1BStoreBinding) -> None:
        if not isinstance(binding, ResidualD1BStoreBinding):
            raise TypeError("D1b block store requires its locked binding")
        self.binding = binding
        self.root = binding.root
        self.root.mkdir(parents=True, exist_ok=True)

    def _checkpoint(self, endpoint: int) -> Path:
        return self.root / f"ppo_block_{endpoint:03d}.pt"

    def _receipt_path(self, endpoint: int) -> Path:
        return self.root / f"ppo_block_{endpoint:03d}.receipt.json"

    @staticmethod
    def _sidecar(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".sha256")

    def _validate_metrics(
        self,
        raw: object,
        *,
        block_index: int,
    ) -> dict[str, object]:
        metrics = _mapping(raw, "D1b PPO block metrics")
        _source_only(metrics, "D1b PPO block metrics")
        offset = (block_index - 1) * D1B_BLOCK_EPISODES
        if (
            metrics.get("episodes") != D1B_BLOCK_EPISODES
            or metrics.get("episode_offset") != offset
            or metrics.get("next_episode_offset") != offset + D1B_BLOCK_EPISODES
        ):
            raise ValueError("D1b PPO metrics violate the block boundary")
        if (
            _integer(metrics.get("trained_episodes"), "D1b trained episodes")
            > D1B_BLOCK_EPISODES
        ):
            raise ValueError("D1b trained episodes exceed the block size")
        source_calls = _integer(
            metrics.get("source_calls"),
            "D1b source calls",
        )
        calls = _mapping(
            metrics.get("source_calls_by_family"),
            "D1b source calls by family",
        )
        victim_calls = _mapping(
            metrics.get("source_calls_by_victim"),
            "D1b source calls by victim",
        )
        weights = _mapping(
            metrics.get("family_weights"),
            "D1b family weights",
        )
        offsets = _mapping(
            metrics.get("instance_offsets"),
            "D1b instance offsets",
        )
        if any(
            set(value) != set(D1_SOURCE_FAMILIES) for value in (calls, weights, offsets)
        ):
            raise ValueError("D1b metrics must contain exactly source families")
        if (
            sum(
                _number(weights[family], f"D1b {family} weight")
                for family in D1_SOURCE_FAMILIES
            )
            <= 0
        ):
            raise ValueError("D1b family weights require positive mass")
        for family in D1_SOURCE_FAMILIES:
            _integer(calls[family], f"D1b {family} source calls")
            _integer(offsets[family], f"D1b {family} instance offset")
        if (
            not victim_calls
            or any(
                not isinstance(victim_id, str)
                or not victim_id
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for victim_id, value in victim_calls.items()
            )
            or source_calls != sum(int(value) for value in calls.values())
            or source_calls != sum(int(value) for value in victim_calls.values())
        ):
            raise ValueError("D1b source-call accounting is inconsistent")
        return metrics

    def _validate_metadata(
        self,
        raw: object,
        metrics: Mapping[str, object],
        *,
        policy_digest: str | None = None,
    ) -> dict[str, object]:
        metadata = _mapping(raw, "D1b checkpoint metadata")
        _source_only(metadata, "D1b checkpoint metadata")
        if set(metadata) != _CORE_FIELDS:
            raise ValueError("D1b checkpoint metadata schema is not exact")
        block_index = _integer(
            metadata.get("block_index"),
            "D1b block index",
            minimum=1,
        )
        if block_index > len(D1B_BLOCK_ENDPOINTS):
            raise ValueError("D1b block index exceeds the protocol")
        endpoint = block_index * D1B_BLOCK_EPISODES
        if (
            metadata.get("schema_version") != 1
            or metadata.get("name") != "phase2-d1b-residual-ranker-ppo-block"
            or metadata.get("episode_offset") != endpoint - D1B_BLOCK_EPISODES
            or metadata.get("episodes") != D1B_BLOCK_EPISODES
            or metadata.get("episodes_completed") != endpoint
        ):
            raise ValueError("D1b checkpoint metadata violates block identity")
        for key, expected in self.binding.identity.items():
            if key in metadata and metadata[key] != expected:
                raise ValueError("D1b checkpoint identity binding mismatch")
        if (
            metadata.get("d1a_manifest_digest") != self.binding.d1a_manifest_digest
            or metadata.get("d1a_checkpoint_sha256")
            != self.binding.d1a_checkpoint_sha256
            or metadata.get("bc_policy_digest") != self.binding.bc_policy_digest
            or metadata.get("source_roles_digest") != self.binding.source_roles_digest
        ):
            raise ValueError("D1b checkpoint identity binding mismatch")
        parent = metadata.get("parent_checkpoint")
        if block_index == 1:
            if parent is not None:
                raise ValueError("D1b first checkpoint cannot have a parent")
        else:
            parent_record = self._validate_record(
                load_verified_json(self._receipt_path(endpoint - D1B_BLOCK_EPISODES)),
                endpoint=endpoint - D1B_BLOCK_EPISODES,
            )
            expected_parent = {
                "reference": parent_record["opaque_reference"],
                "policy_digest": parent_record["policy_digest"],
                "metadata_digest": parent_record["metadata_digest"],
            }
            if parent != expected_parent:
                raise ValueError("D1b checkpoint parent receipt binding mismatch")
        recorded_policy = _digest(
            metadata.get("ppo_policy_digest"),
            "D1b PPO policy digest",
        )
        if policy_digest is not None and recorded_policy != policy_digest:
            raise ValueError("D1b checkpoint policy digest mismatch")
        if metadata.get("ppo_metrics_digest") != canonical_json_digest(metrics):
            raise ValueError("D1b checkpoint metrics digest mismatch")
        if metadata.get("family_weights") != metrics.get(
            "family_weights"
        ) or metadata.get("instance_offsets") != metrics.get("instance_offsets"):
            raise ValueError("D1b checkpoint state differs from block metrics")
        return metadata

    @staticmethod
    def _controls(policy: ResidualRankerPolicy) -> dict[str, object]:
        return {
            "confidence_threshold": policy.confidence_threshold,
            "prior_temperature": policy.prior_temperature,
            "overrides_enabled": policy.overrides_enabled,
        }

    def _record(
        self,
        *,
        endpoint: int,
        checkpoint_sha256: str,
        metadata: Mapping[str, object],
        metrics: Mapping[str, object],
        controls: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "name": "phase2-d1b-residual-ranker-ppo-receipt",
            "binding": self.binding.identity,
            "checkpoint": {
                "name": self._checkpoint(endpoint).name,
                "sha256": checkpoint_sha256,
            },
            "opaque_reference": (
                f"d1b-{metadata['ppo_policy_digest']}-{canonical_json_digest(metadata)}"
            ),
            "policy_digest": metadata["ppo_policy_digest"],
            "metadata_digest": canonical_json_digest(metadata),
            "core_metadata": dict(metadata),
            "metrics": dict(metrics),
            "residual_controls": dict(controls),
            **_seal(),
        }

    def save_block(
        self,
        policy: object,
        metadata: Mapping[str, object],
        metrics: Mapping[str, object],
    ) -> ResidualD1BCheckpointReceipt:
        if not isinstance(policy, ResidualRankerPolicy):
            raise TypeError("D1b block saver requires a residual policy")
        policy_digest = policy.persistent_digest()
        _digest(policy_digest, "D1b saved policy digest")
        raw_metadata = _mapping(metadata, "D1b checkpoint metadata")
        block_index = _integer(
            raw_metadata.get("block_index"),
            "D1b block index",
            minimum=1,
        )
        checked_metrics = self._validate_metrics(
            metrics,
            block_index=block_index,
        )
        checked_metadata = self._validate_metadata(
            raw_metadata,
            checked_metrics,
            policy_digest=policy_digest,
        )
        endpoint = block_index * D1B_BLOCK_EPISODES
        checkpoint = self._checkpoint(endpoint)
        receipt_path = self._receipt_path(endpoint)
        paths = (
            checkpoint,
            self._sidecar(checkpoint),
            receipt_path,
            self._sidecar(receipt_path),
        )
        if any(path.exists() for path in paths):
            raise ValueError("D1b block artifact already exists")
        if block_index > 1:
            predecessor = endpoint - D1B_BLOCK_EPISODES
            required = (
                self._checkpoint(predecessor),
                self._sidecar(self._checkpoint(predecessor)),
                self._receipt_path(predecessor),
                self._sidecar(self._receipt_path(predecessor)),
            )
            if not all(path.is_file() for path in required):
                raise ValueError("D1b block checkpoints must be contiguous")
        controls = self._controls(policy)
        wrapper = {
            "schema_version": 1,
            "kind": "phase2_d1b_source_only_ppo_block",
            "binding": self.binding.identity,
            "core_metadata": checked_metadata,
            "metrics": checked_metrics,
            "residual_controls": controls,
            **_seal(),
        }
        checkpoint_sha256 = save_recurrent_checkpoint(
            checkpoint,
            policy.backbone,
            wrapper,
        )
        record = self._record(
            endpoint=endpoint,
            checkpoint_sha256=checkpoint_sha256,
            metadata=checked_metadata,
            metrics=checked_metrics,
            controls=controls,
        )
        write_verified_json(receipt_path, record)
        return self._receipt_from_record(record)

    def _validate_record(
        self,
        raw: object,
        *,
        endpoint: int,
    ) -> dict[str, object]:
        record = _mapping(raw, "D1b block receipt")
        _source_only(record, "D1b block receipt")
        expected_fields = {
            "schema_version",
            "name",
            "binding",
            "checkpoint",
            "opaque_reference",
            "policy_digest",
            "metadata_digest",
            "core_metadata",
            "metrics",
            "residual_controls",
            *_seal(),
        }
        if (
            set(record) != expected_fields
            or record.get("schema_version") != 1
            or record.get("name") != "phase2-d1b-residual-ranker-ppo-receipt"
            or record.get("binding") != self.binding.identity
        ):
            raise ValueError("D1b receipt identity or schema binding mismatch")
        checkpoint = _mapping(record.get("checkpoint"), "D1b receipt checkpoint")
        if set(checkpoint) != {"name", "sha256"}:
            raise ValueError("D1b receipt checkpoint schema is invalid")
        path = self._checkpoint(endpoint)
        if checkpoint.get("name") != path.name or checkpoint.get(
            "sha256"
        ) != sha256_file(path):
            raise ValueError("D1b receipt checkpoint checksum mismatch")
        metrics = self._validate_metrics(
            record.get("metrics"),
            block_index=endpoint // D1B_BLOCK_EPISODES,
        )
        metadata = self._validate_metadata(record.get("core_metadata"), metrics)
        if (
            record.get("policy_digest") != metadata["ppo_policy_digest"]
            or record.get("metadata_digest") != canonical_json_digest(metadata)
            or record.get("opaque_reference")
            != (f"d1b-{record['policy_digest']}-{record['metadata_digest']}")
        ):
            raise ValueError("D1b receipt digest binding mismatch")
        controls = _mapping(
            record.get("residual_controls"),
            "D1b residual controls",
        )
        if set(controls) != {
            "confidence_threshold",
            "prior_temperature",
            "overrides_enabled",
        }:
            raise ValueError("D1b receipt residual controls are invalid")
        _number(controls["confidence_threshold"], "D1b confidence threshold")
        _number(
            controls["prior_temperature"],
            "D1b prior temperature",
            positive=True,
        )
        if not isinstance(controls["overrides_enabled"], bool):
            raise ValueError("D1b override control must be boolean")
        return record

    def _checkpoint_wrapper(
        self,
        path: Path,
    ) -> tuple[ResidualRankerPolicy, dict[str, object]]:
        backbone, raw_wrapper = load_recurrent_checkpoint(
            path,
            self.binding.device,
            expected_observation_dim=self.binding.observation_dim,
            expected_action_dim=self.binding.action_dim,
            expected_hidden_dim=self.binding.hidden_dim,
            expected_actor_mode="action_conditioned",
        )
        wrapper = _mapping(raw_wrapper, "D1b checkpoint wrapper")
        _source_only(wrapper, "D1b checkpoint wrapper")
        expected_fields = {
            "schema_version",
            "kind",
            "binding",
            "core_metadata",
            "metrics",
            "residual_controls",
            *_seal(),
        }
        if (
            set(wrapper) != expected_fields
            or wrapper.get("schema_version") != 1
            or wrapper.get("kind") != "phase2_d1b_source_only_ppo_block"
            or wrapper.get("binding") != self.binding.identity
        ):
            raise ValueError("D1b checkpoint wrapper binding failed")
        controls = _mapping(
            wrapper.get("residual_controls"),
            "D1b checkpoint controls",
        )
        policy = ResidualRankerPolicy(
            backbone,
            confidence_threshold=_number(
                controls.get("confidence_threshold"),
                "D1b loaded threshold",
            ),
            prior_temperature=_number(
                controls.get("prior_temperature"),
                "D1b loaded prior temperature",
                positive=True,
            ),
            overrides_enabled=controls.get("overrides_enabled"),  # type: ignore[arg-type]
        )
        metrics = _mapping(wrapper.get("metrics"), "D1b loaded metrics")
        metadata = self._validate_metadata(
            wrapper.get("core_metadata"),
            metrics,
            policy_digest=policy.persistent_digest(),
        )
        return policy, {
            **wrapper,
            "core_metadata": metadata,
            "metrics": metrics,
        }

    def _receipt_from_record(
        self,
        record: Mapping[str, object],
    ) -> ResidualD1BCheckpointReceipt:
        return ResidualD1BCheckpointReceipt(
            reference=str(record["opaque_reference"]),
            policy_digest=str(record["policy_digest"]),
            metadata_digest=str(record["metadata_digest"]),
            hidden_target_calls=0,
        )

    def load_block(
        self,
        receipt: ResidualD1BCheckpointReceipt,
    ) -> ResidualD1BLoadedCheckpoint:
        if not isinstance(receipt, ResidualD1BCheckpointReceipt):
            raise TypeError("D1b loader requires a checkpoint receipt")
        if _REFERENCE.fullmatch(receipt.reference) is None:
            raise ValueError("D1b checkpoint reference is unsafe")
        matches: list[tuple[int, dict[str, object]]] = []
        for endpoint in D1B_BLOCK_ENDPOINTS:
            receipt_path = self._receipt_path(endpoint)
            if not receipt_path.is_file() or not self._sidecar(receipt_path).is_file():
                continue
            candidate = self._validate_record(
                load_verified_json(receipt_path),
                endpoint=endpoint,
            )
            if candidate.get("opaque_reference") == receipt.reference:
                matches.append((endpoint, candidate))
        if len(matches) != 1:
            raise ValueError("D1b opaque checkpoint reference is unresolved")
        endpoint, record = matches[0]
        if self._receipt_from_record(record) != receipt:
            raise ValueError("D1b supplied receipt differs from disk")
        policy, wrapper = self._checkpoint_wrapper(self._checkpoint(endpoint))
        metadata = _mapping(
            wrapper["core_metadata"],
            "D1b loaded core metadata",
        )
        if (
            policy.persistent_digest() != receipt.policy_digest
            or canonical_json_digest(metadata) != receipt.metadata_digest
        ):
            raise ValueError("D1b loaded checkpoint differs from its receipt")
        return ResidualD1BLoadedCheckpoint(
            policy=policy,
            metadata=metadata,
            hidden_target_calls=0,
        )

    def _recover_receipt(self, endpoint: int) -> dict[str, object]:
        policy, wrapper = self._checkpoint_wrapper(self._checkpoint(endpoint))
        metadata = _mapping(wrapper["core_metadata"], "D1b recovered metadata")
        metrics = _mapping(wrapper["metrics"], "D1b recovered metrics")
        controls = _mapping(
            wrapper["residual_controls"],
            "D1b recovered controls",
        )
        record = self._record(
            endpoint=endpoint,
            checkpoint_sha256=sha256_file(self._checkpoint(endpoint)),
            metadata=metadata,
            metrics=metrics,
            controls=controls,
        )
        if record["policy_digest"] != policy.persistent_digest():
            raise ValueError("D1b orphan checkpoint policy digest mismatch")
        write_verified_json(self._receipt_path(endpoint), record)
        return record

    def _reject_unknown_block_files(self) -> None:
        expected = {
            path.name
            for endpoint in D1B_BLOCK_ENDPOINTS
            for base in (
                self._checkpoint(endpoint),
                self._receipt_path(endpoint),
            )
            for path in (base, self._sidecar(base))
        }
        unknown = {
            path.name
            for path in self.root.glob("ppo_block_*")
            if path.name not in expected
        }
        if unknown:
            raise ValueError(f"D1b has unexpected block artifacts: {sorted(unknown)}")

    def load_resume_state(self) -> ResidualD1BResumeState | None:
        """Verify every checkpoint and return only a contiguous committed prefix."""

        self._reject_unknown_block_files()
        blocks: list[ResidualD1BBlockState] = []
        gap_seen = False
        for endpoint in D1B_BLOCK_ENDPOINTS:
            checkpoint = self._checkpoint(endpoint)
            receipt_path = self._receipt_path(endpoint)
            checkpoint_state = (
                checkpoint.is_file(),
                self._sidecar(checkpoint).is_file(),
            )
            receipt_state = (
                receipt_path.is_file(),
                self._sidecar(receipt_path).is_file(),
            )
            if checkpoint_state == (False, False):
                if receipt_state != (False, False):
                    raise ValueError("D1b orphan receipt creates a block gap")
                gap_seen = True
                continue
            if gap_seen:
                raise ValueError("D1b checkpoints are not contiguous")
            if checkpoint_state != (True, True):
                raise ValueError("D1b checkpoint artifact pair is incomplete")
            if receipt_state == (False, False):
                record = self._recover_receipt(endpoint)
            elif receipt_state != (True, True):
                raise ValueError("D1b receipt artifact pair is incomplete")
            else:
                record = load_verified_json(receipt_path)
            record = self._validate_record(record, endpoint=endpoint)
            receipt = self._receipt_from_record(record)
            loaded = self.load_block(receipt)
            metadata = _mapping(
                loaded.metadata,
                "D1b resume checkpoint metadata",
            )
            metrics = _mapping(record["metrics"], "D1b resume metrics")
            blocks.append(
                ResidualD1BBlockState(
                    block_index=endpoint // D1B_BLOCK_EPISODES,
                    episode_offset=endpoint - D1B_BLOCK_EPISODES,
                    metrics=metrics,
                    family_weights=_mapping(
                        metrics["family_weights"],
                        "D1b resume family weights",
                    ),
                    instance_offsets=_mapping(
                        metrics["instance_offsets"],
                        "D1b resume instance offsets",
                    ),
                    policy_digest=receipt.policy_digest,
                    checkpoint=receipt,
                    checkpoint_metadata=metadata,
                )
            )
        if not blocks:
            return None
        return ResidualD1BResumeState(
            d1a_manifest_digest=self.binding.d1a_manifest_digest,
            d1a_checkpoint_sha256=self.binding.d1a_checkpoint_sha256,
            bc_policy_digest=self.binding.bc_policy_digest,
            source_roles_digest=self.binding.source_roles_digest,
            blocks=tuple(blocks),
        )


__all__ = (
    "ResidualD1BBlockStore",
    "ResidualD1BStoreBinding",
    "canonical_json_digest",
    "clone_residual_policy",
)
