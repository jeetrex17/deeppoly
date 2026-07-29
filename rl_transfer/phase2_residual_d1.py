"""Fail-closed contract and promotion gate for the Phase 2 D1 diagnostic.

D1 is intentionally source-only. It tests whether a behavior-cloned residual
ranker can improve a deterministic score-greedy proposal order. It cannot
authorize hidden-target evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import AttackConfig

D1_HELDOUT_FAMILY = "modern_cnn"
D1_SEED = 17
D1_SOURCE_IMAGES = 50
D1_BC_EPISODES = 200
D1_MAX_PPO_EPISODES = 200
D1_MAX_SECONDS = 8 * 60 * 60.0
D1_SOURCE_FAMILIES = ("classical_cnn", "transformer")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FALSE_TARGET_FIELDS = {
    "authorizes_hidden_target_evaluation",
    "hidden_target_evaluation_performed",
    "target_evaluation_available",
    "target_evaluation_performed",
}
_AUDITED_SOURCE_CALL_FIELDS = {
    "max_total_target_calls",
    "total_target_calls",
}


@dataclass(frozen=True)
class ResidualD1Request:
    """Immutable, bounded request for the one-fold D1 source diagnostic."""

    source_manifest: Path
    source_root: Path
    output_dir: Path
    data_root: Path
    heldout_family: str = D1_HELDOUT_FAMILY
    seed: int = D1_SEED
    source_images: int = D1_SOURCE_IMAGES
    deadline_seconds: float = D1_MAX_SECONDS
    bc_episodes: int = D1_BC_EPISODES
    ppo_episodes: int = 0
    device: str = "cuda"
    download: bool = False

    def __post_init__(self) -> None:
        if self.heldout_family != D1_HELDOUT_FAMILY:
            raise ValueError("D1 is locked to the modern_cnn-heldout fold")
        if self.seed != D1_SEED:
            raise ValueError("D1 is locked to policy seed 17")
        if self.source_images != D1_SOURCE_IMAGES:
            raise ValueError("D1 requires exactly 50 source evaluation images")
        if self.bc_episodes != D1_BC_EPISODES:
            raise ValueError("D1 requires exactly 200 behavior-cloning episodes")
        if (
            isinstance(self.ppo_episodes, bool)
            or not isinstance(self.ppo_episodes, int)
            or self.ppo_episodes != 0
        ):
            raise ValueError("D1a is BC-only and requires zero PPO episodes")
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(float(self.deadline_seconds))
            or not 0 < float(self.deadline_seconds) <= D1_MAX_SECONDS
        ):
            raise ValueError("D1 deadline must be in (0, 28800] seconds")
        if self.device != "cuda":
            raise ValueError("D1 execution is locked to the CUDA workstation")
        if not isinstance(self.download, bool):
            raise ValueError("D1 download must be boolean")

        source_root = Path(self.source_root).resolve()
        source_manifest = Path(self.source_manifest).resolve()
        output_dir = Path(self.output_dir).resolve()
        data_root = Path(self.data_root).resolve()
        try:
            source_manifest.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                "D1 source manifest must be inside the sealed source root"
            ) from error
        if (
            output_dir == source_root
            or output_dir in source_root.parents
            or source_root in output_dir.parents
        ):
            raise ValueError("D1 output must not overlap the sealed source tree")
        if (
            output_dir == data_root
            or output_dir in data_root.parents
            or data_root in output_dir.parents
        ):
            raise ValueError("D1 output and dataset roots must not overlap")
        if (
            data_root == source_root
            or data_root in source_root.parents
            or source_root in data_root.parents
        ):
            raise ValueError("D1 dataset and sealed source roots must not overlap")

        object.__setattr__(self, "source_manifest", source_manifest)
        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "data_root", data_root)
        object.__setattr__(
            self,
            "deadline_seconds",
            float(self.deadline_seconds),
        )

    def digest(self) -> str:
        payload = {
            **asdict(self),
            "source_manifest": str(self.source_manifest),
            "source_root": str(self.source_root),
            "output_dir": str(self.output_dir),
            "data_root": str(self.data_root),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ResidualCacheBinding:
    """Identity binding for source-only teacher examples."""

    source_manifest_sha256: str
    dataset_content_sha256: str
    victim_cache_digest: str
    request_sha256: str

    def __post_init__(self) -> None:
        for label, value in asdict(self).items():
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def select_residual_action(
    *,
    score_greedy_action: int,
    learned_action: int,
    residual_confidence: float,
    confidence_threshold: float,
) -> int:
    """Use the learned proposal only when its confidence clears the gate."""

    actions = (score_greedy_action, learned_action)
    if any(
        isinstance(action, bool) or not isinstance(action, int) or action < 0
        for action in actions
    ):
        raise ValueError("action indices must be non-negative integers")
    numeric = (residual_confidence, confidence_threshold)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in numeric
    ):
        raise ValueError("residual confidence values must be finite and non-negative")
    return (
        learned_action
        if float(residual_confidence) >= float(confidence_threshold)
        else score_greedy_action
    )


def validate_residual_cache_binding(
    expected: ResidualCacheBinding,
    actual: ResidualCacheBinding,
) -> None:
    if not isinstance(expected, ResidualCacheBinding) or not isinstance(
        actual,
        ResidualCacheBinding,
    ):
        raise TypeError("residual cache bindings must use the locked schema")
    if expected != actual:
        raise ValueError("residual teacher cache binding mismatch")


def validate_d1_attack_contract(
    attack: AttackConfig,
    source_families: Sequence[str],
) -> None:
    """Enforce the preregistered source-only attack operator."""

    if not isinstance(attack, AttackConfig):
        raise TypeError("D1 attack contract requires AttackConfig")
    if (
        tuple(source_families) != D1_SOURCE_FAMILIES
        or attack.max_queries != 50
        or attack.grid_size != 4
        or not math.isclose(attack.epsilon, 8 / 255, abs_tol=1e-12)
        or not math.isclose(attack.step_size, 2 / 255, abs_tol=1e-12)
        or not attack.rollback_on_non_improvement
        or not attack.action_history_features
        or not attack.image_patch_features
        or attack.image_patch_feature_mode != "statistics"
    ):
        raise ValueError(
            "D1 attack/operator contract does not match the locked protocol"
        )


def residual_d1_promotion_decision(
    *,
    bc_validation_score: float,
    prior_validation_score: float,
    score_greedy_asr: float,
    score_greedy_auc: float,
    learned_asr: float,
    learned_auc: float,
) -> dict[str, object]:
    """Return a descriptive D1 decision that never authorizes target access."""

    values = (
        bc_validation_score,
        prior_validation_score,
        score_greedy_asr,
        score_greedy_auc,
        learned_asr,
        learned_auc,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("D1 promotion metrics must be finite numbers")
    if any(not 0 <= float(value) <= 1 for value in values):
        raise ValueError("D1 promotion metrics must be in [0, 1]")

    bc_improved = float(bc_validation_score) > float(prior_validation_score)
    asr_non_regression = float(learned_asr) >= float(score_greedy_asr)
    auc_non_regression = float(learned_auc) >= float(score_greedy_auc)
    passed = bc_improved and asr_non_regression and auc_non_regression
    return {
        "passed": passed,
        "bc_improved_over_prior": bc_improved,
        "asr_observed_non_decrease": asr_non_regression,
        "auc_observed_non_decrease": auc_non_regression,
        "eligible_for_d1b_source_only_ppo": passed,
        "authorizes_hidden_target_evaluation": False,
    }


def validate_residual_source_records(
    records: Sequence[Mapping[str, object]],
    *,
    heldout_family: str,
) -> None:
    """Reject hidden-target calls or held-out-family contamination."""

    if heldout_family != D1_HELDOUT_FAMILY:
        raise ValueError("D1 source records use the wrong held-out family")
    if not records:
        raise ValueError("D1 source evidence cannot be empty")

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("D1 source evidence records must be mappings")
        victim_family = record.get("victim_family")
        victim_id = record.get("victim_id")
        target_calls = record.get("target_calls")
        hidden_target_calls = record.get("hidden_target_calls")
        if (
            victim_family not in D1_SOURCE_FAMILIES
            or victim_family == heldout_family
            or record.get("heldout_family") != heldout_family
        ):
            raise ValueError("held-out victim family leaked into D1 source evidence")
        if not isinstance(victim_id, str) or not victim_id:
            raise ValueError("D1 source evidence requires a victim ID")
        if (
            isinstance(target_calls, bool)
            or not isinstance(target_calls, int)
            or target_calls != 0
            or isinstance(hidden_target_calls, bool)
            or not isinstance(hidden_target_calls, int)
            or hidden_target_calls != 0
        ):
            raise ValueError(
                "D1 source evidence must record zero target and hidden-target calls"
            )
        validate_source_only_payload(record, "D1 source evidence")


def validate_source_only_payload(value: object, label: str) -> None:
    """Reject target access while preserving audited source-query counters."""

    if not isinstance(label, str) or not label:
        raise ValueError("source-only validation requires a label")
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if (
                key.endswith("target_calls")
                and key not in _AUDITED_SOURCE_CALL_FIELDS
                and (isinstance(item, bool) or not isinstance(item, int) or item != 0)
            ):
                raise ValueError(f"{label}.{key} must be integer zero")
            if key in _FALSE_TARGET_FIELDS and item is not False:
                raise ValueError(f"{label}.{key} must be false")
            validate_source_only_payload(item, f"{label}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_source_only_payload(item, f"{label}[{index}]")


__all__ = (
    "D1_BC_EPISODES",
    "D1_HELDOUT_FAMILY",
    "D1_MAX_PPO_EPISODES",
    "D1_MAX_SECONDS",
    "D1_SEED",
    "D1_SOURCE_FAMILIES",
    "D1_SOURCE_IMAGES",
    "ResidualCacheBinding",
    "ResidualD1Request",
    "residual_d1_promotion_decision",
    "select_residual_action",
    "validate_d1_attack_contract",
    "validate_residual_cache_binding",
    "validate_residual_source_records",
    "validate_source_only_payload",
)
