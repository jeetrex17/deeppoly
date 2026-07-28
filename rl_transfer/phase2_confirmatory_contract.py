"""Locked, source-only confirmatory contract for the Phase 2 study.

This module intentionally does not run training or target evaluation. It
validates the preregistered contract and deterministically allocates the
untouched final source-test indices from CIFAR-10's unused training complement.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from types import MappingProxyType
from typing import Any

from .cifar_data import build_cifar_split, indices_digest


DEFAULT_CONFIRMATORY_CONTRACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "rl_transfer"
    / "cifar10_rtx_phase2_confirmatory_contract.json"
)
CONFIRMATORY_CONTRACT_SHA256 = (
    "787d9a53e1bd5fc4eec1d55da2b080e933034b2401cd4e67354b21cec5706e52"
)
_PHASE1_SEEDS = (17, 29, 41, 53, 67, 79, 97, 113, 131, 149)
_STAGE_B_SEEDS = (17,)
_FINAL_ALLOCATION_ALGORITHM = "balanced-complement-shuffle-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ConfirmatoryContract:
    """Immutable validated view of the locked JSON contract."""

    contract_name: str
    payload_digest: str
    phase1_seeds: tuple[int, ...]
    stage_b_seeds: tuple[int, ...]
    stage_c_seeds: tuple[int, ...]
    final_confirmatory_seeds: tuple[int, ...]
    split_seed: int
    victim_fit_images: int
    policy_train_images: int
    source_validation_images: int
    outer_test_images: int
    untouched_complement_images: int
    base_split_digest: str
    phase1_role_digests: Mapping[str, str]
    final_source_test_algorithm: str
    final_source_test_count: int
    final_source_test_digest: str
    final_source_candidate_digest: str
    final_source_test_allowed_stage: str
    final_source_test_prohibited_stages: tuple[str, ...]
    rules: Mapping[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def canonical_contract_digest(payload: Mapping[str, Any]) -> str:
    """Return SHA-256 over canonical JSON, independent of file formatting."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(
    parent: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a JSON object")
    return value


def _integer(
    parent: Mapping[str, Any],
    key: str,
    *,
    minimum: int = 0,
) -> int:
    value = parent.get(key)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(f"{key} must be an integer of at least {minimum}")
    return value


def _integer_tuple(
    parent: Mapping[str, Any],
    key: str,
) -> tuple[int, ...]:
    value = parent.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            for item in value
        )
    ):
        raise ValueError(f"{key} must be a non-empty integer array")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{key} must contain unique seeds")
    return result


def _string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _sha256(parent: Mapping[str, Any], key: str) -> str:
    value = _string(parent, key)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def next_primes(strictly_above: int, count: int) -> tuple[int, ...]:
    """Return the requested number of consecutive primes above a boundary."""

    if (
        not isinstance(strictly_above, int)
        or isinstance(strictly_above, bool)
        or strictly_above < 0
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
    ):
        raise ValueError("prime boundary and count must be positive integers")
    values: list[int] = []
    candidate = strictly_above + 1
    while len(values) < count:
        if _is_prime(candidate):
            values.append(candidate)
        candidate += 1
    return tuple(values)


def _validate_seed_contract(
    seeds: Mapping[str, Any],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    phase1 = _integer_tuple(seeds, "phase1")
    stage_b = _integer_tuple(seeds, "stage_b")
    stage_c = _integer_tuple(seeds, "stage_c")
    final = _integer_tuple(seeds, "final_confirmatory")
    if phase1 != _PHASE1_SEEDS:
        raise ValueError("Phase 1 seed identity changed")
    if stage_b != _STAGE_B_SEEDS:
        raise ValueError("Stage B seed identity changed")
    if stage_c != next_primes(max(phase1), 2):
        raise ValueError(
            "Stage C seeds must be the first two primes above Phase 1"
        )
    if final != next_primes(max(stage_c), 10):
        raise ValueError(
            "final seeds must be the next ten primes after Stage C"
        )
    development = set(phase1) | set(stage_b)
    if development.intersection(stage_c) or development.intersection(final):
        raise ValueError("confirmatory seeds overlap development seeds")
    if set(stage_c).intersection(final):
        raise ValueError("Stage C and final seeds overlap")
    return phase1, stage_b, stage_c, final


def _validate_dataset_contract(
    dataset: Mapping[str, Any],
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    str,
    Mapping[str, str],
]:
    train_images = _integer(dataset, "train_images", minimum=1)
    class_count = _integer(dataset, "class_count", minimum=1)
    split_seed = _integer(dataset, "split_seed")
    victim_fit = _integer(dataset, "victim_fit_images", minimum=1)
    policy_train = _integer(dataset, "policy_train_images", minimum=1)
    validation = _integer(dataset, "source_validation_images", minimum=1)
    outer_test = _integer(dataset, "outer_test_images", minimum=1)
    complement = _integer(
        dataset,
        "untouched_complement_images",
        minimum=1,
    )
    if (
        train_images != 50_000
        or class_count != 10
        or split_seed != 20_260_727
        or (victim_fit, policy_train, validation) != (40_000, 4_000, 1_000)
        or outer_test != 1_000
        or complement != 5_000
        or victim_fit + policy_train + validation + complement
        != train_images
    ):
        raise ValueError("locked CIFAR-10 data partition changed")
    base_split_digest = _sha256(dataset, "base_split_digest")
    role_digests = _object(dataset, "role_digests")
    frozen_role_digests = MappingProxyType(
        {
            role: _sha256(role_digests, role)
            for role in (
                "victim_fit",
                "policy_train",
                "source_validation",
                "outer_test",
            )
        }
    )
    return (
        split_seed,
        victim_fit,
        policy_train,
        validation,
        outer_test,
        complement,
        base_split_digest,
        frozen_role_digests,
    )


def _validate_final_source_test_contract(
    block: Mapping[str, Any],
    *,
    split_seed: int,
    complement: int,
) -> tuple[str, int, str, str, str, tuple[str, ...]]:
    algorithm = _string(block, "algorithm")
    if (
        algorithm != _FINAL_ALLOCATION_ALGORITHM
        or _integer(block, "algorithm_version", minimum=1) != 1
        or _integer(block, "split_seed") != split_seed
        or _integer(block, "selection_seed") != split_seed
    ):
        raise ValueError("final source-test allocation algorithm changed")
    candidate_count = _integer(block, "candidate_images", minimum=1)
    candidate_per_class = _integer(
        block,
        "candidate_images_per_class",
        minimum=1,
    )
    selected_count = _integer(block, "selected_images", minimum=1)
    selected_per_class = _integer(
        block,
        "selected_images_per_class",
        minimum=1,
    )
    if (
        candidate_count != complement
        or candidate_per_class != complement // 10
        or selected_count != 1_000
        or selected_per_class != selected_count // 10
    ):
        raise ValueError("final source-test count or balance changed")
    candidate_digest = _sha256(block, "candidate_indices_digest")
    selected_digest = _sha256(block, "selected_indices_digest")
    allowed = _string(block, "allowed_stage")
    prohibited_value = block.get("prohibited_stages")
    if not isinstance(prohibited_value, list) or any(
        not isinstance(stage, str) or not stage
        for stage in prohibited_value
    ):
        raise ValueError("prohibited_stages must be a string array")
    prohibited = tuple(prohibited_value)
    expected_prohibited = (
        "stage_a",
        "stage_b",
        "stage_c_first_rung",
        "stage_c_extended_rung",
        "stage_c_promotion",
    )
    if (
        allowed != "final_confirmatory_source"
        or prohibited != expected_prohibited
        or _integer(block, "evaluation_access_count_before_allowed_stage") != 0
        or block.get("indices_preregistered_before_stage_b") is not True
    ):
        raise ValueError("final source-test access boundary changed")
    return (
        algorithm,
        selected_count,
        selected_digest,
        candidate_digest,
        allowed,
        prohibited,
    )


def _validate_rules(rules: Mapping[str, Any]) -> None:
    stage_c = _object(rules, "stage_c")
    first_rung = _object(stage_c, "first_rung")
    promotion = _object(stage_c, "promotion")
    final = _object(rules, "final_confirmatory_source")
    exact_requirements = (
        (stage_c, "entry_condition", "stage_b_gate_passed"),
        (first_rung, "initial_ppo_episodes", 600),
        (first_rung, "maximum_ppo_episodes", 1_200),
        (first_rung, "minimum_macro_asr_gain", 0.025),
        (first_rung, "minimum_macro_auc_gain", 0.010),
        (
            first_rung,
            "minimum_soft_top5_gain_over_validation_oracle",
            0.010,
        ),
        (
            first_rung,
            "minimum_soft_cross_entropy_improvement_over_validation_oracle",
            0.020,
        ),
        (first_rung, "no_regression_tail_episodes", 200),
        (promotion, "source_images", 200),
        (promotion, "minimum_macro_asr_gain", 0.040),
        (promotion, "minimum_macro_auc_gain", 0.015),
        (promotion, "required_target_calls", 0),
        (final, "entry_condition", "stage_c_promotion_passed"),
        (final, "final_source_test_images", 1_000),
        (final, "minimum_macro_asr_gain", 0.050),
        (final, "minimum_macro_auc_gain", 0.020),
        (final, "maximum_exact_sign_flip_pvalue", 0.050),
        (final, "minimum_hybrid_ablation_asr_gain", 0.010),
        (final, "required_target_calls", 0),
        (final, "run_once", True),
        (final, "replace_failed_seeds", False),
        (final, "settings_change_after_observation", False),
    )
    for parent, key, expected in exact_requirements:
        if parent.get(key) != expected:
            raise ValueError(f"locked confirmatory rule changed: {key}")


def validate_confirmatory_contract_payload(
    payload: Mapping[str, Any],
) -> ConfirmatoryContract:
    """Validate exact lock identity and return an immutable contract view."""

    if not isinstance(payload, Mapping):
        raise ValueError("confirmatory contract must be a JSON object")
    digest = canonical_contract_digest(payload)
    if digest != CONFIRMATORY_CONTRACT_SHA256:
        raise ValueError(
            "confirmatory contract does not match its locked digest"
        )
    if payload.get("schema_version") != 1:
        raise ValueError("confirmatory contract schema version changed")
    if payload.get("status") != "locked_before_stage_b_results":
        raise ValueError("confirmatory contract is not locked before Stage B")
    contract_name = _string(payload, "contract_name")
    dataset = _object(payload, "dataset")
    (
        split_seed,
        victim_fit,
        policy_train,
        validation,
        outer_test,
        complement,
        base_split_digest,
        role_digests,
    ) = _validate_dataset_contract(dataset)
    phase1, stage_b, stage_c, final = _validate_seed_contract(
        _object(payload, "seeds")
    )
    (
        algorithm,
        final_count,
        final_digest,
        candidate_digest,
        allowed,
        prohibited,
    ) = _validate_final_source_test_contract(
        _object(payload, "final_source_test"),
        split_seed=split_seed,
        complement=complement,
    )
    rules = _object(payload, "rules")
    _validate_rules(rules)
    return ConfirmatoryContract(
        contract_name=contract_name,
        payload_digest=digest,
        phase1_seeds=phase1,
        stage_b_seeds=stage_b,
        stage_c_seeds=stage_c,
        final_confirmatory_seeds=final,
        split_seed=split_seed,
        victim_fit_images=victim_fit,
        policy_train_images=policy_train,
        source_validation_images=validation,
        outer_test_images=outer_test,
        untouched_complement_images=complement,
        base_split_digest=base_split_digest,
        phase1_role_digests=role_digests,
        final_source_test_algorithm=algorithm,
        final_source_test_count=final_count,
        final_source_test_digest=final_digest,
        final_source_candidate_digest=candidate_digest,
        final_source_test_allowed_stage=allowed,
        final_source_test_prohibited_stages=prohibited,
        rules=_freeze_json(dict(rules)),
    )


def load_confirmatory_contract(
    path: Path = DEFAULT_CONFIRMATORY_CONTRACT_PATH,
) -> ConfirmatoryContract:
    """Load a duplicate-key-safe JSON file and verify the immutable lock."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError("confirmatory contract is not valid JSON") from error
    return validate_confirmatory_contract_payload(payload)


def _validated_label_buckets(
    train_labels: Sequence[int],
) -> dict[int, list[int]]:
    if len(train_labels) != 50_000:
        raise ValueError("official CIFAR-10 must contain exactly 50,000 labels")
    buckets = {label: [] for label in range(10)}
    for index, raw_label in enumerate(train_labels):
        if (
            not isinstance(raw_label, int)
            or isinstance(raw_label, bool)
            or raw_label not in buckets
        ):
            raise ValueError("CIFAR-10 labels must be integers in [0, 9]")
        buckets[raw_label].append(index)
    if any(len(indices) != 5_000 for indices in buckets.values()):
        raise ValueError("official CIFAR-10 must have 5,000 labels per class")
    return buckets


def _validated_allocation(
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    contract: ConfirmatoryContract,
) -> tuple[int, ...]:
    if contract.final_source_test_algorithm != _FINAL_ALLOCATION_ALGORITHM:
        raise ValueError("unsupported final source-test allocation algorithm")
    buckets = _validated_label_buckets(train_labels)
    split = build_cifar_split(
        train_labels,
        test_labels,
        contract.victim_fit_images,
        contract.policy_train_images,
        contract.source_validation_images,
        contract.outer_test_images,
        contract.split_seed,
    )
    if split.digest != contract.base_split_digest:
        raise ValueError("base CIFAR split digest does not match contract")
    role_indices = {
        "victim_fit": split.victim_fit,
        "policy_train": split.policy_train,
        "source_validation": split.source_validation,
        "outer_test": split.outer_test,
    }
    for role, indices in role_indices.items():
        if indices_digest(indices) != contract.phase1_role_digests[role]:
            raise ValueError(f"{role} digest does not match contract")
    train_roles = (
        split.victim_fit,
        split.policy_train,
        split.source_validation,
    )
    train_role_sets = tuple(set(role) for role in train_roles)
    if any(
        left.intersection(right)
        for offset, left in enumerate(train_role_sets)
        for right in train_role_sets[offset + 1 :]
    ):
        raise ValueError("Phase 1 train roles are not disjoint")
    used = set().union(*train_role_sets)
    exact_complement = set(range(len(train_labels))) - used
    if (
        len(used)
        != (
            contract.victim_fit_images
            + contract.policy_train_images
            + contract.source_validation_images
        )
        or len(exact_complement) != contract.untouched_complement_images
    ):
        raise ValueError("Phase 1 train roles do not leave the locked complement")

    candidates: list[int] = []
    selected: list[int] = []
    per_class = contract.final_source_test_count // 10
    for label in range(10):
        split_order = list(buckets[label])
        random.Random(contract.split_seed + label).shuffle(split_order)
        class_candidates = [
            index for index in split_order if index in exact_complement
        ]
        candidates.extend(class_candidates)
        selection_order = list(class_candidates)
        random.Random(
            contract.split_seed + 40_000 + label
        ).shuffle(selection_order)
        selected.extend(selection_order[:per_class])
    if (
        len(candidates) != len(set(candidates))
        or set(candidates) != exact_complement
    ):
        raise ValueError("allocator candidates are not the exact complement")
    random.Random(contract.split_seed + 50_000).shuffle(selected)
    candidate_tuple = tuple(candidates)
    selected_tuple = tuple(selected)
    if (
        indices_digest(candidate_tuple)
        != contract.final_source_candidate_digest
    ):
        raise ValueError("untouched complement digest does not match contract")
    if indices_digest(selected_tuple) != contract.final_source_test_digest:
        raise ValueError("final source-test digest does not match contract")
    class_counts = Counter(train_labels[index] for index in selected_tuple)
    expected = Counter({label: per_class for label in range(10)})
    if (
        len(selected_tuple) != contract.final_source_test_count
        or len(selected_tuple) != len(set(selected_tuple))
        or class_counts != expected
        or not set(selected_tuple).issubset(exact_complement)
    ):
        raise ValueError("final source-test allocation is invalid")
    return selected_tuple


def allocate_final_source_test_indices(
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    contract: ConfirmatoryContract,
    *,
    stage: str,
) -> tuple[int, ...]:
    """Return the locked indices only to the final confirmatory source stage."""

    assert_final_source_test_access(contract, stage)
    return _validated_allocation(
        train_labels,
        test_labels,
        contract,
    )


def validate_final_source_test_indices(
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    contract: ConfirmatoryContract,
    *,
    stage: str,
) -> tuple[int, ...]:
    """Recompute and digest-check the preregistered final source-test indices."""

    assert_final_source_test_access(contract, stage)
    return _validated_allocation(
        train_labels,
        test_labels,
        contract,
    )


def assert_final_source_test_access(
    contract: ConfirmatoryContract,
    stage: str,
) -> None:
    """Fail closed when a stage tries to evaluate the sealed final data."""

    known = {
        contract.final_source_test_allowed_stage,
        *contract.final_source_test_prohibited_stages,
    }
    if stage not in known:
        raise ValueError(f"unknown research stage: {stage}")
    if stage != contract.final_source_test_allowed_stage:
        raise PermissionError(
            "the final source-test allocation is sealed until the "
            "final confirmatory source study"
        )
