"""Deterministic CIFAR data partitions and sample extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CIFARSplit:
    victim_fit: tuple[int, ...]
    policy_train: tuple[int, ...]
    source_validation: tuple[int, ...]
    outer_test: tuple[int, ...]
    digest: str


def _class_buckets(labels: Sequence[int]) -> dict[int, list[int]]:
    buckets = {label: [] for label in range(10)}
    for index, label in enumerate(labels):
        if label not in buckets:
            raise ValueError("CIFAR-10 labels must be integers in [0, 9]")
        buckets[label].append(index)
    if any(not indices for indices in buckets.values()):
        raise ValueError("every CIFAR-10 class must be present")
    return buckets


def build_cifar_split(
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    victim_fit_count: int,
    policy_train_count: int,
    source_validation_count: int,
    outer_test_count: int,
    seed: int,
) -> CIFARSplit:
    counts = (
        victim_fit_count,
        policy_train_count,
        source_validation_count,
        outer_test_count,
    )
    if any(count <= 0 or count % 10 for count in counts):
        raise ValueError("CIFAR split counts must be positive multiples of ten")
    train_buckets = _class_buckets(train_labels)
    test_buckets = _class_buckets(test_labels)
    train_per_class = tuple(count // 10 for count in counts[:3])
    test_per_class = outer_test_count // 10
    if any(
        len(train_buckets[label]) < sum(train_per_class)
        for label in range(10)
    ):
        raise ValueError("insufficient train examples for requested stratified split")
    if any(
        len(test_buckets[label]) < test_per_class
        for label in range(10)
    ):
        raise ValueError("insufficient test examples for requested stratified split")
    roles = [[], [], []]
    outer: list[int] = []
    for label in range(10):
        train_indices = list(train_buckets[label])
        test_indices = list(test_buckets[label])
        random.Random(seed + label).shuffle(train_indices)
        random.Random(seed + 10_000 + label).shuffle(test_indices)
        start = 0
        for role, size in zip(roles, train_per_class):
            role.extend(train_indices[start:start + size])
            start += size
        outer.extend(test_indices[:test_per_class])
    for offset, role in enumerate(roles):
        random.Random(seed + 20_000 + offset).shuffle(role)
    random.Random(seed + 30_000).shuffle(outer)
    ordered_roles = tuple(tuple(role) for role in roles)
    ordered_outer = tuple(outer)
    encoded = json.dumps(
        (*ordered_roles, ordered_outer),
        separators=(",", ":"),
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return CIFARSplit(*ordered_roles, ordered_outer, digest)


def dataset_samples(
    dataset: Dataset,
    indices: Sequence[int],
) -> tuple[tuple[torch.Tensor, int], ...]:
    return tuple(
        (dataset[index][0].float(), int(dataset[index][1]))
        for index in indices
    )


def indices_digest(indices: Sequence[int]) -> str:
    encoded = json.dumps(
        tuple(int(index) for index in indices),
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def balanced_subset_indices(
    dataset: Dataset,
    indices: Sequence[int],
    count: int,
) -> tuple[int, ...]:
    if count <= 0 or count % 10:
        raise ValueError(
            "balanced CIFAR subset count must be a positive multiple of ten"
        )
    per_class = count // 10
    selected: list[int] = []
    class_counts = {label: 0 for label in range(10)}
    for index in indices:
        label = int(dataset[index][1])
        if class_counts[label] < per_class:
            selected.append(int(index))
            class_counts[label] += 1
        if len(selected) == count:
            break
    if any(value != per_class for value in class_counts.values()):
        raise ValueError(
            "source validation split cannot supply the balanced subset"
        )
    return tuple(selected)


def disjoint_balanced_subsets(
    dataset: Dataset,
    indices: Sequence[int],
    counts: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Allocate ordered, disjoint, class-balanced subsets."""

    requested = tuple(counts)
    if (
        not requested
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or count % 10
            for count in requested
        )
    ):
        raise ValueError(
            "subset counts must be positive multiples of ten"
        )
    remaining = tuple(int(index) for index in indices)
    subsets: list[tuple[int, ...]] = []
    for count in requested:
        selected = balanced_subset_indices(
            dataset,
            remaining,
            count,
        )
        selected_set = frozenset(selected)
        remaining = tuple(
            index for index in remaining if index not in selected_set
        )
        subsets.append(selected)
    flattened = tuple(index for subset in subsets for index in subset)
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("disjoint subset allocation produced overlap")
    return tuple(subsets)
