"""Stable CIFAR victim-cache contract and fingerprint helpers."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch

from .cifar_config import MacPilotConfig
from .cifar_training import train_classifier


def victim_cache_contract(
    config: MacPilotConfig,
    split_digest: str,
    dataset_version: str,
    victim_code_digest: str,
    device_type: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": config.dataset,
        "dataset_version": dataset_version,
        "split_digest": split_digest,
        "victim_seed": (
            config.victim_seed if config.victim_seed is not None else config.seed
        ),
        "victim_profile": config.victim_profile,
        "victim_train_images": config.victim_train_images,
        "source_validation_images": config.source_validation_images,
        "victim_epochs": config.victim_epochs,
        "victim_learning_rate": config.victim_learning_rate,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "victim_code_digest": victim_code_digest,
        "device_type": device_type,
        "torch_version": torch.__version__,
    }


def victim_cache_digest(
    config: MacPilotConfig,
    split_digest: str,
    dataset_version: str,
    victim_code_digest: str,
    device_type: str,
) -> str:
    """Fingerprint only inputs that can change victim fitting."""

    payload = victim_cache_contract(
        config,
        split_digest,
        dataset_version,
        victim_code_digest,
        device_type,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def victim_code_digest() -> str:
    hasher = hashlib.sha256()
    package_root = Path(__file__).parent
    hasher.update((package_root / "cifar_models.py").read_bytes())
    hasher.update((package_root / "reproducibility.py").read_bytes())
    hasher.update(inspect.getsource(train_classifier).encode("utf-8"))
    return hasher.hexdigest()
