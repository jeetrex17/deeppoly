from dataclasses import asdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import torch

from .recurrent import PPOConfig, RecurrentAttackPolicy
from .paths import resolve_descendant


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return resolve_descendant(
        path.parent,
        path.with_suffix(path.suffix + ".sha256"),
        label="checkpoint checksum",
    )


@contextmanager
def exclusive_file_lock(path: Path):
    """Serialize checkpoint readers/writers across local study processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_text_write(path: Path, value: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=".tmp",
            mode="w",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(value)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    digest = sha256_file(path)
    _atomic_text_write(_sidecar_path(path), digest + "\n")
    return digest


def save_recurrent_checkpoint(
    path: Path,
    policy: RecurrentAttackPolicy,
    metadata: Mapping[str, object],
) -> str:
    safe_metadata = json.loads(json.dumps(dict(metadata), sort_keys=True, allow_nan=False))
    payload = {
        "schema_version": 1,
        "observation_dim": policy.observation_dim,
        "action_dim": policy.action_dim,
        "hidden_dim": policy.hidden_dim,
        "ppo_config": asdict(policy.config),
        "model": policy.state_dict(),
        "optimizer": policy.optimizer.state_dict(),
        "metadata": safe_metadata,
    }
    return _atomic_torch_save(path, payload)


def load_recurrent_checkpoint(
    path: Path,
    device: str | torch.device = "cpu",
    *,
    expected_observation_dim: int | None = None,
    expected_action_dim: int | None = None,
    expected_hidden_dim: int | None = None,
    max_file_bytes: int = 128 * 1024 * 1024,
) -> tuple[RecurrentAttackPolicy, dict[str, object]]:
    if (
        not path.is_file()
        or not 0 < path.stat().st_size <= max_file_bytes
    ):
        raise ValueError("recurrent checkpoint size is outside the safe limit")
    expected_digest = _sidecar_path(path).read_text().strip()
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ValueError("checkpoint SHA-256 sidecar is malformed")
    if sha256_file(path) != expected_digest:
        raise ValueError("checkpoint SHA-256 verification failed")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported recurrent checkpoint schema")
    dimensions = (
        int(payload["observation_dim"]),
        int(payload["action_dim"]),
        int(payload["hidden_dim"]),
    )
    expected = (
        expected_observation_dim,
        expected_action_dim,
        expected_hidden_dim,
    )
    if any(
        expected_value is not None and actual != expected_value
        for actual, expected_value in zip(dimensions, expected)
    ):
        raise ValueError("recurrent checkpoint dimensions do not match the run")
    if not (
        1 <= dimensions[0] <= 10_000
        and 1 <= dimensions[1] <= 5_000
        and 1 <= dimensions[2] <= 1_024
    ):
        raise ValueError("recurrent checkpoint dimensions exceed safe limits")
    policy = RecurrentAttackPolicy(
        dimensions[0],
        dimensions[1],
        hidden_dim=dimensions[2],
        config=PPOConfig(**payload["ppo_config"]),
    ).to(device)
    policy.load_state_dict(payload["model"])
    policy.optimizer.load_state_dict(payload["optimizer"])
    return policy, dict(payload["metadata"])


def save_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    metadata: Mapping[str, object],
) -> str:
    safe_metadata = json.loads(json.dumps(dict(metadata), sort_keys=True, allow_nan=False))
    return _atomic_torch_save(
        path,
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "metadata": safe_metadata,
        },
    )


def load_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: str | torch.device,
    *,
    max_file_bytes: int = 1024 * 1024 * 1024,
) -> dict[str, object]:
    if (
        not path.is_file()
        or not 0 < path.stat().st_size <= max_file_bytes
    ):
        raise ValueError("model checkpoint size is outside the safe limit")
    expected_digest = _sidecar_path(path).read_text().strip()
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ValueError("model checkpoint SHA-256 sidecar is malformed")
    if sha256_file(path) != expected_digest:
        raise ValueError("model checkpoint SHA-256 verification failed")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported model checkpoint schema")
    model.load_state_dict(payload["model"])
    return dict(payload["metadata"])
