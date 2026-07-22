from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch

from .recurrent import PPOConfig, RecurrentAttackPolicy


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


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
    _sidecar_path(path).write_text(digest + "\n")
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
) -> tuple[RecurrentAttackPolicy, dict[str, object]]:
    expected_digest = _sidecar_path(path).read_text().strip()
    if sha256_file(path) != expected_digest:
        raise ValueError("checkpoint SHA-256 verification failed")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported recurrent checkpoint schema")
    policy = RecurrentAttackPolicy(
        int(payload["observation_dim"]),
        int(payload["action_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
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
) -> dict[str, object]:
    expected_digest = _sidecar_path(path).read_text().strip()
    if sha256_file(path) != expected_digest:
        raise ValueError("model checkpoint SHA-256 verification failed")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported model checkpoint schema")
    model.load_state_dict(payload["model"])
    return dict(payload["metadata"])
