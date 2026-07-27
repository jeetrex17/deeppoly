"""Locked GPU environment checks and run-start provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import torch

from .paths import resolve_descendant


PROTOCOL_PATHS = (
    "rl_transfer",
    "configs/rl_transfer",
    "requirements",
    "tests",
    "scripts/build_gpu_research_notebook.py",
    "pyproject.toml",
)


def require_clean_protocol_tree(repository: Path) -> None:
    """Reject uncommitted changes to anything that can affect results."""

    status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *PROTOCOL_PATHS,
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError(
            "the locked study requires clean committed protocol files"
        )


def capture_runtime_environment(
    study_dir: Path,
    repository: Path,
    requirements_path: Path,
) -> dict[str, object]:
    """Write and describe the exact package environment at run start."""

    freeze = subprocess.run(
        (sys.executable, "-m", "pip", "freeze"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    study_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = resolve_descendant(
        study_dir,
        "pip_freeze.txt",
        label="run-start package snapshot",
    )
    freeze_path.write_text(freeze)
    driver = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "pip_freeze_sha256": hashlib.sha256(
            freeze.encode("utf-8")
        ).hexdigest(),
        "pip_freeze_path": str(
            freeze_path.relative_to(repository)
        ),
        "requirements_sha256": hashlib.sha256(
            requirements_path.read_bytes()
        ).hexdigest(),
        "nvidia_driver": (
            driver.stdout.strip().splitlines()[0]
            if driver.returncode == 0 and driver.stdout.strip()
            else "unavailable"
        ),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
