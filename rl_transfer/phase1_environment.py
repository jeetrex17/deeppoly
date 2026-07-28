"""Sanitized dependency and hardware provenance for Phase 1."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping

from .artifacts import sha256_file
from .paths import REPOSITORY_ROOT, resolve_descendant
from .phase1_export_validation import (
    digest,
    nonnegative_integer,
    require_mapping,
    validate_portable_value,
)


_PIN_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)=="
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)"
)
_EDITABLE_PATTERN = re.compile(
    r"-e git\+https://github\.com/"
    r"(?P<repository>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\.git@"
    r"(?P<commit>[0-9a-f]{40})#egg="
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)"
)
_AUDITED_GPU_MODEL = "NVIDIA GeForce RTX 2080 Ti"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sanitized_freeze(
    freeze_path: Path,
    expected_sha256: str,
) -> tuple[str, dict[str, str], dict[str, str]]:
    if not freeze_path.is_file() or sha256_file(freeze_path) != expected_sha256:
        raise ValueError("dependency freeze checksum verification failed")
    try:
        source = freeze_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("dependency freeze must be UTF-8") from error
    packages: dict[str, str] = {}
    editable: dict[str, str] | None = None
    canonical_lines: list[str] = []
    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            raise ValueError("dependency freeze contains an empty record")
        pinned = _PIN_PATTERN.fullmatch(line)
        editable_match = _EDITABLE_PATTERN.fullmatch(line)
        if pinned is None and editable_match is None:
            raise ValueError(f"dependency freeze line {line_number} is not a safe pin")
        match = pinned if pinned is not None else editable_match
        assert match is not None
        normalized_name = match.group("name").casefold().replace("_", "-")
        if normalized_name in packages:
            raise ValueError("dependency freeze contains a duplicate package")
        packages[normalized_name] = (
            match.group("version") if pinned is not None else "editable"
        )
        if editable_match is not None:
            if editable is not None:
                raise ValueError(
                    "dependency freeze contains multiple editable installs"
                )
            editable = {
                "repository": (
                    "https://github.com/" + editable_match.group("repository")
                ),
                "commit": editable_match.group("commit"),
                "package": editable_match.group("name"),
            }
        canonical_lines.append(line)
    if editable is None:
        raise ValueError("dependency freeze is missing the editable code pin")
    for required in ("torch", "torchvision"):
        if required not in packages:
            raise ValueError(
                f"dependency freeze is missing required package: {required}"
            )
    return "\n".join(canonical_lines) + "\n", packages, editable


def verified_environment_evidence(
    study_root: Path,
    study: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    """Verify and sanitize run-start dependency and runtime evidence."""

    runtime = require_mapping(
        study.get("runtime_environment"),
        "runtime environment",
    )
    expected_freeze_sha = digest(
        runtime.get("pip_freeze_sha256"),
        "dependency freeze checksum",
    )
    freeze_path = resolve_descendant(
        study_root,
        "pip_freeze.txt",
        label="dependency freeze",
    )
    freeze, packages, editable = _sanitized_freeze(
        freeze_path,
        expected_freeze_sha,
    )
    requirements_path = resolve_descendant(
        REPOSITORY_ROOT,
        "requirements/rtx-publication.txt",
        label="RTX requirements",
    )
    expected_requirements_sha = digest(
        runtime.get("requirements_sha256"),
        "RTX requirements checksum",
    )
    if (
        not requirements_path.is_file()
        or sha256_file(requirements_path) != expected_requirements_sha
    ):
        raise ValueError("RTX requirements checksum verification failed")
    study_git_revision = runtime.get("git_revision")
    if study_git_revision is not None and (
        not isinstance(study_git_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", study_git_revision) is None
    ):
        raise ValueError("study git revision is malformed")
    environment = {
        "schema_version": 1,
        "run_start_manifest": {
            "cuda_runtime": str(runtime["cuda_runtime"]),
            "cudnn_version": nonnegative_integer(
                runtime.get("cudnn_version"),
                "cuDNN version",
            ),
            "nvidia_driver": str(runtime["nvidia_driver"]),
            "gpu_model": None,
            "git_revision": study_git_revision,
        },
        "dependencies": {
            "freeze_file": "dependency_freeze.txt",
            "package_count": len(packages),
            "pip_freeze_sha256": expected_freeze_sha,
            "sanitized_freeze_sha256": _sha256_text(freeze),
            "requirements_file": "requirements/rtx-publication.txt",
            "requirements_sha256": expected_requirements_sha,
            "torch": packages["torch"],
            "torchvision": packages["torchvision"],
        },
        "code_mapping": {
            "editable_install_repository": editable["repository"],
            "editable_install_commit": editable["commit"],
            "editable_install_package": editable["package"],
            "study_git_revision": study_git_revision,
            "basis": "editable install pin in the verified dependency freeze",
            "limitation": (
                "The study run-start environment record did not provide a "
                "git revision; the commit mapping comes from the editable "
                "dependency pin."
            ),
        },
        "post_run_workstation_audit": {
            "gpu_model": _AUDITED_GPU_MODEL,
            "timing": "after study completion",
            "same_workstation_reported": True,
            "recorded_at_run_start": False,
            "source": ("operator-reported NVIDIA-SMI audit on the study workstation"),
        },
    }
    validate_portable_value(environment, "environment evidence")
    return freeze, environment
