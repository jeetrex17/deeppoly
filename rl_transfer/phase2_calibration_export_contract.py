"""Locked request and provenance contract for D0 evidence export."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Mapping

from .phase1_export_validation import (
    digest,
    finite_number,
    nonnegative_integer,
    require_mapping,
)
from .phase2_calibration_manifest import (
    CALIBRATION_MAX_SECONDS,
    CALIBRATION_POLICY_SEEDS,
    CALIBRATION_TEMPERATURES,
    FOLDS,
)


_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_OUTPUT_ROOT = "output/rl_transfer/cifar10_rtx_phase2_calibration"
_SOURCE_ROOT = "output/rl_transfer/cifar10_rtx_phase2_screen"
_SOURCE_MANIFEST = f"{_SOURCE_ROOT}/screen_manifest.json"
_EXPECTED_IDENTIFIERS = {
    "calibration_code_digest": (
        "d4134959fa11aadf2495d5ee51ca5eee9b18fbb7978a2eac284d38392dc9ffe1"
    ),
    "calibration_git_revision": "59f4d0691d621206a9120e9f2a0b8d46e41945d7",
    "dataset_content_sha256": (
        "c1adf901d7d67ca1df1a1d0d5ae49a079ed82d7ac742568da8041aa60a54f9b7"
    ),
    "source_manifest_sha256": (
        "efd96c5775187ac29fbd1453e3d1654d26373fc17b7c0d22b0e4955215a0e054"
    ),
}
_EXPECTED_WORKTREE = {
    "dirty": True,
    "status_sha256": (
        "ca80a0a5c3148b1bad6e609c96d94487453f977535771a5f18482a3c691f4cac"
    ),
}
_EXPECTED_RUNTIME = {
    "python_version": "3.12.3",
    "torch_version": "2.13.0+cu130",
    "torchvision_version": "0.28.0+cu130",
    "cuda_runtime_version": "13.0",
    "cudnn_version": 92000,
    "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
    "gpu_total_memory_bytes": 11_336_482_816,
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
_EXPECTED_REQUEST = {
    "source_manifest": _SOURCE_MANIFEST,
    "source_root": _SOURCE_ROOT,
    "output_dir": _OUTPUT_ROOT,
    "data_root": "data/cifar10",
    "seeds": list(CALIBRATION_POLICY_SEEDS),
    "folds": list(FOLDS),
    "temperatures": list(CALIBRATION_TEMPERATURES),
    "deadline_seconds": CALIBRATION_MAX_SECONDS,
    "device": "cuda",
    "download": False,
}
_MANIFEST_FIELDS = {
    "schema_version",
    "name",
    "status",
    "diagnostic_only",
    "research_valid",
    "publication_candidate",
    "exploratory_screen",
    "training_performed",
    "calibration_code_digest",
    "calibration_git_revision",
    "calibration_git_worktree",
    "runtime_environment",
    "request",
    "dataset_version",
    "dataset_content_sha256",
    "source_manifest",
    "source_manifest_sha256",
    "fold_summaries",
    "calibration_decision",
    "completed_folds",
    "partial_folds",
    "selected_folds",
    "source_model_calls",
    "target_calls",
    "target_evaluation_available",
    "target_evaluation_performed",
    "results_path",
    "results_sha256",
    "query_traces_path",
    "query_traces_sha256",
    "elapsed_seconds",
    "deadline_seconds",
    "deadline_reached",
}
_RUNTIME_FIELDS = {
    "python_version",
    "torch_version",
    "torchvision_version",
    "cuda_runtime_version",
    "cudnn_version",
    "gpu_name",
    "gpu_total_memory_bytes",
    "deterministic_algorithms",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "environment_sha256",
}


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_calibration_manifest_contract(
    manifest: Mapping[str, object],
) -> None:
    """Fail closed unless request and provenance match the locked D0 run."""

    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("calibration manifest schema is not locked")
    request = require_mapping(manifest.get("request"), "calibration request")
    if request != _EXPECTED_REQUEST:
        raise ValueError("calibration request does not match the locked D0 protocol")

    digest(manifest.get("calibration_code_digest"), "calibration code digest")
    digest(manifest.get("dataset_content_sha256"), "dataset content digest")
    digest(manifest.get("source_manifest_sha256"), "source manifest digest")
    if any(
        manifest.get(field) != expected
        for field, expected in _EXPECTED_IDENTIFIERS.items()
    ):
        raise ValueError("calibration provenance identifier is not release-anchored")
    revision = manifest.get("calibration_git_revision")
    if not isinstance(revision, str) or _GIT_REVISION.fullmatch(revision) is None:
        raise ValueError("calibration Git revision must be a lowercase commit ID")

    worktree = require_mapping(
        manifest.get("calibration_git_worktree"),
        "calibration Git worktree",
    )
    if (
        set(worktree) != {"dirty", "status_sha256"}
        or not isinstance(worktree.get("dirty"), bool)
        or worktree != _EXPECTED_WORKTREE
    ):
        raise ValueError("calibration Git worktree provenance is invalid")
    digest(worktree.get("status_sha256"), "calibration worktree status digest")

    runtime = require_mapping(
        manifest.get("runtime_environment"),
        "calibration runtime environment",
    )
    if (
        set(runtime) != _RUNTIME_FIELDS
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("cudnn_deterministic") is not True
        or runtime.get("cudnn_benchmark") is not False
        or any(
            not _nonempty_string(runtime.get(field), f"runtime {field}")
            for field in (
                "python_version",
                "torch_version",
                "torchvision_version",
                "cuda_runtime_version",
                "gpu_name",
            )
        )
        or nonnegative_integer(
            runtime.get("cudnn_version"),
            "runtime cuDNN version",
        )
        <= 0
        or nonnegative_integer(
            runtime.get("gpu_total_memory_bytes"),
            "runtime GPU memory",
        )
        <= 0
    ):
        raise ValueError("calibration runtime provenance is invalid")
    digest(runtime.get("environment_sha256"), "runtime environment digest")
    runtime_payload = {
        key: value for key, value in runtime.items() if key != "environment_sha256"
    }
    recomputed_environment_digest = hashlib.sha256(
        json.dumps(
            runtime_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        runtime_payload != _EXPECTED_RUNTIME
        or runtime.get("environment_sha256") != recomputed_environment_digest
    ):
        raise ValueError("calibration runtime provenance is not release-anchored")

    deadline = finite_number(
        manifest.get("deadline_seconds"),
        "calibration deadline",
    )
    elapsed = finite_number(
        manifest.get("elapsed_seconds"),
        "calibration elapsed time",
    )
    if (
        deadline != CALIBRATION_MAX_SECONDS
        or not 0 <= elapsed <= deadline
        or manifest.get("partial_folds") != 0
        or manifest.get("source_manifest") != _SOURCE_MANIFEST
        or manifest.get("results_path") != f"{_OUTPUT_ROOT}/calibration_results.jsonl"
        or manifest.get("query_traces_path")
        != f"{_OUTPUT_ROOT}/calibration_query_traces.jsonl"
        or manifest.get("dataset_version")
        != f"torchvision-{runtime['torchvision_version']}"
    ):
        raise ValueError("calibration runtime or artifact provenance is inconsistent")
    nonnegative_integer(
        manifest.get("source_model_calls"),
        "calibration source-model calls",
    )


__all__ = ("validate_calibration_manifest_contract",)
