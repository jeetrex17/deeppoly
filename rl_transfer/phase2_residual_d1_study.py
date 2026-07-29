"""Persistent, shared-deadline orchestration for the complete D1 study."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import re
import time

from torch.utils.data import Dataset

from .artifacts import exclusive_file_lock, sha256_file
from .cifar_manifest import code_digest, git_revision, git_worktree_state
from .phase2_residual_d1 import (
    D1_MAX_SECONDS,
    ResidualD1Request,
    validate_source_only_payload as _source_only,
)
from .phase2_residual_d1_runner import (
    ResidualD1Deadline,
    build_residual_d1_dry_run,
    run_residual_d1_from_datasets,
)
from .phase2_residual_d1b_artifacts import canonical_json_digest
from .verified_artifacts import load_verified_json, write_verified_json


_DIGEST = re.compile(r"[0-9a-f]{64}")
_ROOT_FILES = {
    ".study.lock",
    "study_manifest.json",
    "study_manifest.json.sha256",
    "d1a",
    "d1b",
}


class ResidualD1StudyDeadline(ResidualD1Deadline):
    """Raised when the persisted study-wide deadline is reached."""


@dataclass(frozen=True)
class ResidualD1StudyStages:
    """Injectable stage entry points used by synthetic lifecycle tests."""

    run_d1a: Callable[..., Mapping[str, object]]
    run_d1b: Callable[..., Mapping[str, object]]

    def __post_init__(self) -> None:
        if not callable(self.run_d1a) or not callable(self.run_d1b):
            raise TypeError("D1 study stages must be callable")


def _production_stages() -> ResidualD1StudyStages:
    from .phase2_residual_d1b_runner import run_residual_d1b_from_datasets

    return ResidualD1StudyStages(
        run_d1a=run_residual_d1_from_datasets,
        run_d1b=run_residual_d1b_from_datasets,
    )


def build_residual_d1_study_dry_run(
    request: ResidualD1Request,
    source: Mapping[str, object],
) -> dict[str, object]:
    """Describe the complete conditional source-only study without ML work."""

    d1a = build_residual_d1_dry_run(request, source)
    return {
        "schema_version": 1,
        "name": "phase2-d1-residual-ranker-study",
        "mode": "dry_run",
        "study_root": str(request.output_dir.parent),
        "d1a_output": str(request.output_dir),
        "d1b_output": str(request.output_dir.parent / "d1b"),
        "shared_persisted_deadline_seconds": request.deadline_seconds,
        "d1a": d1a,
        "d1b": {
            "conditional_on_d1a_gate": True,
            "ppo_blocks": 4,
            "ppo_episodes_per_block": 50,
            "ppo_episodes_maximum": 200,
            "evaluation_methods": [
                "score_greedy",
                "residual_ranker_bc",
                "residual_ranker_bc_ppo",
            ],
            "evaluation_role": "reserved_d1b_second_50_images",
        },
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _seal() -> dict[str, object]:
    return {
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _study_root(request: ResidualD1Request) -> Path:
    if request.output_dir.name != "d1a":
        raise ValueError("D1a output must be the study-root/d1a directory")
    root = request.output_dir.parent.resolve()
    if request.output_dir != root / "d1a":
        raise ValueError("D1 study directory routing is invalid")
    return root


def _validate_root(root: Path) -> None:
    names = {path.name for path in root.iterdir()}
    unexpected = names - _ROOT_FILES
    if unexpected:
        raise ValueError(
            f"D1 study root contains unexpected artifacts: {sorted(unexpected)}"
        )
    for name in ("d1a", "d1b"):
        path = root / name
        if path.is_symlink():
            raise ValueError("D1 stage directories cannot be symlinks")


def _base_control(
    request: ResidualD1Request,
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
    now: float,
) -> dict[str, object]:
    environment_digest = _digest(
        runtime_environment.get("environment_sha256"),
        "D1 runtime environment digest",
    )
    source_digest = sha256_file(request.source_manifest)
    return {
        "schema_version": 1,
        "name": "phase2-d1-residual-ranker-study",
        "status": "running",
        "diagnostic_only": True,
        "research_valid": False,
        "publication_candidate": False,
        "request_sha256": request.digest(),
        "source_manifest_sha256": source_digest,
        "dataset_version": dataset_version,
        "dataset_content_sha256": dataset_content_sha256,
        "runtime_environment_sha256": environment_digest,
        "code_digest": code_digest(),
        "git_revision": git_revision(),
        "git_worktree_state": git_worktree_state(),
        "deadline_seconds": request.deadline_seconds,
        "started_epoch_seconds": now,
        "deadline_epoch_seconds": now + request.deadline_seconds,
        "last_observed_epoch_seconds": now,
        "elapsed_seconds": 0.0,
        "d1a_directory": "d1a",
        "d1b_directory": "d1b",
        "d1a_source_gate_passed": None,
        "study_outcome": None,
        **_seal(),
    }


def _validate_control(
    control: Mapping[str, object],
    request: ResidualD1Request,
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
    now: float,
) -> dict[str, object]:
    _source_only(control, "D1 study manifest")
    started = _number(
        control.get("started_epoch_seconds"),
        "D1 study start time",
    )
    deadline = _number(
        control.get("deadline_epoch_seconds"),
        "D1 study deadline",
    )
    duration = _number(
        control.get("deadline_seconds"),
        "D1 study duration",
    )
    last_observed = _number(
        control.get("last_observed_epoch_seconds"),
        "D1 study last-observed time",
    )
    expected = {
        "schema_version": 1,
        "name": "phase2-d1-residual-ranker-study",
        "request_sha256": request.digest(),
        "source_manifest_sha256": sha256_file(request.source_manifest),
        "dataset_version": dataset_version,
        "dataset_content_sha256": dataset_content_sha256,
        "runtime_environment_sha256": runtime_environment.get("environment_sha256"),
        "code_digest": code_digest(),
        "git_revision": git_revision(),
        "git_worktree_state": git_worktree_state(),
        "deadline_seconds": request.deadline_seconds,
        "d1a_directory": "d1a",
        "d1b_directory": "d1b",
    }
    if any(control.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "D1 study request, source, dataset, runtime, code, or Git provenance changed"
        )
    if (
        not 0 < duration <= D1_MAX_SECONDS
        or not math.isclose(deadline - started, duration, abs_tol=1e-6)
        or last_observed < started
    ):
        raise ValueError("D1 study persisted deadline is malformed")
    if now < started or now < last_observed:
        raise ValueError("D1 study clock rollback detected")
    return dict(control)


def _load_or_create_control(
    root: Path,
    request: ResidualD1Request,
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
    now: float,
) -> dict[str, object]:
    manifest_path = root / "study_manifest.json"
    checksum_path = root / "study_manifest.json.sha256"
    states = manifest_path.is_file(), checksum_path.is_file()
    if states == (False, False):
        if any(
            path.exists() and any(path.iterdir())
            for path in (root / "d1a", root / "d1b")
        ):
            raise ValueError("D1 child artifacts exist without a study manifest")
        control = _base_control(
            request,
            dataset_version=dataset_version,
            dataset_content_sha256=dataset_content_sha256,
            runtime_environment=runtime_environment,
            now=now,
        )
        write_verified_json(manifest_path, control)
        return control
    if states != (True, True):
        raise ValueError("D1 study manifest artifact pair is incomplete")
    return _validate_control(
        load_verified_json(manifest_path),
        request,
        dataset_version=dataset_version,
        dataset_content_sha256=dataset_content_sha256,
        runtime_environment=runtime_environment,
        now=now,
    )


def _child_record(
    root: Path,
    relative_path: str,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    path = root / relative_path
    if path.is_symlink() or not path.is_file():
        raise ValueError("D1 child manifest is missing or unsafe")
    return {
        "path": relative_path,
        "status": manifest.get("status"),
        "file_sha256": sha256_file(path),
        "canonical_sha256": canonical_json_digest(manifest),
    }


def _verify_complete_children(
    root: Path,
    study: Mapping[str, object],
) -> None:
    if study.get("status") != "complete":
        raise ValueError("D1 complete-child verifier requires a complete study")
    expected_paths = {
        "d1a": "d1a/d1_manifest.json",
        "d1b": "d1b/d1b_manifest.json",
    }
    for stage, relative in expected_paths.items():
        recorded = _mapping(study.get(stage), f"D1 {stage} child record")
        if recorded.get("path") != relative:
            raise ValueError("D1 child manifest path binding mismatch")
        child_path = root / relative
        if child_path.is_symlink():
            raise ValueError("D1 child manifest cannot be a symlink")
        child = load_verified_json(child_path)
        _source_only(child, f"D1 {stage} child manifest")
        if (
            recorded.get("status") != child.get("status")
            or recorded.get("file_sha256") != sha256_file(child_path)
            or recorded.get("canonical_sha256") != canonical_json_digest(child)
        ):
            raise ValueError("D1 child manifest checksum or status mismatch")
    d1a_status = _mapping(study["d1a"], "D1a child").get("status")
    d1b_status = _mapping(study["d1b"], "D1b child").get("status")
    if d1a_status != "complete" or d1b_status not in {"complete", "skipped"}:
        raise ValueError("D1 complete study has invalid child status")


def _deadline_callback(
    control: Mapping[str, object],
    *,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
) -> Callable[[], None]:
    started = float(control["started_epoch_seconds"])
    last_observed = float(control["last_observed_epoch_seconds"])
    deadline = float(control["deadline_epoch_seconds"])
    loaded_now = wall_clock()
    loaded_monotonic = monotonic_clock()
    monotonic_deadline = loaded_monotonic + max(0.0, deadline - loaded_now)

    def check() -> None:
        now = wall_clock()
        if now < started or now < last_observed:
            raise ValueError("D1 study clock rollback detected")
        if now >= deadline or monotonic_clock() >= monotonic_deadline:
            raise ResidualD1StudyDeadline("persisted D1 study deadline reached")

    return check


def _write_status(
    path: Path,
    control: Mapping[str, object],
    *,
    status: str,
    outcome: str,
    now: float,
    additions: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result = {
        **dict(control),
        **dict(additions or {}),
        "status": status,
        "study_outcome": outcome,
        "last_observed_epoch_seconds": now,
        "elapsed_seconds": max(
            0.0,
            now - float(control["started_epoch_seconds"]),
        ),
        **_seal(),
    }
    write_verified_json(path, result)
    return result


def run_residual_d1_study_from_datasets(
    request: ResidualD1Request,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
    progress: Callable[[str], None] = print,
    wall_clock: Callable[[], float] = time.time,
    monotonic_clock: Callable[[], float] = time.monotonic,
    stages: ResidualD1StudyStages | None = None,
) -> dict[str, object]:
    """Run D1a and conditional D1b under one persisted eight-hour deadline."""

    if not isinstance(request, ResidualD1Request):
        raise TypeError("D1 study requires a residual D1 request")
    _digest(dataset_content_sha256, "D1 dataset digest")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("D1 dataset version is required")
    if not all(
        callable(callback) for callback in (progress, wall_clock, monotonic_clock)
    ):
        raise TypeError("D1 study callbacks must be callable")
    selected_stages = stages or _production_stages()
    root = _study_root(request)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "study_manifest.json"
    with exclusive_file_lock(root / ".study.lock"):
        _validate_root(root)
        now = wall_clock()
        control = _load_or_create_control(
            root,
            request,
            dataset_version=dataset_version,
            dataset_content_sha256=dataset_content_sha256,
            runtime_environment=runtime_environment,
            now=now,
        )
        if control.get("status") == "complete":
            _verify_complete_children(root, control)
            return control
        deadline_check = _deadline_callback(
            control,
            wall_clock=wall_clock,
            monotonic_clock=monotonic_clock,
        )
        try:
            deadline_check()
            running = {
                **control,
                "status": "running",
                "last_observed_epoch_seconds": wall_clock(),
                "study_outcome": None,
                **_seal(),
            }
            write_verified_json(manifest_path, running)
            d1a = dict(
                selected_stages.run_d1a(
                    request,
                    train_dataset,
                    test_dataset,
                    dataset_version=dataset_version,
                    dataset_content_sha256=dataset_content_sha256,
                    runtime_environment=runtime_environment,
                    progress=progress,
                    clock=monotonic_clock,
                    external_deadline_check=deadline_check,
                )
            )
            _source_only(d1a, "D1a returned manifest")
            request.output_dir.mkdir(parents=True, exist_ok=True)
            d1a_path = request.output_dir / "d1_manifest.json"
            write_verified_json(d1a_path, d1a)
            d1a_record = _child_record(
                root,
                "d1a/d1_manifest.json",
                d1a,
            )
            decision = _mapping(d1a.get("d1_decision"), "D1a decision")
            passed = decision.get("passed")
            eligible = decision.get("eligible_for_d1b_source_only_ppo")
            if not isinstance(passed, bool) or eligible is not passed:
                raise ValueError("D1a returned an inconsistent source gate")
            if d1a.get("status") != "complete":
                status = (
                    "deadline_reached"
                    if d1a.get("status") == "deadline_reached"
                    else "failed"
                )
                return _write_status(
                    manifest_path,
                    running,
                    status=status,
                    outcome=f"d1a_{d1a.get('status')}",
                    now=wall_clock(),
                    additions={
                        "d1a": d1a_record,
                        "d1a_source_gate_passed": False,
                    },
                )
            deadline_check()
            d1b_root = root / "d1b"
            d1b = dict(
                selected_stages.run_d1b(
                    request,
                    d1b_root,
                    d1a,
                    train_dataset,
                    test_dataset,
                    dataset_version=dataset_version,
                    dataset_content_sha256=dataset_content_sha256,
                    runtime_environment=runtime_environment,
                    progress=progress,
                    clock=monotonic_clock,
                    deadline_check=deadline_check,
                )
            )
            _source_only(d1b, "D1b returned manifest")
            d1b_root.mkdir(parents=True, exist_ok=True)
            d1b_path = d1b_root / "d1b_manifest.json"
            write_verified_json(d1b_path, d1b)
            d1b_record = _child_record(
                root,
                "d1b/d1b_manifest.json",
                d1b,
            )
            if passed and d1b.get("status") == "skipped":
                raise ValueError("passing D1a cannot silently skip D1b")
            if d1b.get("status") not in {"complete", "skipped"}:
                status = (
                    "deadline_reached"
                    if d1b.get("status") == "deadline_reached"
                    else "failed"
                )
                return _write_status(
                    manifest_path,
                    running,
                    status=status,
                    outcome=f"d1b_{d1b.get('status')}",
                    now=wall_clock(),
                    additions={
                        "d1a": d1a_record,
                        "d1b": d1b_record,
                        "d1a_source_gate_passed": passed,
                    },
                )
            outcome = "d1b_source_study_complete" if passed else "valid_d1a_negative"
            final = _write_status(
                manifest_path,
                running,
                status="complete",
                outcome=outcome,
                now=wall_clock(),
                additions={
                    "d1a": d1a_record,
                    "d1b": d1b_record,
                    "d1a_source_gate_passed": passed,
                    "source_model_calls": (
                        int(d1a.get("source_model_calls", 0))
                        + int(d1b.get("source_model_calls", 0))
                    ),
                },
            )
            _verify_complete_children(root, final)
            return final
        except ResidualD1StudyDeadline:
            return _write_status(
                manifest_path,
                control,
                status="deadline_reached",
                outcome="persisted_study_deadline_reached",
                now=wall_clock(),
            )
        except Exception:
            _write_status(
                manifest_path,
                control,
                status="failed",
                outcome="study_exception",
                now=wall_clock(),
            )
            raise


__all__ = (
    "ResidualD1StudyDeadline",
    "ResidualD1StudyStages",
    "build_residual_d1_study_dry_run",
    "run_residual_d1_study_from_datasets",
)
