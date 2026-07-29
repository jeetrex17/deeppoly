"""Remote-only deep verification for a completed D1 study."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .artifacts import (
    exclusive_file_lock,
    load_recurrent_checkpoint,
    sha256_file,
)
from .cifar_manifest import code_digest, git_revision, git_worktree_state
from .phase2_calibration_cli import _runtime_environment
from .phase2_residual_d1 import (
    ResidualD1Request,
    validate_source_only_payload,
)
from .phase2_residual_d1_cache import load_residual_teacher_cache
from .phase2_residual_d1_runner import _verify_complete_d1_children
from .phase2_residual_d1_source import load_d1_source_context
from .phase2_residual_d1_study import (
    _validate_root,
    _verify_complete_children,
)
from .phase2_residual_d1_teacher import (
    D1_HIDDEN_DIM,
    _cache_binding,
)
from .phase2_residual_d1b_artifacts import (
    ResidualD1BBlockStore,
    ResidualD1BStoreBinding,
    canonical_json_digest,
)
from .phase2_residual_d1b_policy import (
    ResidualD1AArtifactBundle,
    build_d1b_source_roles,
    verify_d1a_artifacts,
)
from .phase2_residual_d1b_runner import _validate_output
from .phase2_residual_d1b_verification import verify_complete_d1b_children
from .reproducibility import seed_everything, tree_digest
from .residual_ranker import ResidualRankerPolicy
from .verified_artifacts import load_verified_json


_SOURCE_SEAL = {
    "target_calls": 0,
    "hidden_target_calls": 0,
    "target_evaluation_performed": False,
    "hidden_target_evaluation_performed": False,
    "target_evaluation_available": False,
    "authorizes_hidden_target_evaluation": False,
}
_D1A_PROMOTED_FILES = {
    name
    for stem in (
        "residual_ranker_bc.pt",
        "source_results.jsonl",
        "source_query_traces.jsonl",
        "asr_by_query.svg",
        "final_asr.svg",
    )
    for name in (stem, f"{stem}.sha256")
}
_D1B_PROMOTED_FILES = {
    name
    for stem in (
        "residual_ranker_ppo.pt",
        "source_results.jsonl",
        "source_query_traces.jsonl",
        "asr_by_query.svg",
        "final_asr.svg",
    )
    for name in (stem, f"{stem}.sha256")
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _source_sealed(value: Mapping[str, object], label: str) -> None:
    validate_source_only_payload(value, label)
    if any(value.get(key) != expected for key, expected in _SOURCE_SEAL.items()):
        raise ValueError(f"{label} lacks the explicit source-only seal")


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _assert_no_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("D1 study artifacts cannot contain symlinks")


def _study_file_digests(root: Path) -> dict[str, str]:
    _assert_no_symlinks(root)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=".tmp",
            mode="w",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _external_output(root: Path, path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    resolved = raw.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    raise ValueError(f"{label} must remain outside the study root")


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    info.pax_headers = {}
    return info


def _write_locked_package(
    root: Path,
    archive_path: Path,
    checksums_path: Path,
    expected_files: Mapping[str, str],
) -> dict[str, object]:
    archive = _external_output(root, archive_path, "D1 archive")
    checksums = _external_output(root, checksums_path, "D1 checksum inventory")
    archive.parent.mkdir(parents=True, exist_ok=True)
    checksums.parent.mkdir(parents=True, exist_ok=True)
    if archive.parent != checksums.parent:
        raise ValueError("D1 package outputs must share one control directory")
    current = _study_file_digests(root)
    if current != dict(expected_files):
        raise ValueError("D1 verified artifact snapshot changed before packaging")
    checksum_text = "".join(
        f"{digest}  ./{relative}\n" for relative, digest in sorted(current.items())
    )
    checksum_temporary: Path | None = None
    archive_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=checksums.parent,
            suffix=".tmp",
            mode="w",
            delete=False,
        ) as handle:
            checksum_temporary = Path(handle.name)
            handle.write(checksum_text)
        with tempfile.NamedTemporaryFile(
            dir=archive.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            archive_temporary = Path(handle.name)
        with archive_temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as package:
                    for relative in sorted(current):
                        package.add(
                            root / relative,
                            arcname=f"{root.name}/{relative}",
                            recursive=False,
                            filter=_normalized_tar_info,
                        )
        if _study_file_digests(root) != current:
            raise ValueError("D1 verified artifact snapshot changed during packaging")
        os.replace(checksum_temporary, checksums)
        checksum_temporary = None
        os.replace(archive_temporary, archive)
        archive_temporary = None
    finally:
        for temporary in (checksum_temporary, archive_temporary):
            if temporary is not None and temporary.exists():
                temporary.unlink()
    archive_digest = sha256_file(archive)
    _atomic_text(
        Path(f"{archive}.sha256"),
        f"{archive_digest}  {archive.name}\n",
    )
    return {
        "archive": archive.name,
        "archive_sha256": archive_digest,
        "artifact_checksums": checksums.name,
        "artifact_checksums_sha256": sha256_file(checksums),
        "artifact_file_count": len(current),
        "artifact_tree_sha256": canonical_json_digest(current),
    }


def _request_from_study(
    study_root: Path,
    source_manifest: Path,
    source_root: Path,
    data_root: Path,
    study: Mapping[str, object],
) -> ResidualD1Request:
    deadline = study.get("deadline_seconds")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise ValueError("D1 study deadline is missing")
    request = ResidualD1Request(
        source_manifest=source_manifest,
        source_root=source_root,
        output_dir=study_root / "d1a",
        data_root=data_root,
        deadline_seconds=float(deadline),
    )
    if request.digest() != study.get("request_sha256"):
        raise ValueError("D1 verification request does not match the study")
    return request


def _verify_current_provenance(
    study: Mapping[str, object],
    request: ResidualD1Request,
    runtime_environment: Mapping[str, object],
) -> dict[str, object]:
    worktree = git_worktree_state()
    current_code = code_digest()
    current_git = git_revision()
    if (
        request.source_manifest.is_symlink()
        or request.source_root.is_symlink()
        or request.data_root.is_symlink()
        or any(path.is_symlink() for path in request.source_root.rglob("*"))
        or any(path.is_symlink() for path in request.data_root.rglob("*"))
    ):
        raise ValueError("D1 source and data provenance cannot contain symlinks")
    dataset_digest = tree_digest(request.data_root)
    source_digest = sha256_file(request.source_manifest)
    environment_digest = runtime_environment.get("environment_sha256")
    if (
        worktree.get("dirty") is not False
        or current_code != study.get("code_digest")
        or current_git != study.get("git_revision")
        or dataset_digest != study.get("dataset_content_sha256")
        or source_digest != study.get("source_manifest_sha256")
        or environment_digest != study.get("runtime_environment_sha256")
    ):
        raise ValueError("D1 code, Git, dataset, source, or runtime provenance changed")
    return {
        "git_revision": current_git,
        "code_digest": current_code,
        "dataset_content_sha256": dataset_digest,
        "source_manifest_sha256": source_digest,
        "runtime_environment_sha256": environment_digest,
        "worktree_clean": True,
    }


def _verify_final_ppo(
    root: Path,
    manifest: Mapping[str, object],
    binding: ResidualD1BStoreBinding,
) -> str:
    checkpoint = _mapping(manifest.get("checkpoint"), "D1b final checkpoint")
    path = root / "residual_ranker_ppo.pt"
    backbone, metadata = load_recurrent_checkpoint(
        path,
        binding.device,
        expected_observation_dim=binding.observation_dim,
        expected_action_dim=binding.action_dim,
        expected_hidden_dim=binding.hidden_dim,
        expected_actor_mode="action_conditioned",
    )
    validate_source_only_payload(metadata, "D1b final checkpoint metadata")
    if (
        checkpoint.get("name") != path.name
        or checkpoint.get("sha256") != sha256_file(path)
        or metadata.get("d1a_manifest_digest") != binding.d1a_manifest_digest
        or metadata.get("source_roles_digest") != binding.source_roles_digest
    ):
        raise ValueError("D1b final checkpoint provenance is invalid")
    policy = ResidualRankerPolicy(
        backbone,
        confidence_threshold=metadata.get("confidence_threshold"),  # type: ignore[arg-type]
        prior_temperature=metadata.get("prior_temperature"),  # type: ignore[arg-type]
        overrides_enabled=metadata.get("overrides_enabled"),  # type: ignore[arg-type]
    )
    digest = policy.persistent_digest()
    if (
        metadata.get("policy_digest") != digest
        or checkpoint.get("persistent_digest") != digest
    ):
        raise ValueError("D1b final policy digest changed")
    return digest


def _verify_residual_d1_study_locked(
    study_root: Path,
    source_manifest: Path,
    source_root: Path,
    data_root: Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("deep D1 verification must run on the RTX CUDA host")
    root = Path(study_root).resolve()
    _assert_no_symlinks(root)
    _validate_root(root)
    study = load_verified_json(root / "study_manifest.json")
    validate_source_only_payload(study, "D1 study manifest")
    if study.get("status") != "complete":
        raise ValueError("deep D1 verification requires a complete study")
    request = _request_from_study(
        root,
        Path(source_manifest).resolve(),
        Path(source_root).resolve(),
        Path(data_root).resolve(),
        study,
    )
    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError("D1 verification requires torchvision") from error
    seed_everything(request.seed)
    runtime_environment = _runtime_environment(torchvision.__version__)
    provenance = _verify_current_provenance(
        study,
        request,
        runtime_environment,
    )
    _verify_complete_children(root, study)
    d1a = load_verified_json(root / "d1a" / "d1_manifest.json")
    _verify_complete_d1_children(request, d1a)
    if d1a.get("runtime_environment") != runtime_environment:
        raise ValueError("D1a runtime environment changed")
    train = torchvision.datasets.CIFAR10(
        root=str(request.data_root),
        train=True,
        transform=ToTensor(),
        download=False,
    )
    test = torchvision.datasets.CIFAR10(
        root=str(request.data_root),
        train=False,
        transform=ToTensor(),
        download=False,
    )
    context = load_d1_source_context(
        request,
        train,
        test,
        dataset_content_sha256=str(provenance["dataset_content_sha256"]),
    )
    cache_binding, cache_protocol = _cache_binding(request, context)
    cache = load_residual_teacher_cache(
        request.output_dir,
        expected_binding=cache_binding,
        expected_protocol=cache_protocol,
        action_dim=context.config.attack_config().action_dim,
        observation_dim=context.config.attack_config().recurrent_observation_dim,
    )
    roles = build_d1b_source_roles(context, cache)
    bundle = ResidualD1AArtifactBundle(
        request=request,
        dataset_content_sha256=str(provenance["dataset_content_sha256"]),
        observation_dim=context.config.attack_config().recurrent_observation_dim,
        action_dim=context.config.attack_config().action_dim,
        hidden_dim=D1_HIDDEN_DIM,
        device=request.device,
    )
    verified_d1a = verify_d1a_artifacts(d1a, bundle)
    d1b_root = root / "d1b"
    d1b = load_verified_json(d1b_root / "d1b_manifest.json")
    validate_source_only_payload(d1b, "D1b manifest")
    if d1b.get("runtime_environment") != runtime_environment:
        raise ValueError("D1b runtime environment changed")
    d1a_passed = _mapping(d1a.get("d1_decision"), "D1a decision").get("passed")
    blocks = 0
    final_ppo_digest: str | None = None
    if d1b.get("status") == "complete":
        if d1a_passed is not True:
            raise ValueError("completed D1b requires a passing D1a gate")
        _validate_output(d1b_root)
        verify_complete_d1b_children(d1b_root, d1b)
        store_binding = ResidualD1BStoreBinding(
            root=d1b_root,
            device=request.device,
            observation_dim=bundle.observation_dim,
            action_dim=bundle.action_dim,
            hidden_dim=bundle.hidden_dim,
            d1a_manifest_digest=verified_d1a.manifest_digest,
            d1a_checkpoint_sha256=verified_d1a.checkpoint_sha256,
            bc_policy_digest=verified_d1a.bc_policy_digest,
            source_roles_digest=roles.digest,
        )
        resume = ResidualD1BBlockStore(store_binding).load_resume_state()
        if (
            resume is None
            or resume.completed_episodes != 200
            or len(resume.blocks) != 4
        ):
            raise ValueError("D1b checkpoint chain is incomplete")
        blocks = len(resume.blocks)
        final_ppo_digest = _verify_final_ppo(
            d1b_root,
            d1b,
            store_binding,
        )
    elif d1b.get("status") == "skipped":
        if d1a_passed is not False:
            raise ValueError("D1b can be skipped only after a negative D1a gate")
    else:
        raise ValueError("complete D1 study has invalid D1b status")
    artifact_files = _study_file_digests(root)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "verified",
        "study_manifest_sha256": sha256_file(root / "study_manifest.json"),
        "study_outcome": study.get("study_outcome"),
        "d1a_gate_passed": d1a_passed,
        "d1b_status": d1b.get("status"),
        "d1b_verified_blocks": blocks,
        "bc_policy_digest": verified_d1a.bc_policy_digest,
        "ppo_policy_digest": final_ppo_digest,
        "source_roles_digest": roles.digest,
        "target_calls": 0,
        "hidden_target_calls": 0,
        "artifact_file_count": len(artifact_files),
        "artifact_files": artifact_files,
        "artifact_tree_sha256": canonical_json_digest(artifact_files),
        **provenance,
    }
    return {
        **result,
        "verification_sha256": canonical_json_digest(result),
    }


def _resolved_existing_path(value: Path, label: str, *, directory: bool) -> Path:
    raw = Path(value)
    if raw.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    resolved = raw.resolve()
    predicate = resolved.is_dir if directory else resolved.is_file
    if not predicate():
        raise ValueError(f"{label} is missing")
    return resolved


def verify_residual_d1_study(
    study_root: Path,
    source_manifest: Path,
    source_root: Path,
    data_root: Path,
) -> dict[str, object]:
    """Deeply verify a complete study under its exclusive artifact lock."""

    root = _resolved_existing_path(study_root, "D1 study root", directory=True)
    manifest = _resolved_existing_path(
        source_manifest,
        "D1 source manifest",
        directory=False,
    )
    source = _resolved_existing_path(
        source_root,
        "D1 source root",
        directory=True,
    )
    data = _resolved_existing_path(data_root, "D1 data root", directory=True)
    lock = root / ".study.lock"
    if lock.is_symlink():
        raise ValueError("D1 study lock cannot be a symlink")
    with exclusive_file_lock(lock):
        return _verify_residual_d1_study_locked(
            root,
            manifest,
            source,
            data,
        )


def verify_and_package_residual_d1_study(
    study_root: Path,
    source_manifest: Path,
    source_root: Path,
    data_root: Path,
    *,
    archive_path: Path,
    checksums_path: Path,
) -> dict[str, object]:
    """Verify and package exactly one immutable study snapshot."""

    root = _resolved_existing_path(study_root, "D1 study root", directory=True)
    manifest = _resolved_existing_path(
        source_manifest,
        "D1 source manifest",
        directory=False,
    )
    source = _resolved_existing_path(
        source_root,
        "D1 source root",
        directory=True,
    )
    data = _resolved_existing_path(data_root, "D1 data root", directory=True)
    lock = root / ".study.lock"
    if lock.is_symlink():
        raise ValueError("D1 study lock cannot be a symlink")
    with exclusive_file_lock(lock):
        verified = _verify_residual_d1_study_locked(
            root,
            manifest,
            source,
            data,
        )
        artifact_files = _mapping(
            verified.get("artifact_files"),
            "D1 verified artifact inventory",
        )
        package = _write_locked_package(
            root,
            archive_path,
            checksums_path,
            {str(key): str(value) for key, value in artifact_files.items()},
        )
    result = {**verified, "package": package}
    return {
        **result,
        "verification_sha256": canonical_json_digest(
            {
                key: value
                for key, value in result.items()
                if key != "verification_sha256"
            }
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deeply verify a completed source-only D1 study"
    )
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--checksums", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if (arguments.archive is None) != (arguments.checksums is None):
        raise ValueError("--archive and --checksums must be supplied together")
    result = (
        verify_and_package_residual_d1_study(
            arguments.study_root,
            arguments.source_manifest,
            arguments.source_root,
            arguments.data_root,
            archive_path=arguments.archive,
            checksums_path=arguments.checksums,
        )
        if arguments.archive is not None and arguments.checksums is not None
        else verify_residual_d1_study(
            arguments.study_root,
            arguments.source_manifest,
            arguments.source_root,
            arguments.data_root,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
