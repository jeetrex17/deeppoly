from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Callable

import torch
from torch.utils.data import Dataset

from .artifacts import (
    exclusive_file_lock,
    load_model_checkpoint,
    save_model_checkpoint,
    sha256_file,
)
from .cifar_config import MacPilotConfig
from .cifar_data import (
    build_cifar_split,
    dataset_samples as _dataset_samples,
    disjoint_balanced_subsets,
    indices_digest,
)
from .cifar_execution import (
    CIFAR_VICTIM_FAMILIES,
    portable_checkpoint_records,
    portable_descendant,
    preflight_cache_only_victims,
    validate_victim_population,
)
from .cifar_manifest import (
    code_digest as _code_digest,
    git_revision as _git_revision,
    git_worktree_state as _git_worktree_state,
    write_json as _write_json,
)
from .cifar_models import build_cifar_victim_population
from .cifar_policy_training import train_policy_bundle
from .cifar_source_evaluation import source_evidence
from .cifar_target_evaluation import target_evidence
from .cifar_training import (
    classifier_accuracy as _classifier_accuracy,
    train_classifier as _train_classifier,
)
from .cifar_victim_cache import (
    victim_cache_contract as _victim_cache_contract,
    victim_cache_digest as _victim_cache_digest,
    victim_code_digest as _victim_code_digest,
)
from .models import freeze_model
from .paths import resolve_descendant
from .phase2_policy import FrozenTemperaturePolicy
from .reproducibility import seed_everything
from .runtime import resolve_device
from .source_gates import SourceGateThresholds
from .victim_provenance import victim_bank_digest

def _checkpoint_matches(metadata: dict[str, object], fingerprint: str) -> None:
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("checkpoint fingerprint does not match this pilot run")


def run_cifar_pilot_from_datasets(
    config: MacPilotConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    resume: bool = True,
    dataset_version: str = "in-memory-fixture",
    progress: Callable[[str], None] | None = None,
    evaluate_target: bool = True,
    source_victims_only: bool = False,
    victim_cache_only: bool = False,
    victim_cache_dataset_version: str | None = None,
    portable_paths: bool = False,
) -> dict[str, object]:
    for label, value in (
        ("evaluate_target", evaluate_target),
        ("source_victims_only", source_victims_only),
        ("victim_cache_only", victim_cache_only),
        ("portable_paths", portable_paths),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{label} must be a boolean")
    if source_victims_only and evaluate_target:
        raise ValueError(
            "source_victims_only requires evaluate_target=False"
        )
    if victim_cache_only and not resume:
        raise ValueError("victim_cache_only requires resume=True")
    if victim_cache_dataset_version is not None and (
        not isinstance(victim_cache_dataset_version, str)
        or not victim_cache_dataset_version
    ):
        raise TypeError(
            "victim_cache_dataset_version must be a non-empty string"
        )
    if (
        victim_cache_dataset_version is not None
        and not victim_cache_only
    ):
        raise ValueError(
            "victim_cache_dataset_version is cache-only metadata"
        )
    cache_dataset_version = (
        victim_cache_dataset_version
        if victim_cache_dataset_version is not None
        else dataset_version
    )
    report = progress or (lambda _message: None)
    started = time.monotonic()
    seed_everything(config.seed)
    selection = resolve_device(config.device)
    device = selection.device
    report(f"resolved device: {device.type}")
    split_seed = config.split_seed if config.split_seed is not None else config.seed
    victim_seed = config.victim_seed if config.victim_seed is not None else config.seed
    split = build_cifar_split(
        train_dataset.targets,
        test_dataset.targets,
        config.victim_train_images,
        config.policy_train_images,
        config.source_validation_images,
        config.outer_test_images,
        split_seed,
    )
    if (
        config.victim_validation_images > 0
        and config.behavior_cloning_validation_episodes > 0
        and config.source_evaluation_images > 0
    ):
        (
            victim_validation_indices,
            bc_validation_indices,
            source_gate_indices,
        ) = disjoint_balanced_subsets(
            train_dataset,
            split.source_validation,
            (
                config.victim_validation_images,
                config.behavior_cloning_validation_episodes,
                config.source_evaluation_images,
            ),
        )
    else:
        victim_validation_indices = split.source_validation
        bc_validation_indices = split.source_validation
        source_gate_indices = (
            disjoint_balanced_subsets(
                train_dataset,
                split.source_validation,
                (config.source_evaluation_images,),
            )[0]
            if config.source_evaluation_images > 0
            else ()
        )
    role_digests = {
        "victim_fit": indices_digest(split.victim_fit),
        "policy_train": indices_digest(split.policy_train),
        "victim_validation": indices_digest(victim_validation_indices),
        "bc_validation": indices_digest(bc_validation_indices),
        "source_gate": indices_digest(source_gate_indices),
        "outer_test": indices_digest(split.outer_test),
    }
    code_digest = _code_digest()
    victim_code_digest = _victim_code_digest()
    victim_cache_contract = _victim_cache_contract(
        config,
        split.digest,
        cache_dataset_version,
        victim_code_digest,
        device.type,
    )
    victim_cache_digest = _victim_cache_digest(
        config,
        split.digest,
        cache_dataset_version,
        victim_code_digest,
        device.type,
    )
    fingerprint_source = (
        f"{config.digest()}:{split.digest}:{code_digest}:"
        f"{dataset_version}:{victim_cache_digest}"
    )
    fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
    output_root = Path(config.output_dir).resolve()
    run_dir = resolve_descendant(
        output_root,
        fingerprint[:12],
        label="pilot run directory",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_descendant(
        run_dir,
        "manifest.json",
        label="pilot manifest",
    )
    report(f"run directory: {run_dir}")
    manifest_run_dir = (
        portable_descendant(
            output_root,
            run_dir,
            label="portable pilot run directory",
        )
        if portable_paths
        else str(run_dir)
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "name": config.name,
        "status": "running",
        "research_valid": False,
        "fingerprint": fingerprint,
        "run_dir": manifest_run_dir,
        "config": asdict(config),
        "config_digest": config.digest(),
        "split_digest": split.digest,
        "data_role_digests": role_digests,
        "validation_roles_disjoint": not (
            set(victim_validation_indices)
            & set(bc_validation_indices)
            or set(victim_validation_indices)
            & set(source_gate_indices)
            or set(bc_validation_indices)
            & set(source_gate_indices)
        ),
        "seed": config.seed,
        "split_seed": split_seed,
        "victim_seed": victim_seed,
        "target_family": config.target_family,
        "source_families": [
            family
            for family in CIFAR_VICTIM_FAMILIES
            if family != config.target_family
        ],
        "execution_mode": {
            "evaluate_target": evaluate_target,
            "source_victims_only": source_victims_only,
            "victim_cache_only": victim_cache_only,
            "portable_paths": portable_paths,
        },
        "dataset": {"name": config.dataset, "version": dataset_version},
        "device": selection.as_dict(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "git_revision": _git_revision(),
            "code_digest": code_digest,
            "git_worktree": _git_worktree_state(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "cuda_total_memory_bytes": (
                torch.cuda.get_device_properties(device).total_memory
                if device.type == "cuda"
                else None
            ),
            "determinism": (
                "warn_only; some MPS operators may not have deterministic implementations"
                if device.type == "mps"
                else "deterministic algorithms requested with warn_only"
            ),
        },
        "victim_cache_digest": victim_cache_digest,
        "victim_cache_contract": victim_cache_contract,
        "victim_cache_dataset_version": cache_dataset_version,
        "victim_code_digest": victim_code_digest,
    }
    _write_json(manifest_path, manifest)
    selected_families = (
        tuple(
            family
            for family in CIFAR_VICTIM_FAMILIES
            if family != config.target_family
        )
        if source_victims_only
        else CIFAR_VICTIM_FAMILIES
    )
    instance_counts = {
        family: (
            config.target_instances_per_family
            if family == config.target_family
            else (
                config.source_instances_per_family
                + config.source_holdout_instances_per_family
            )
        )
        for family in selected_families
    }
    victim_population = build_cifar_victim_population(
        victim_seed,
        instance_counts,
        families=selected_families,
        profile=config.victim_profile,
    )
    victim_ids = validate_victim_population(
        victim_population,
        instance_counts,
    )
    victim_metrics: dict[str, object] = {}
    victim_instance_metrics: dict[str, list[dict[str, object]]] = {}
    victim_cache_root = resolve_descendant(
        output_root,
        "victim_cache",
        label="victim cache root",
    )
    victim_cache_dir = resolve_descendant(
        victim_cache_root,
        victim_cache_digest[:12],
        label="victim cache directory",
    )
    if victim_cache_only:
        preflight_cache_only_victims(
            victim_cache_dir,
            victim_ids,
        )
    for family, instances in victim_population.items():
        family_metrics: list[dict[str, object]] = []
        for instance_index, (victim_id, model) in enumerate(instances):
            model.to(device)
            training_seed = int.from_bytes(
                hashlib.sha256(f"victim-fit-v1:{victim_id}".encode()).digest()[:8],
                "big",
            ) % (2**63 - 1)
            checkpoint_path = resolve_descendant(
                victim_cache_dir,
                f"{victim_id}.pt",
                label="victim checkpoint",
            )
            checksum_path = resolve_descendant(
                victim_cache_dir,
                f"{victim_id}.pt.sha256",
                label="victim checkpoint checksum",
            )
            lock_path = resolve_descendant(
                victim_cache_dir,
                f"{victim_id}.pt.lock",
                label="victim checkpoint lock",
            )
            with exclusive_file_lock(lock_path):
                if resume and checkpoint_path.is_file() and checksum_path.is_file():
                    report(f"loading {family} victim instance {instance_index} checkpoint")
                    metadata = load_model_checkpoint(checkpoint_path, model, device)
                    _checkpoint_matches(metadata, victim_cache_digest)
                    if metadata.get("training_seed") != training_seed:
                        raise ValueError("victim checkpoint training seed mismatch")
                    if metadata.get("cache_contract") != victim_cache_contract:
                        raise ValueError("victim checkpoint cache contract mismatch")
                    history = metadata["history"]
                    fit_elapsed_seconds = float(
                        metadata.get("fit_elapsed_seconds", 0.0)
                    )
                    resumed = True
                elif victim_cache_only:
                    raise ValueError(
                        "cache-only victim checkpoint became unavailable "
                        f"after preflight: {victim_id}"
                    )
                else:
                    report(f"training {family} victim instance {instance_index} ({victim_id})")
                    fit_started = time.monotonic()
                    history = _train_classifier(
                        model,
                        train_dataset,
                        split.victim_fit,
                        config,
                        device,
                        training_seed,
                        report,
                    )
                    fit_elapsed_seconds = (
                        time.monotonic() - fit_started
                    )
                    metadata = {
                        "fingerprint": victim_cache_digest,
                        "cache_contract": victim_cache_contract,
                        "family": family,
                        "instance_index": instance_index,
                        "training_seed": training_seed,
                        "history": history,
                        "fit_elapsed_seconds": fit_elapsed_seconds,
                    }
                    save_model_checkpoint(checkpoint_path, model, metadata)
                    resumed = False
            validation_accuracy = _classifier_accuracy(
                model,
                train_dataset,
                victim_validation_indices,
                config.batch_size,
                config.num_workers,
                device,
            )
            freeze_model(model)
            report(
                f"{family} instance {instance_index} validation accuracy: "
                f"{validation_accuracy:.3f}"
            )
            metrics = {
                "victim_id": victim_id,
                "family": family,
                "instance_index": instance_index,
                "training_seed": training_seed,
                "checkpoint": (
                    portable_descendant(
                        output_root,
                        checkpoint_path,
                        label="portable victim checkpoint",
                    )
                    if portable_paths
                    else str(checkpoint_path)
                ),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "history": history,
                "fit_elapsed_seconds": fit_elapsed_seconds,
                "source_validation_accuracy": validation_accuracy,
                "resumed": resumed,
            }
            family_metrics.append(metrics)
        victim_instance_metrics[family] = family_metrics
        victim_metrics[family] = family_metrics[0]
    attack = config.attack_config()
    source_victims = {
        family: instances[: config.source_instances_per_family]
        for family, instances in victim_population.items()
        if family != config.target_family
    }
    source_holdout_victims = {
        family: instances[
            config.source_instances_per_family :
            config.source_instances_per_family
            + config.source_holdout_instances_per_family
        ]
        for family, instances in victim_population.items()
        if family != config.target_family
    }
    bank_digest = victim_bank_digest(victim_instance_metrics)
    source_victim_checkpoints = {
        metrics["victim_id"]: metrics["checkpoint_sha256"]
        for family in source_victims
        for metrics in victim_instance_metrics[family]
    }
    policy_binding = {
        "run_fingerprint": fingerprint,
        "dataset_version": dataset_version,
        "victim_cache_digest": victim_cache_digest,
        "source_victim_checkpoints": source_victim_checkpoints,
    }
    policy_fingerprint = hashlib.sha256(
        json.dumps(
            policy_binding,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    policy_samples = _dataset_samples(train_dataset, split.policy_train)
    policy_bundle = train_policy_bundle(
        config=config,
        attack=attack,
        source_victims=source_victims,
        policy_samples=policy_samples,
        bc_validation_samples=_dataset_samples(
            train_dataset,
            bc_validation_indices,
        ),
        run_dir=run_dir,
        fingerprint=policy_fingerprint,
        split_digest=split.digest,
        device=device,
        resume=resume,
        report=report,
    )
    policy = policy_bundle.main
    evaluation_policy = FrozenTemperaturePolicy(
        policy,
        config.policy_evaluation_temperature,
    )
    training = policy_bundle.training
    manifest_policy_checkpoints = (
        portable_checkpoint_records(
            policy_bundle.checkpoints,
            run_dir=run_dir,
        )
        if portable_paths
        else policy_bundle.checkpoints
    )
    policy_resumed = policy_bundle.main_resumed
    behavior_cloning = training["behavior_cloning"]
    policy_path = Path(policy_bundle.checkpoints["main"]["path"])
    ablation_policies = (
        {
            "gradient_bc_only_stochastic": (
                policy_bundle.bc_only,
                False,
            ),
            "ppo_only_stochastic": (
                policy_bundle.ppo_only,
                False,
            ),
        }
        if (
            policy_bundle.bc_only is not None
            and policy_bundle.ppo_only is not None
        )
        else {}
    )
    evaluation_ablation_policies = {
        method: (
            FrozenTemperaturePolicy(
                ablation_policy,
                config.policy_evaluation_temperature,
            ),
            deterministic,
        )
        for method, (
            ablation_policy,
            deterministic,
        ) in ablation_policies.items()
    }
    raw_main_method_prefix = training.get("method_id")
    if (
        not isinstance(raw_main_method_prefix, str)
        or not raw_main_method_prefix
    ):
        raise RuntimeError("policy training did not provide a method ID")
    main_method_prefix = raw_main_method_prefix
    report(f"trained policy episodes: {training['trained_episodes']}")

    all_accuracy_thresholds = {
        "classical_cnn": config.classical_cnn_min_accuracy,
        "modern_cnn": config.modern_cnn_min_accuracy,
        "transformer": config.transformer_min_accuracy,
    }
    accuracy_thresholds = {
        family: all_accuracy_thresholds[family]
        for family in victim_instance_metrics
    }
    victim_accuracy_gate = {
        "thresholds": accuracy_thresholds,
        "passed": all(
            all(
                float(metrics["source_validation_accuracy"]) >= threshold
                for metrics in victim_instance_metrics[family]
            )
            for family, threshold in accuracy_thresholds.items()
        ),
    }
    model_instances_by_family = {
        family: len(victim_population.get(family, ()))
        for family in CIFAR_VICTIM_FAMILIES
    }
    validation_evaluations_by_family = {
        family: len(victim_instance_metrics.get(family, ()))
        for family in CIFAR_VICTIM_FAMILIES
    }
    heldout_model_calls = model_instances_by_family[
        config.target_family
    ]
    heldout_validation_calls = validation_evaluations_by_family[
        config.target_family
    ]
    victim_access_audit = {
        "source_victims_only": source_victims_only,
        "victim_cache_only": victim_cache_only,
        "constructed_families": list(victim_population),
        "untouched_families": (
            [config.target_family] if source_victims_only else []
        ),
        "model_instances_by_family": model_instances_by_family,
        "validation_evaluations_by_family": (
            validation_evaluations_by_family
        ),
        "heldout_family": config.target_family,
        "heldout_family_model_calls": heldout_model_calls,
        "heldout_family_validation_calls": heldout_validation_calls,
        "passed": (
            not source_victims_only
            or (
                heldout_model_calls == 0
                and heldout_validation_calls == 0
                and config.target_family not in victim_population
            )
        ),
    }
    if config.source_evaluation_images > 0:
        source_indices = source_gate_indices
        source_samples = _dataset_samples(train_dataset, source_indices)
        evidence = source_evidence(
            policy=evaluation_policy,
            additional_policies=evaluation_ablation_policies,
            source_victims=source_victims,
            source_holdout_victims=source_holdout_victims,
            samples=source_samples,
            indices=source_indices,
            attack=attack,
            seed=config.seed + 800_000,
            main_method_prefix=main_method_prefix,
            trace_samples_per_method=(
                config.query_trace_samples_per_method
            ),
            thresholds=SourceGateThresholds(
                minimum_asr_gain=config.minimum_source_asr_gain,
                minimum_auc_gain=config.minimum_source_auc_gain,
                entropy_min=config.source_entropy_min,
                entropy_max=config.source_entropy_max,
            ),
            run_dir=run_dir,
            binding={
                "config_digest": config.digest(),
                "code_digest": code_digest,
                "split_digest": split.digest,
                "data_role_digests": role_digests,
                "policy_checkpoints": manifest_policy_checkpoints,
                "source_victim_checkpoints": (
                    source_victim_checkpoints
                ),
            },
            resume=resume,
            report=report,
        )
        source_evaluation = evidence.evaluation
        source_competence_gate = evidence.gate
        source_evaluation_audits = evidence.audits
        source_cache_resumed = evidence.cache_resumed
        source_evaluation_elapsed_seconds = (
            evidence.evaluation_elapsed_seconds
        )
    else:
        source_evaluation = {}
        source_evaluation_audits = {}
        source_cache_resumed = False
        source_evaluation_elapsed_seconds = 0.0
        source_competence_gate = {
            "passed": False,
            "reason": "source evaluation disabled in this legacy configuration",
            "slices": {},
            "errors": ["source_evaluation_images is zero"],
        }

    shared_manifest = {
        "victims": victim_metrics,
        "victim_instances": victim_instance_metrics,
        "victim_bank_digest": bank_digest,
        "policy": {
            "checkpoint": (
                portable_descendant(
                    run_dir,
                    policy_path,
                    label="portable main policy checkpoint",
                )
                if portable_paths
                else str(policy_path)
            ),
            "checkpoint_sha256": sha256_file(policy_path),
            "persistent_digest": policy.persistent_digest(),
            "checkpoints": manifest_policy_checkpoints,
            "resumed": policy_resumed,
            "training": training,
            "training_fingerprint": policy_fingerprint,
            "training_binding": policy_binding,
        },
        "victim_accuracy_gate": victim_accuracy_gate,
        "victim_access_audit": victim_access_audit,
        "source_evaluation": source_evaluation,
        "source_evaluation_audits": source_evaluation_audits,
        "source_cache_resumed": source_cache_resumed,
        "source_evaluation_elapsed_seconds": (
            source_evaluation_elapsed_seconds
        ),
        "source_competence_gate": source_competence_gate,
    }
    if not evaluate_target:
        if _code_digest() != code_digest:
            raise RuntimeError("package code changed during source phase; discard and rerun")
        manifest.update(
            {
                **shared_manifest,
                "status": "source_complete",
                "target_evaluation_performed": False,
                "target_calls": 0,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        _write_json(manifest_path, manifest)
        return manifest

    if (
        config.source_evaluation_images > 0
        and (
            not victim_accuracy_gate["passed"]
            or not source_competence_gate["passed"]
            or not behavior_cloning["gate"]["passed"]
        )
    ):
        raise RuntimeError(
            "target evaluation is locked because a source-side gate failed"
        )
    target_binding = {
        "config_digest": config.digest(),
        "code_digest": code_digest,
        "split_digest": split.digest,
        "data_role_digests": role_digests,
        "policy_checkpoints": manifest_policy_checkpoints,
        "target_victim_checkpoints": {
            metrics["victim_id"]: metrics["checkpoint_sha256"]
            for metrics in victim_instance_metrics[
                config.target_family
            ]
        },
    }
    target_samples = _dataset_samples(test_dataset, split.outer_test)

    def target_accuracy() -> dict[str, float]:
        return {
            victim_id: _classifier_accuracy(
                victim,
                test_dataset,
                split.outer_test,
                config.batch_size,
                config.num_workers,
                device,
            )
            for victim_id, victim in victim_population[
                config.target_family
            ]
        }

    def verify_code_unchanged() -> None:
        if _code_digest() != code_digest:
            raise RuntimeError(
                "package code changed during the pilot; discard and rerun"
            )

    evidence = target_evidence(
        policy=evaluation_policy,
        additional_policies=evaluation_ablation_policies,
        target_victims=victim_population[config.target_family],
        samples=target_samples,
        indices=split.outer_test,
        attack=attack,
        seed=config.seed,
        target_family=config.target_family,
        trace_samples_per_method=config.query_trace_samples_per_method,
        main_method_prefix=main_method_prefix,
        run_dir=run_dir,
        binding=target_binding,
        resume=resume,
        report=report,
        accuracy_by_victim=target_accuracy,
        verify_code_unchanged=verify_code_unchanged,
    )
    manifest.update(
        {
            **shared_manifest,
            "status": "complete",
            "target_evaluation_performed": True,
            "target_cache_resumed": evidence.cache_resumed,
            "target_evaluation_elapsed_seconds": (
                evidence.evaluation_elapsed_seconds
            ),
            "evaluation": evidence.evaluation,
            "evaluation_audit": evidence.audit,
            "target_test_accuracy_by_victim": (
                evidence.accuracy_by_victim
            ),
            "target_test_accuracy": statistics.fmean(
                evidence.accuracy_by_victim.values()
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def run_cifar_pilot(
    config_path: Path,
    resume: bool = True,
    device: str | None = None,
    output_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    config = MacPilotConfig.from_json(config_path)
    if device is not None:
        config = replace(config, device=device)
    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))
    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError("install the vision extra before running the CIFAR pilot") from error
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=True,
        transform=ToTensor(),
        download=config.download,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=False,
        transform=ToTensor(),
        download=config.download,
    )
    return run_cifar_pilot_from_datasets(
        config,
        train_dataset,
        test_dataset,
        resume=resume,
        dataset_version=f"torchvision-{torchvision.__version__}",
        progress=progress,
    )
