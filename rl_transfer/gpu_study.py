"""Two-phase, source-gated RTX study orchestration."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from typing import Callable

from torch.utils.data import Dataset

from .cifar_pilot import (
    MacPilotConfig,
    _code_digest,
    run_cifar_pilot_from_datasets,
)
from .cifar_study import summarize_study
from .confirmatory_metrics import victim_macro_metrics
from .gpu_config import RTXPublicationConfig
from .gpu_environment import (
    capture_runtime_environment,
    require_clean_protocol_tree,
)
from .paths import resolve_descendant, resolve_within_repository
from .reproducibility import tree_digest
from .source_grid_gate import source_grid_gate as _source_grid_gate
from .statistics import (
    bootstrap_interval,
    exact_paired_sign_flip_pvalue,
)
from .verified_artifacts import (
    load_verified_json,
    write_verified_json as _write_json,
)


PRIMARY_METHOD = "gradient_bc_groupdro_ppo_stochastic"
BC_ONLY_METHOD = "gradient_bc_only_stochastic"
PPO_ONLY_METHOD = "ppo_only_stochastic"
def _validate_base_contract(
    study: RTXPublicationConfig,
    base: MacPilotConfig,
) -> None:
    errors: list[str] = []
    if base.device != "cuda":
        errors.append("base device must be cuda")
    if base.query_budget != 50:
        errors.append("primary query budget must be 50 including initialization")
    if abs(base.epsilon - 8 / 255) > 1e-12:
        errors.append("primary epsilon must be 8/255")
    if abs(base.step_size - 2 / 255) > 1e-12:
        errors.append("matched proposal step must be 2/255")
    if not base.rollback_on_non_improvement:
        errors.append("all comparable methods require rollback on non-improvement")
    if not base.action_history_features:
        errors.append("the recurrent state requires action-history features")
    if not base.image_patch_features:
        errors.append("the recurrent state requires patch image features")
    if base.behavior_cloning_teacher != "gradient":
        errors.append("the main candidate requires the locked gradient teacher")
    if base.behavior_cloning_episodes < 1 or base.behavior_cloning_epochs < 1:
        errors.append("the main candidate requires behavior-cloning warm start")
    if not base.train_ablation_policies:
        errors.append("BC-only and PPO-only ablations are required")
    if base.source_evaluation_images < 100:
        errors.append("source competence requires at least 100 balanced images")
    if base.victim_validation_images < 100:
        errors.append("victim quality requires a disjoint validation role")
    if (
        base.victim_validation_images
        + base.behavior_cloning_validation_episodes
        + base.source_evaluation_images
        > base.source_validation_images
    ):
        errors.append(
            "victim, BC, and source-gate validation roles must fit disjointly"
        )
    if base.behavior_cloning_validation_episodes % 10:
        errors.append("BC validation episodes must form a balanced CIFAR role")
    if (
        base.source_holdout_instances_per_family
        != study.source_holdout_instances_per_family
    ):
        errors.append("source holdout instance counts disagree")
    if base.target_instances_per_family != study.target_instances_per_family:
        errors.append("target instance counts disagree")
    if errors:
        raise ValueError("; ".join(errors))


def load_validated_study_config(
    config_path: Path,
) -> tuple[RTXPublicationConfig, MacPilotConfig, Path]:
    locked_path = resolve_within_repository(
        config_path,
        allowed_directory="configs/rl_transfer",
        label="publication config",
    )
    study = RTXPublicationConfig.from_json(locked_path)
    base = MacPilotConfig.from_json(
        resolve_within_repository(
            study.base_config,
            allowed_directory="configs/rl_transfer",
            label="base_config",
        )
    )
    _validate_base_contract(study, base)
    return study, base, locked_path


def _derived_config(
    study: RTXPublicationConfig,
    base: MacPilotConfig,
    target_family: str,
    seed: int,
    run_output_dir: Path,
) -> MacPilotConfig:
    return replace(
        base,
        name=f"{study.name}-{target_family}-seed-{seed}",
        seed=seed,
        target_family=target_family,
        output_dir=str(run_output_dir),
        device=study.device,
        split_seed=study.split_seed,
        victim_seed=study.victim_seed,
        source_holdout_instances_per_family=(
            study.source_holdout_instances_per_family
        ),
        target_instances_per_family=study.target_instances_per_family,
        minimum_source_asr_gain=study.minimum_source_asr_gain,
        minimum_source_auc_gain=study.minimum_source_auc_gain,
    )


def confirmatory_transfer_gate(
    runs: list[dict[str, object]],
    study: RTXPublicationConfig,
    *,
    expected_victim_bank_digest: str | None = None,
    expected_policy_checkpoints: dict[str, str] | None = None,
) -> dict[str, object]:
    """Apply the one authoritative target promotion rule.

    Families and target victims are repeated measurements within a policy
    seed. Exact inference is therefore performed on one macro difference per
    independently trained policy.
    """

    expected = {
        (family, seed)
        for family in study.target_families
        for seed in study.seeds
    }
    observed: set[tuple[str, int]] = set()
    errors: list[str] = []
    by_seed: dict[int, list[tuple[float, float, float, float]]] = {
        seed: [] for seed in study.seeds
    }
    family_differences: dict[str, list[float]] = {
        family: [] for family in study.target_families
    }
    victim_bank_digests: set[str] = set()
    policy_digests_by_family: dict[str, set[str]] = {
        family: set() for family in study.target_families
    }
    for run in runs:
        family = str(run.get("target_family"))
        seed = int(run.get("seed", -1))
        key = (family, seed)
        if key not in expected:
            errors.append(f"unexpected run {family}/seed-{seed}")
            continue
        if key in observed:
            errors.append(f"duplicate run {family}/seed-{seed}")
            continue
        observed.add(key)
        run_id = f"{family}/seed-{seed}"
        if run.get("status") != "complete":
            errors.append(f"{run_id}: target run is incomplete")
        if run.get("victim_seed") != study.victim_seed:
            errors.append(f"{run_id}: fixed victim seed mismatch")
        bank_digest = run.get("victim_bank_digest")
        if not isinstance(bank_digest, str) or len(bank_digest) != 64:
            errors.append(f"{run_id}: victim bank digest is invalid")
        else:
            victim_bank_digests.add(bank_digest)
            if (
                expected_victim_bank_digest is not None
                and bank_digest != expected_victim_bank_digest
            ):
                errors.append(f"{run_id}: victim bank changed after source gate")
        policy_block = run.get("policy")
        checkpoint_sha = (
            policy_block.get("checkpoint_sha256")
            if isinstance(policy_block, dict)
            else None
        )
        if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
            errors.append(f"{run_id}: policy checkpoint digest is invalid")
        elif (
            expected_policy_checkpoints is not None
            and expected_policy_checkpoints.get(run_id) != checkpoint_sha
        ):
            errors.append(f"{run_id}: policy checkpoint changed after source gate")
        victim_gate = run.get("victim_accuracy_gate")
        if (
            not isinstance(victim_gate, dict)
            or victim_gate.get("passed") is not True
        ):
            errors.append(f"{run_id}: victim gate failed")
        audit = run.get("evaluation_audit")
        if (
            not isinstance(audit, dict)
            or audit.get("passed") is not True
            or audit.get("expected_cohort_verified") is not True
        ):
            errors.append(f"{run_id}: raw evaluation audit failed")
        evaluation = run.get("evaluation")
        required = (
            PRIMARY_METHOD,
            study.primary_control,
            BC_ONLY_METHOD,
            PPO_ONLY_METHOD,
        )
        if (
            not isinstance(evaluation, dict)
            or any(method not in evaluation for method in required)
        ):
            errors.append(f"{run_id}: required methods are missing")
            continue
        try:
            learned = evaluation[PRIMARY_METHOD]
            control = evaluation[study.primary_control]
            bc_only = evaluation[BC_ONLY_METHOD]
            ppo_only = evaluation[PPO_ONLY_METHOD]
            selected = (learned, control, bc_only, ppo_only)
            if any(not isinstance(metrics, dict) for metrics in selected):
                raise ValueError("method metrics must be mappings")
            alignments = {
                (
                    metrics.get("eligible"),
                    metrics.get("eligible_sample_ids_sha256"),
                    metrics.get("query_budget"),
                )
                for metrics in selected
            }
            if len(alignments) != 1:
                raise ValueError("method cohorts or budgets are not aligned")
            macro_metrics = tuple(
                victim_macro_metrics(
                    metrics,
                    expected_victim_count=(
                        study.target_instances_per_family
                    ),
                )
                for metrics in selected
            )
            if len({metrics[2] for metrics in macro_metrics}) != 1:
                raise ValueError(
                    "per-victim eligible cohorts are not aligned"
                )
            if any(metrics.get("frozen") is not True for metrics in selected):
                raise ValueError("an evaluated policy was not frozen")
            matched_operator_digests = {
                evaluation[method].get("operator_digest")
                for method in (
                    PRIMARY_METHOD,
                    study.primary_control,
                    BC_ONLY_METHOD,
                    PPO_ONLY_METHOD,
                )
            }
            if len(matched_operator_digests) != 1:
                raise ValueError("primary and ablation operators differ")
            learned_asr, learned_auc, _ = macro_metrics[0]
            control_asr, control_auc, _ = macro_metrics[1]
            bc_asr, _, _ = macro_metrics[2]
            ppo_asr, _, _ = macro_metrics[3]
            learned_digest = learned.get("policy_digest_before")
            if not isinstance(learned_digest, str) or not learned_digest:
                raise ValueError("learned persistent digest is invalid")
            policy_digests_by_family[family].add(learned_digest)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{run_id}: {error}")
            continue
        asr_difference = learned_asr - control_asr
        family_differences[family].append(asr_difference)
        by_seed[seed].append(
            (
                asr_difference,
                learned_auc - control_auc,
                learned_asr - bc_asr,
                learned_asr - ppo_asr,
            )
        )
    missing = sorted(expected - observed)
    errors.extend(
        f"missing run {family}/seed-{seed}"
        for family, seed in missing
    )
    if len(victim_bank_digests) != 1:
        errors.append("target runs do not share one fixed victim bank")
    for family, digests in policy_digests_by_family.items():
        if len(digests) != len(study.seeds):
            errors.append(
                f"{family}: policy seeds did not produce distinct policies"
            )
    seed_summaries: dict[int, tuple[float, float, float, float]] = {}
    for seed, values in by_seed.items():
        if len(values) != len(study.target_families):
            errors.append(
                f"seed-{seed}: expected one result per target family"
            )
            continue
        seed_summaries[seed] = tuple(
            sum(cell[index] for cell in values) / len(values)
            for index in range(4)
        )
    asr_differences = tuple(
        seed_summaries[seed][0]
        for seed in study.seeds
        if seed in seed_summaries
    )
    auc_differences = tuple(
        seed_summaries[seed][1]
        for seed in study.seeds
        if seed in seed_summaries
    )
    bc_differences = tuple(
        seed_summaries[seed][2]
        for seed in study.seeds
        if seed in seed_summaries
    )
    ppo_differences = tuple(
        seed_summaries[seed][3]
        for seed in study.seeds
        if seed in seed_summaries
    )
    complete = (
        not errors
        and observed == expected
        and len(asr_differences) == len(study.seeds)
    )
    if complete:
        asr_interval = bootstrap_interval(
            asr_differences,
            samples=study.bootstrap_samples,
            seed=study.split_seed + 1,
        )
        auc_interval = bootstrap_interval(
            auc_differences,
            samples=study.bootstrap_samples,
            seed=study.split_seed + 2,
        )
        pvalue = exact_paired_sign_flip_pvalue(asr_differences)
        asr_mean = sum(asr_differences) / len(asr_differences)
        auc_mean = sum(auc_differences) / len(auc_differences)
        family_means = {
            family: sum(values) / len(values)
            for family, values in family_differences.items()
        }
        ablation_checks = {
            "hybrid_minus_bc_only_mean": (
                sum(bc_differences) / len(bc_differences)
            ),
            "hybrid_minus_ppo_only_mean": (
                sum(ppo_differences) / len(ppo_differences)
            ),
        }
    else:
        asr_interval = None
        auc_interval = None
        pvalue = None
        asr_mean = None
        auc_mean = None
        family_means = {}
        ablation_checks = {}
    passed = bool(
        complete
        and asr_mean is not None
        and auc_mean is not None
        and asr_interval is not None
        and auc_interval is not None
        and pvalue is not None
        and asr_mean >= study.minimum_target_asr_gain
        and asr_interval[0] > 0
        and pvalue <= 0.05
        and auc_mean >= study.minimum_target_auc_gain
        and auc_interval[0] > 0
        and all(value > 0 for value in family_means.values())
        and all(value > 0 for value in ablation_checks.values())
    )
    return {
        "passed": passed,
        "authoritative": True,
        "replicate_unit": study.replicate_unit,
        "victim_bank_digest": (
            next(iter(victim_bank_digests))
            if len(victim_bank_digests) == 1
            else None
        ),
        "grid_complete": complete,
        "errors": errors,
        "policy_seed_differences": {
            str(seed): {
                "macro_asr_delta": values[0],
                "macro_auc_delta": values[1],
                "hybrid_minus_bc_only_asr": values[2],
                "hybrid_minus_ppo_only_asr": values[3],
            }
            for seed, values in seed_summaries.items()
        },
        "primary": {
            "method": PRIMARY_METHOD,
            "control": study.primary_control,
            "endpoint": study.primary_metric,
            "mean_difference": asr_mean,
            "bootstrap_ci95": (
                list(asr_interval)
                if asr_interval is not None
                else None
            ),
            "exact_sign_flip_pvalue": pvalue,
            "minimum_practical_gain": study.minimum_target_asr_gain,
        },
        "secondary_query_efficiency": {
            "mean_auc_difference": auc_mean,
            "bootstrap_ci95": (
                list(auc_interval)
                if auc_interval is not None
                else None
            ),
            "minimum_practical_gain": study.minimum_target_auc_gain,
        },
        "family_macro_asr_differences": family_means,
        "component_ablation_differences": ablation_checks,
        "requirements": [
            "one macro observation per independently trained policy seed",
            "fixed victim bank crossed with every policy seed",
            "positive ASR effect in every held-out family",
            "positive policy-seed bootstrap intervals",
            "one exact two-sided primary sign-flip test at alpha 0.05",
            "all raw perturbation, query, cohort, operator, and frozen-policy audits pass",
            "BC-only and PPO-only checkpoints are evaluated",
            "the hybrid has positive mean ASR gains over both component ablations",
        ],
    }


def run_gpu_study_from_datasets(
    config: RTXPublicationConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_version: str,
    phase: str = "all",
    progress: Callable[[str], None] | None = None,
    runtime_environment: dict[str, object] | None = None,
) -> dict[str, object]:
    if phase not in {"source", "all"}:
        raise ValueError("phase must be 'source' or 'all'")
    report = progress or (lambda _message: None)
    started = time.monotonic()
    base = MacPilotConfig.from_json(
        resolve_within_repository(
            config.base_config,
            allowed_directory="configs/rl_transfer",
            label="base_config",
        )
    )
    _validate_base_contract(config, base)
    code_digest = _code_digest()
    config_record = json.loads(json.dumps(asdict(config)))
    output_root = resolve_within_repository(
        config.output_dir,
        allowed_directory="output/rl_transfer",
        label="output_dir",
    )
    study_dir = resolve_within_repository(
        output_root / config.name,
        allowed_directory="output/rl_transfer",
        label="study directory",
    )
    run_output_dir = resolve_within_repository(
        study_dir / "runs",
        allowed_directory="output/rl_transfer",
        label="study run directory",
    )
    study_manifest_path = resolve_descendant(
        study_dir,
        "study_manifest.json",
        label="study manifest",
    )
    prior_source_elapsed = 0.0
    existing_manifest_path = study_manifest_path
    if existing_manifest_path.is_file():
        try:
            existing_manifest = load_verified_json(
                existing_manifest_path
            )
            if (
                existing_manifest.get("study_code_digest")
                == code_digest
                and existing_manifest.get("config")
                == config_record
            ):
                prior_source_elapsed = float(
                    existing_manifest.get(
                        "source_phase_elapsed_seconds",
                        0.0,
                    )
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            prior_source_elapsed = 0.0
    source_runs: list[dict[str, object]] = []
    for target_family in config.target_families:
        for seed in config.seeds:
            if _code_digest() != code_digest:
                raise RuntimeError("package code changed during the source phase")
            report(f"source phase target={target_family} seed={seed}")
            derived = _derived_config(
                config,
                base,
                target_family,
                seed,
                run_output_dir,
            )
            run = run_cifar_pilot_from_datasets(
                derived,
                train_dataset,
                test_dataset,
                resume=config.resume,
                dataset_version=dataset_version,
                progress=lambda message, family=target_family, run_seed=seed: report(
                    f"[{family}/seed-{run_seed}] {message}"
                ),
                evaluate_target=False,
            )
            source_runs.append(run)
            _write_json(
                study_manifest_path,
                {
                    "schema_version": 1,
                    "name": config.name,
                    "status": "source_running",
                    "research_valid": False,
                    "publication_candidate": False,
                    "study_code_digest": code_digest,
                    "config": config_record,
                    "runtime_environment": runtime_environment or {},
                    "source_runs": source_runs,
                },
            )
    source_gate = _source_grid_gate(source_runs, config)
    current_source_elapsed = time.monotonic() - started
    recorded_source_elapsed = (
        prior_source_elapsed
        if prior_source_elapsed > 0
        else current_source_elapsed
    )
    if phase == "source" or not source_gate["passed"]:
        status = (
            "source_complete"
            if source_gate["passed"]
            else "source_learning_failed"
        )
        manifest = {
            "schema_version": 1,
            "name": config.name,
            "status": status,
            "research_valid": False,
            "publication_candidate": False,
            "study_code_digest": code_digest,
            "config": config_record,
            "runtime_environment": runtime_environment or {},
            "source_runs": source_runs,
            "source_competence_gate": source_gate,
            "target_evaluation_performed": False,
            "target_calls": 0,
            "elapsed_seconds": time.monotonic() - started,
            "source_phase_elapsed_seconds": recorded_source_elapsed,
        }
        _write_json(study_manifest_path, manifest)
        return manifest

    target_started = time.monotonic()
    target_runs: list[dict[str, object]] = []
    for target_family in config.target_families:
        for seed in config.seeds:
            if _code_digest() != code_digest:
                raise RuntimeError("package code changed during the target phase")
            report(f"locked target phase target={target_family} seed={seed}")
            derived = _derived_config(
                config,
                base,
                target_family,
                seed,
                run_output_dir,
            )
            run = run_cifar_pilot_from_datasets(
                derived,
                train_dataset,
                test_dataset,
                resume=config.resume,
                dataset_version=dataset_version,
                progress=lambda message, family=target_family, run_seed=seed: report(
                    f"[{family}/seed-{run_seed}] {message}"
                ),
                evaluate_target=True,
            )
            target_runs.append(run)
            _write_json(
                study_manifest_path,
                {
                    "schema_version": 1,
                    "name": config.name,
                    "status": "target_running",
                    "research_valid": False,
                    "publication_candidate": False,
                    "study_code_digest": code_digest,
                    "config": config_record,
                    "runtime_environment": runtime_environment or {},
                    "source_runs": source_runs,
                    "source_competence_gate": source_gate,
                    "runs": target_runs,
                    "source_phase_elapsed_seconds": recorded_source_elapsed,
                    "target_phase_elapsed_seconds": (
                        time.monotonic() - target_started
                    ),
                },
            )
    target_summary = summarize_study(
        target_runs,
        expected_families=config.target_families,
        expected_seeds=config.seeds,
        minimum_seeds=config.minimum_seeds,
        minimum_asr_gain=config.minimum_target_asr_gain,
        minimum_auc_gain=config.minimum_target_auc_gain,
    )
    confirmatory_gate = confirmatory_transfer_gate(
        target_runs,
        config,
        expected_victim_bank_digest=source_gate.get(
            "victim_bank_digest"
        ),
        expected_policy_checkpoints=source_gate.get(
            "policy_checkpoints"
        ),
    )
    publication_candidate = bool(
        source_gate["passed"]
        and confirmatory_gate["passed"]
    )
    research_valid = bool(
        source_gate["passed"]
        and confirmatory_gate["grid_complete"]
    )
    manifest = {
        "schema_version": 1,
        "name": config.name,
        "status": "complete",
        "research_valid": research_valid,
        "publication_candidate": publication_candidate,
        "claim_scope": (
            "publication candidate for frozen-parameter target-query "
            "policy reuse on the fixed custom CIFAR-10 victim bank"
            if publication_candidate
            else (
                "completed fixed custom CIFAR-10 victim-bank study; "
                "no positive transfer claim"
                if research_valid
                else "no target claim; inspect failed evidence gates"
            )
        ),
        "study_code_digest": code_digest,
        "config": config_record,
        "runtime_environment": runtime_environment or {},
        "source_runs": source_runs,
        "source_competence_gate": source_gate,
        "runs": target_runs,
        "confirmatory_gate": confirmatory_gate,
        "descriptive_target_summary": target_summary,
        "elapsed_seconds": time.monotonic() - started,
        "source_phase_elapsed_seconds": recorded_source_elapsed,
        "target_phase_elapsed_seconds": (
            time.monotonic() - target_started
        ),
        "total_recorded_elapsed_seconds": (
            recorded_source_elapsed
            + time.monotonic()
            - target_started
        ),
    }
    _write_json(study_manifest_path, manifest)
    return manifest


def run_gpu_study(
    config_path: Path,
    *,
    phase: str = "all",
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError(
            "install the vision extra before running the RTX study"
        ) from error
    config, base, locked_config_path = load_validated_study_config(
        config_path
    )
    if config.require_clean_worktree:
        require_clean_protocol_tree(locked_config_path.parents[2])
    data_root = resolve_within_repository(
        base.data_root,
        allowed_directory="data",
        label="CIFAR data root",
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=ToTensor(),
        download=base.download,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=ToTensor(),
        download=base.download,
    )
    study_dir = resolve_within_repository(
        Path(config.output_dir) / config.name,
        allowed_directory="output/rl_transfer",
        label="study directory",
    )
    requirements_path = resolve_within_repository(
        "requirements/rtx-publication.txt",
        allowed_directory="requirements",
        label="RTX requirements",
    )
    environment = capture_runtime_environment(
        study_dir,
        locked_config_path.parents[2],
        requirements_path,
    )
    dataset_digest = tree_digest(data_root)
    return run_gpu_study_from_datasets(
        config,
        train_dataset,
        test_dataset,
        dataset_version=(
            f"torchvision-{torchvision.__version__};"
            f"content-sha256={dataset_digest};"
            f"environment-sha256="
            f"{environment['pip_freeze_sha256']}"
        ),
        phase=phase,
        progress=progress,
        runtime_environment=environment,
    )
