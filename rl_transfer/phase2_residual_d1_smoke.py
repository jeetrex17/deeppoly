"""Non-promotable real-GPU smoke test for the locked D1 pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
import time

import torch
from torch.utils.data import Dataset

from .artifacts import (
    exclusive_file_lock,
    load_recurrent_checkpoint,
    save_recurrent_checkpoint,
)
from .phase2_residual_d1 import (
    D1_SOURCE_FAMILIES,
    ResidualD1Request,
    validate_residual_source_records,
)
from .phase2_residual_d1_evaluation import evaluate_residual_policy_cohort
from .phase2_residual_d1_source import (
    _validated_runtime_environment,
    load_d1_source_context,
)
from .phase2_residual_d1_teacher import (
    D1_HIDDEN_DIM,
    D1_PRIOR_TEMPERATURE,
    _collect_teacher_blocks,
    _write_verified_jsonl,
)
from .recurrent import PPOConfig, RecurrentAttackPolicy
from .residual_bc import fit_residual_ranker_bc
from .residual_ranker import ResidualRankerPolicy
from .verified_artifacts import write_verified_json


D1_SMOKE_EPISODES = 10
D1_SMOKE_DECISIONS = 2
D1_SMOKE_BC_EPOCHS = 1
D1_SMOKE_MAX_SECONDS = 300.0


def build_residual_d1_smoke_plan(
    request: ResidualD1Request,
) -> dict[str, object]:
    """Return the immutable, explicitly non-promotable smoke contract."""

    return {
        "schema_version": 1,
        "name": "phase2-d1-real-gpu-smoke",
        "smoke_only": True,
        "research_valid": False,
        "publication_candidate": False,
        "request_sha256": request.digest(),
        "source_families": list(D1_SOURCE_FAMILIES),
        "heldout_family": request.heldout_family,
        "seed": request.seed,
        "teacher_episodes": D1_SMOKE_EPISODES,
        "teacher_decisions": D1_SMOKE_DECISIONS,
        "bc_epochs": D1_SMOKE_BC_EPOCHS,
        "training_role": "subset_of_d1_train_only",
        "evaluation_role": "subset_of_d1_train_only",
        "query_budget": 50,
        "deadline_seconds": D1_SMOKE_MAX_SECONDS,
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "hidden_target_evaluation_performed": False,
        "authorizes_d1_promotion": False,
        "authorizes_hidden_target_evaluation": False,
    }


def run_residual_d1_gpu_smoke(
    request: ResidualD1Request,
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    dataset_content_sha256: str,
    runtime_environment: Mapping[str, object],
    progress: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Exercise source loading, teacher, BC, checkpoint, and paired attack."""

    if not torch.cuda.is_available() or request.device != "cuda":
        raise RuntimeError("the D1 real smoke test requires the remote CUDA GPU")
    if not callable(progress) or not callable(clock):
        raise TypeError("D1 smoke callbacks must be callable")
    request.output_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(request.output_dir / ".smoke.lock"):
        unexpected = tuple(
            path.name
            for path in request.output_dir.iterdir()
            if path.name != ".smoke.lock"
        )
        if unexpected:
            raise ValueError("D1 smoke output directory must be fresh")
        started = clock()
        deadline = started + D1_SMOKE_MAX_SECONDS

        def deadline_check() -> None:
            if clock() >= deadline:
                raise TimeoutError("D1 real-GPU smoke deadline reached")

        plan = {
            **build_residual_d1_smoke_plan(request),
            "status": "running",
            "runtime_environment": _validated_runtime_environment(runtime_environment),
        }
        manifest_path = request.output_dir / "smoke_manifest.json"
        write_verified_json(manifest_path, plan)
        context = load_d1_source_context(
            request,
            train_dataset,
            test_dataset,
            dataset_content_sha256=dataset_content_sha256,
        )
        attack = context.config.attack_config()
        victims = {
            family: (context.teacher_victims[family][0],)
            for family in D1_SOURCE_FAMILIES
        }
        steps, teacher = _collect_teacher_blocks(
            victims=victims,
            samples=context.train_samples[:D1_SMOKE_EPISODES],
            config=attack,
            episodes=D1_SMOKE_EPISODES,
            decisions=D1_SMOKE_DECISIONS,
            seed=request.seed + 91_000,
            role="gpu_smoke_train",
            deadline_check=deadline_check,
            progress=progress,
        )
        backbone = RecurrentAttackPolicy(
            attack.recurrent_observation_dim,
            attack.action_dim,
            hidden_dim=D1_HIDDEN_DIM,
            seed=request.seed + 92_000,
            config=PPOConfig(
                learning_rate=context.config.policy_learning_rate,
                entropy_weight=context.config.policy_entropy_weight,
                update_epochs=1,
            ),
            actor_mode="action_conditioned",
            action_grid_size=attack.grid_size,
        ).to(torch.device(request.device))
        training = fit_residual_ranker_bc(
            backbone,
            steps,
            epochs=D1_SMOKE_BC_EPOCHS,
            seed=request.seed + 93_000,
            prior_seed=request.seed + 94_000,
            prior_temperature=D1_PRIOR_TEMPERATURE,
            deadline_check=deadline_check,
            required_source_families=D1_SOURCE_FAMILIES,
        )
        checkpoint_path = request.output_dir / "smoke_residual_ranker.pt"
        checkpoint_sha256 = save_recurrent_checkpoint(
            checkpoint_path,
            backbone,
            {
                "kind": "phase2_d1_non_promotable_gpu_smoke",
                "request_sha256": request.digest(),
                "target_calls": 0,
                "hidden_target_calls": 0,
                "target_evaluation_performed": False,
                "hidden_target_evaluation_performed": False,
                "target_evaluation_available": False,
                "authorizes_hidden_target_evaluation": False,
            },
        )
        loaded, metadata = load_recurrent_checkpoint(
            checkpoint_path,
            request.device,
            expected_observation_dim=attack.recurrent_observation_dim,
            expected_action_dim=attack.action_dim,
            expected_hidden_dim=D1_HIDDEN_DIM,
            expected_actor_mode="action_conditioned",
        )
        if (
            loaded.persistent_digest() != backbone.persistent_digest()
            or metadata.get("target_calls") != 0
        ):
            raise ValueError("D1 smoke checkpoint round trip failed")
        policy = ResidualRankerPolicy(
            loaded,
            confidence_threshold=0.0,
            prior_temperature=D1_PRIOR_TEMPERATURE,
            overrides_enabled=False,
        )
        conditions: dict[str, object] = {}
        rows = []
        traces = []
        for offset, family in enumerate(D1_SOURCE_FAMILIES):
            condition, family_rows, family_traces = evaluate_residual_policy_cohort(
                policies={"smoke_residual_fallback": policy},
                victims=(context.teacher_victims[family][0],),
                samples=(context.train_samples[offset],),
                indices=(context.train_indices[offset],),
                attack=attack,
                family=family,
                seed=request.seed + 95_000 + offset,
                heldout_family=request.heldout_family,
                source_slice="exact_source",
                deadline_check=deadline_check,
                progress=progress,
            )
            conditions[family] = condition
            rows.extend(family_rows)
            traces.extend(family_traces)
        enriched_rows = [
            {
                **asdict(row),
                "heldout_family": request.heldout_family,
                "source_slice": "exact_source",
                "target_calls": 0,
                "hidden_target_calls": 0,
            }
            for row in rows
        ]
        validate_residual_source_records(
            enriched_rows,
            heldout_family=request.heldout_family,
        )
        validate_residual_source_records(
            traces,
            heldout_family=request.heldout_family,
        )
        results_sha256 = _write_verified_jsonl(
            request.output_dir / "smoke_results.jsonl",
            enriched_rows,
        )
        traces_sha256 = _write_verified_jsonl(
            request.output_dir / "smoke_query_traces.jsonl",
            traces,
        )
        final = {
            **plan,
            "status": "complete",
            "elapsed_seconds": clock() - started,
            "teacher": teacher,
            "training": training,
            "checkpoint_sha256": checkpoint_sha256,
            "source_evaluation": conditions,
            "result_rows": len(rows),
            "query_traces": len(traces),
            "results_sha256": results_sha256,
            "query_traces_sha256": traces_sha256,
            "target_calls": 0,
            "hidden_target_calls": 0,
            "target_evaluation_performed": False,
            "hidden_target_evaluation_performed": False,
            "authorizes_d1_promotion": False,
            "authorizes_hidden_target_evaluation": False,
        }
        write_verified_json(manifest_path, final)
        return final


__all__ = (
    "D1_SMOKE_BC_EPOCHS",
    "D1_SMOKE_DECISIONS",
    "D1_SMOKE_EPISODES",
    "D1_SMOKE_MAX_SECONDS",
    "build_residual_d1_smoke_plan",
    "run_residual_d1_gpu_smoke",
)
