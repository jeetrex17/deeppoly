"""Resumable, cohort-verified target evaluation for CIFAR studies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import torch
from torch import nn

from .artifacts import sha256_file
from .cifar_evaluation import evaluate_methods
from .config import AttackConfig
from .evaluation_audit import audit_evaluation
from .paths import resolve_descendant
from .recurrent import RecurrentAttackPolicy
from .results import read_jsonl, write_jsonl
from .verified_artifacts import load_verified_json, write_verified_json


@dataclass(frozen=True)
class TargetEvidence:
    evaluation: dict[str, object]
    audit: dict[str, object]
    accuracy_by_victim: dict[str, float]
    cache_resumed: bool
    evaluation_elapsed_seconds: float


def _expected_cohort(
    target_family: str,
    victim_ids: Sequence[str],
    indices: Sequence[int],
) -> set[str]:
    return {
        f"cifar10:{target_family}:{victim_id}:{index}"
        for victim_id in victim_ids
        for index in indices
    }


def target_evidence(
    *,
    policy: RecurrentAttackPolicy,
    additional_policies: Mapping[
        str,
        tuple[RecurrentAttackPolicy, bool],
    ],
    target_victims: Sequence[tuple[str, nn.Module]],
    samples: tuple[tuple[torch.Tensor, int], ...],
    indices: Sequence[int],
    attack: AttackConfig,
    seed: int,
    target_family: str,
    trace_samples_per_method: int,
    main_method_prefix: str,
    run_dir: Path,
    binding: dict[str, object],
    resume: bool,
    report: Callable[[str], None],
    accuracy_by_victim: Callable[[], dict[str, float]],
    verify_code_unchanged: Callable[[], None],
) -> TargetEvidence:
    """Evaluate or revalidate one locked target cell."""

    cache_path = resolve_descendant(
        run_dir,
        "target_evaluation.json",
        label="target evaluation cache",
    )
    results_path = resolve_descendant(
        run_dir,
        "results.jsonl",
        label="target result rows",
    )
    traces_path = resolve_descendant(
        run_dir,
        "query_traces.jsonl",
        label="target query traces",
    )
    victim_ids = tuple(victim_id for victim_id, _ in target_victims)
    expected_sample_ids = _expected_cohort(
        target_family,
        victim_ids,
        indices,
    )
    checksum_path = resolve_descendant(
        run_dir,
        "target_evaluation.json.sha256",
        label="target evaluation checksum",
    )
    if resume and cache_path.is_file() and checksum_path.is_file():
        report("loading verified completed target cell")
        cached = load_verified_json(cache_path)
        if cached.get("binding") != binding:
            raise ValueError("target cache binding mismatch")
        if (
            cached.get("results_sha256") != sha256_file(results_path)
            or cached.get("query_traces_sha256")
            != sha256_file(traces_path)
        ):
            raise ValueError("target cache result checksum mismatch")
        evaluation = cached.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("target cache evaluation is invalid")
        audit = audit_evaluation(
            read_jsonl(results_path),
            evaluation,
            attack,
            expected_sample_ids=expected_sample_ids,
            expected_victim_ids=victim_ids,
        )
        if not audit["passed"]:
            raise ValueError(
                "cached target evaluation failed raw-row audit"
            )
        target_accuracy = cached.get(
            "target_test_accuracy_by_victim"
        )
        if not isinstance(target_accuracy, dict) or set(
            target_accuracy
        ) != set(victim_ids):
            raise ValueError("target cache accuracy block is invalid")
        return TargetEvidence(
            evaluation=evaluation,
            audit=audit,
            accuracy_by_victim={
                str(victim_id): float(value)
                for victim_id, value in target_accuracy.items()
            },
            cache_resumed=True,
            evaluation_elapsed_seconds=float(
                cached.get(
                    "target_evaluation_elapsed_seconds",
                    0.0,
                )
            ),
        )

    started = time.monotonic()
    rows, traces, evaluation = evaluate_methods(
        policy,
        target_victims,
        samples,
        indices,
        attack,
        seed,
        target_family,
        report,
        trace_samples_per_method=trace_samples_per_method,
        additional_policies=additional_policies,
        main_method_prefix=main_method_prefix,
    )
    audit = audit_evaluation(
        rows,
        evaluation,
        attack,
        expected_sample_ids=expected_sample_ids,
        expected_victim_ids=victim_ids,
    )
    if not audit["passed"]:
        raise RuntimeError("fresh target evaluation failed its raw audit")
    verify_code_unchanged()
    write_jsonl(results_path, rows)
    traces_path.write_text(
        "".join(
            json.dumps(trace, sort_keys=True) + "\n"
            for trace in traces
        )
    )
    target_accuracy = accuracy_by_victim()
    if set(target_accuracy) != set(victim_ids):
        raise ValueError("fresh target accuracy block is incomplete")
    elapsed = time.monotonic() - started
    write_verified_json(
        cache_path,
        {
            "binding": binding,
            "results_sha256": sha256_file(results_path),
            "query_traces_sha256": sha256_file(traces_path),
            "evaluation": evaluation,
            "target_test_accuracy_by_victim": target_accuracy,
            "target_evaluation_elapsed_seconds": elapsed,
        },
    )
    return TargetEvidence(
        evaluation=evaluation,
        audit=audit,
        accuracy_by_victim=target_accuracy,
        cache_resumed=False,
        evaluation_elapsed_seconds=elapsed,
    )
