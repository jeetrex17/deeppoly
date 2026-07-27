"""Verified source evidence for the two-phase GPU protocol."""

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
from .results import ResearchResultRow, read_jsonl, write_jsonl
from .source_gates import (
    SourceGateThresholds,
    summarize_source_competence,
)
from .verified_artifacts import load_verified_json, write_verified_json


VictimPopulation = Mapping[str, Sequence[tuple[str, nn.Module]]]


@dataclass(frozen=True)
class SourceEvidence:
    evaluation: dict[str, dict[str, object]]
    gate: dict[str, object]
    audits: dict[str, object]
    cache_resumed: bool
    results_sha256: str
    evaluation_elapsed_seconds: float


def _audit_slices(
    rows: Sequence[ResearchResultRow],
    evaluation: Mapping[str, object],
    populations: Mapping[str, VictimPopulation],
    attack: AttackConfig,
    indices: Sequence[int],
) -> dict[str, object]:
    audits: dict[str, object] = {}
    for slice_name, families in populations.items():
        slice_evaluation = evaluation.get(slice_name)
        if not isinstance(slice_evaluation, Mapping):
            raise ValueError(f"missing cached source slice: {slice_name}")
        family_audits: dict[str, object] = {}
        for family, victims in families.items():
            method_metrics = slice_evaluation.get(family)
            if not isinstance(method_metrics, Mapping):
                raise ValueError(
                    f"missing cached source family: {slice_name}/{family}"
                )
            victim_ids = {victim_id for victim_id, _ in victims}
            expected_sample_ids = {
                f"cifar10:{family}:{victim_id}:{index}"
                for victim_id in victim_ids
                for index in indices
            }
            selected_rows = tuple(
                row
                for row in rows
                if row.victim_family == family
                and row.victim_id in victim_ids
            )
            audit = audit_evaluation(
                selected_rows,
                method_metrics,
                attack,
                expected_sample_ids=expected_sample_ids,
                expected_victim_ids=victim_ids,
            )
            if not audit["passed"]:
                raise ValueError(
                    f"cached source rows failed audit: "
                    f"{slice_name}/{family}"
                )
            family_audits[family] = audit
        audits[slice_name] = family_audits
    return audits


def source_evidence(
    *,
    policy: RecurrentAttackPolicy,
    additional_policies: Mapping[
        str,
        tuple[RecurrentAttackPolicy, bool],
    ],
    source_victims: VictimPopulation,
    source_holdout_victims: VictimPopulation,
    samples: tuple[tuple[torch.Tensor, int], ...],
    indices: Sequence[int],
    attack: AttackConfig,
    seed: int,
    main_method_prefix: str,
    trace_samples_per_method: int,
    thresholds: SourceGateThresholds,
    run_dir: Path,
    binding: dict[str, object],
    resume: bool,
    report: Callable[[str], None],
) -> SourceEvidence:
    cache_path = resolve_descendant(
        run_dir,
        "source_evaluation.json",
        label="source evaluation cache",
    )
    results_path = resolve_descendant(
        run_dir,
        "source_results.jsonl",
        label="source result rows",
    )
    traces_path = resolve_descendant(
        run_dir,
        "source_query_traces.jsonl",
        label="source query traces",
    )
    populations = {
        "exact_source": source_victims,
        "seen_family_new_instance": source_holdout_victims,
    }
    if (
        resume
        and cache_path.is_file()
        and cache_path.with_suffix(
            cache_path.suffix + ".sha256"
        ).is_file()
    ):
        report("revalidating cached raw source evidence")
        cached = load_verified_json(cache_path)
        if cached.get("binding") != binding:
            raise ValueError("source evidence binding mismatch")
        if (
            cached.get("results_sha256")
            != sha256_file(results_path)
            or cached.get("query_traces_sha256")
            != sha256_file(traces_path)
        ):
            raise ValueError("source evidence row checksum mismatch")
        evaluation = cached.get("source_evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("cached source evaluation is invalid")
        rows = read_jsonl(results_path)
        audits = _audit_slices(
            rows,
            evaluation,
            populations,
            attack,
            indices,
        )
        gate = summarize_source_competence(
            evaluation,
            thresholds,
        )
        return SourceEvidence(
            evaluation=evaluation,
            gate=gate,
            audits=audits,
            cache_resumed=True,
            results_sha256=sha256_file(results_path),
            evaluation_elapsed_seconds=float(
                cached.get("evaluation_elapsed_seconds", 0.0)
            ),
        )

    report("evaluating source competence from frozen checkpoints")
    evaluation_started = time.monotonic()
    evaluation: dict[str, dict[str, object]] = {
        slice_name: {} for slice_name in populations
    }
    all_rows: list[ResearchResultRow] = []
    traces: list[dict[str, object]] = []
    for slice_offset, (slice_name, families) in enumerate(
        populations.items()
    ):
        for family_offset, (family, victims) in enumerate(
            families.items()
        ):
            if not victims:
                continue
            rows, family_traces, family_evaluation = evaluate_methods(
                policy,
                victims,
                samples,
                indices,
                attack,
                seed
                + 10_000 * slice_offset
                + family_offset,
                family,
                report,
                trace_samples_per_method=trace_samples_per_method,
                additional_policies=additional_policies,
                main_method_prefix=main_method_prefix,
            )
            evaluation[slice_name][family] = family_evaluation
            all_rows.extend(rows)
            traces.extend(
                {"source_slice": slice_name, **trace}
                for trace in family_traces
            )
    write_jsonl(results_path, all_rows)
    traces_path.write_text(
        "".join(
            json.dumps(trace, sort_keys=True) + "\n"
            for trace in traces
        )
    )
    audits = _audit_slices(
        all_rows,
        evaluation,
        populations,
        attack,
        indices,
    )
    gate = summarize_source_competence(evaluation, thresholds)
    evaluation_elapsed = time.monotonic() - evaluation_started
    write_verified_json(
        cache_path,
        {
            "binding": binding,
            "results_sha256": sha256_file(results_path),
            "query_traces_sha256": sha256_file(traces_path),
            "source_evaluation": evaluation,
            "evaluation_elapsed_seconds": evaluation_elapsed,
        },
    )
    return SourceEvidence(
        evaluation=evaluation,
        gate=gate,
        audits=audits,
        cache_resumed=False,
        results_sha256=sha256_file(results_path),
        evaluation_elapsed_seconds=evaluation_elapsed,
    )
