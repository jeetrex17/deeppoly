"""Independent semantic audit for exported Phase 2 calibration evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, fields
import hashlib
import json
import math
from typing import Mapping, Sequence

from .cifar_data import indices_digest
from .config import AttackConfig
from .evaluation_audit import audit_evaluation
from .operator import AttackOperatorContract
from .phase1_export_validation import (
    digest,
    finite_number,
    nonnegative_integer,
    require_mapping,
    require_sequence,
)
from .phase2_calibration_manifest import CALIBRATION_TEMPERATURES, FOLDS
from .phase2_calibration_screen import select_calibration_temperature
from .research_metrics import asr_query_auc
from .results import ResearchResultRow


_SOURCE_SLICES = ("exact_source", "seen_family_new_instance")
_EXPECTED_VICTIMS = {
    "exact_source": 2,
    "seen_family_new_instance": 1,
}
_SAMPLES_PER_VICTIM = 100
_ATTACK = AttackConfig(
    max_queries=50,
    rollback_on_non_improvement=True,
)
_OPERATOR = AttackOperatorContract.from_config(_ATTACK)
_OPERATOR_MAPPING = _OPERATOR.as_dict()
_RESULT_FIELDS = {field.name for field in fields(ResearchResultRow)}
_ENRICHED_RESULT_FIELDS = _RESULT_FIELDS | {
    "heldout_family",
    "source_slice",
    "temperature",
    "hidden_target_calls",
}
_TRACE_FIELDS = {
    "actions",
    "clean_correct",
    "family",
    "heldout_family",
    "hidden_target_calls",
    "l2",
    "linf",
    "method",
    "policy_digest_after",
    "policy_digest_before",
    "query_to_success",
    "query_trace",
    "sample_id",
    "source_slice",
    "success",
    "temperature",
    "total_target_calls",
    "victim_id",
}
_QUERY_EVENT_FIELDS = {
    "call_index",
    "error",
    "feedback",
    "predicted_label",
    "purpose",
    "sample_id",
    "step",
    "victim_id",
}
_EXPECTED_RESULT_ROWS = (
    len(FOLDS)
    * len(CALIBRATION_TEMPERATURES)
    * (2 * 2 * _SAMPLES_PER_VICTIM + 2 * _SAMPLES_PER_VICTIM)
)
_EXPECTED_TRACE_ROWS = (
    len(FOLDS) * len(_SOURCE_SLICES) * (len(FOLDS) - 1) * len(CALIBRATION_TEMPERATURES)
)


def _method(temperature: float) -> str:
    return f"temperature_{temperature:.2f}".replace(".", "p")


def _temperature(value: object, label: str) -> float:
    temperature = finite_number(value, label)
    if temperature not in CALIBRATION_TEMPERATURES:
        raise ValueError(f"{label} is outside the calibration grid")
    return temperature


def _equivalent(left: object, right: object) -> bool:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _equivalent(left[key], right[key]) for key in left
        )
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes))
    ):
        return len(left) == len(right) and all(
            _equivalent(first, second) for first, second in zip(left, right)
        )
    return left == right


def _result_row(
    raw: Mapping[str, object],
    *,
    index: int,
) -> tuple[tuple[str, str, str, float], ResearchResultRow]:
    if set(raw) != _ENRICHED_RESULT_FIELDS:
        raise ValueError(f"result row {index} has an invalid schema")
    heldout = raw.get("heldout_family")
    source_slice = raw.get("source_slice")
    family = raw.get("victim_family")
    temperature = _temperature(raw.get("temperature"), f"result row {index}")
    if (
        heldout not in FOLDS
        or source_slice not in _SOURCE_SLICES
        or family not in FOLDS
        or family == heldout
        or raw.get("method") != _method(temperature)
        or raw.get("hidden_target_calls") != 0
        or raw.get("threat_model") != "T1"
        or raw.get("query_budget") != _ATTACK.max_queries
        or not isinstance(raw.get("clean_correct"), bool)
        or not isinstance(raw.get("success"), bool)
    ):
        raise ValueError(f"result row {index} violates the source-only contract")
    action_trace = require_sequence(
        raw.get("action_trace"),
        f"result row {index} action trace",
    )
    values = {name: raw[name] for name in _RESULT_FIELDS}
    values["action_trace"] = tuple(action_trace)
    try:
        row = ResearchResultRow(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"result row {index} is invalid") from error
    return (
        str(heldout),
        str(source_slice),
        str(family),
        temperature,
    ), row


def _group_results(
    records: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[str, str, str, float], list[ResearchResultRow]],
    int,
]:
    if len(records) != _EXPECTED_RESULT_ROWS:
        raise ValueError("raw result row cardinality is invalid")
    grouped: dict[
        tuple[str, str, str, float],
        list[ResearchResultRow],
    ] = defaultdict(list)
    identities = set()
    source_model_calls = 0
    for index, raw in enumerate(records):
        key, row = _result_row(raw, index=index)
        identity = (*key, row.method, row.victim_id, row.sample_id)
        if identity in identities:
            raise ValueError("raw result rows contain a duplicate identity")
        identities.add(identity)
        grouped[key].append(row)
        source_model_calls += row.total_target_calls
    return dict(grouped), source_model_calls


def _sample_index(
    sample_id: str,
    *,
    family: str,
    victim_id: str,
) -> int:
    prefix = f"cifar10:{family}:{victim_id}:"
    if not sample_id.startswith(prefix):
        raise ValueError("raw sample ID does not match its family and victim")
    suffix = sample_id[len(prefix) :]
    if not suffix.isdigit():
        raise ValueError("raw sample ID has an invalid CIFAR index")
    return int(suffix)


def _expected_cohort(
    rows: Sequence[ResearchResultRow],
    *,
    family: str,
    victim_ids: tuple[str, ...],
    expected_indices_sha: str,
) -> set[str]:
    first_victim = victim_ids[0]
    indices = tuple(
        _sample_index(
            row.sample_id,
            family=family,
            victim_id=first_victim,
        )
        for row in rows
        if row.victim_id == first_victim
    )
    if (
        len(indices) != _SAMPLES_PER_VICTIM
        or len(set(indices)) != _SAMPLES_PER_VICTIM
        or indices_digest(indices) != expected_indices_sha
    ):
        raise ValueError("raw result rows do not match the source image cohort")
    expected = {
        f"cifar10:{family}:{victim_id}:{index}"
        for victim_id in victim_ids
        for index in indices
    }
    if (
        {row.sample_id for row in rows} != expected
        or len(rows) != len(expected)
        or {row.victim_id for row in rows} != set(victim_ids)
    ):
        raise ValueError("raw result rows do not match the expected cohort")
    return expected


def _validate_curve(
    metrics: Mapping[str, object],
    *,
    label: str,
) -> dict[int, float]:
    eligible = nonnegative_integer(metrics.get("eligible"), f"{label} eligible")
    successes = nonnegative_integer(
        metrics.get("successes"),
        f"{label} successes",
    )
    curve = require_mapping(metrics.get("asr_at_budgets"), f"{label} ASR curve")
    try:
        normalized = {
            int(key): finite_number(value, label) for key, value in curve.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} ASR curve is invalid") from error
    ordered = [normalized.get(budget) for budget in range(_ATTACK.max_queries + 1)]
    if (
        eligible <= 0
        or successes > eligible
        or set(normalized) != set(range(_ATTACK.max_queries + 1))
        or any(value is None or not 0.0 <= value <= 1.0 for value in ordered)
        or any(
            float(first) > float(second) + 1e-12
            for first, second in zip(ordered, ordered[1:])
        )
        or not math.isclose(
            float(ordered[-1]),
            successes / eligible,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            finite_number(metrics.get("asr_query_auc"), f"{label} AUC"),
            asr_query_auc({key: float(value) for key, value in normalized.items()}),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{label} ASR curve is inconsistent")
    return {key: float(value) for key, value in normalized.items()}


def _validate_score(
    score: Mapping[str, object],
    *,
    victim_ids: tuple[str, ...],
    label: str,
) -> None:
    overall_curve = _validate_curve(score, label=label)
    if (
        score.get("frozen") is not True
        or score.get("initialization_included") is not True
        or score.get("query_budget") != _ATTACK.max_queries
        or score.get("victim_count") != len(victim_ids)
        or score.get("operator") != _OPERATOR_MAPPING
        or score.get("operator_digest") != _OPERATOR.digest()
    ):
        raise ValueError(f"{label} violates the score-control contract")
    by_victim = require_mapping(score.get("by_victim"), f"{label} by victim")
    if set(by_victim) != set(victim_ids):
        raise ValueError(f"{label} victim summaries are incomplete")
    victim_curves = {}
    for victim_id in victim_ids:
        victim_metrics = require_mapping(
            by_victim[victim_id],
            f"{label}/{victim_id}",
        )
        digest(
            victim_metrics.get("eligible_sample_ids_sha256"),
            f"{label}/{victim_id} eligible cohort",
        )
        victim_curves[victim_id] = _validate_curve(
            victim_metrics,
            label=f"{label}/{victim_id}",
        )
    digest(
        score.get("eligible_sample_ids_sha256"),
        f"{label} eligible cohort",
    )
    if (
        sum(
            int(require_mapping(by_victim[item], label)["eligible"])
            for item in victim_ids
        )
        != score["eligible"]
        or sum(
            int(require_mapping(by_victim[item], label)["successes"])
            for item in victim_ids
        )
        != score["successes"]
    ):
        raise ValueError(f"{label} victim summaries are inconsistent")
    total_eligible = int(score["eligible"])
    expected_curve = {
        budget: sum(
            int(require_mapping(by_victim[victim_id], label)["eligible"])
            * victim_curves[victim_id][budget]
            for victim_id in victim_ids
        )
        / total_eligible
        for budget in range(_ATTACK.max_queries + 1)
    }
    if not _equivalent(overall_curve, expected_curve):
        raise ValueError(f"{label} aggregate curve is inconsistent")


def _validate_temperature_metrics(
    rows: Sequence[ResearchResultRow],
    metrics: Mapping[str, object],
    audit: Mapping[str, object],
    *,
    fold_digest: str,
    expected_sample_ids: set[str],
    victim_ids: tuple[str, ...],
    temperature: float,
    condition_seed: int,
) -> None:
    method = _method(temperature)
    try:
        recomputed_audit = audit_evaluation(
            rows,
            {method: metrics},
            _ATTACK,
            expected_sample_ids=expected_sample_ids,
            expected_victim_ids=set(victim_ids),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("raw calibration row audit could not be recomputed") from error
    action_counts = Counter(action for row in rows for action in row.action_trace)
    action_histogram = {
        str(action): count for action, count in sorted(action_counts.items())
    }
    if (
        recomputed_audit.get("passed") is not True
        or not _equivalent(audit, recomputed_audit)
        or metrics.get("frozen") is not True
        or metrics.get("cohort_matched_score") is not True
        or metrics.get("policy_digest_before") != fold_digest
        or metrics.get("policy_digest_after") != fold_digest
        or metrics.get("operator") != _OPERATOR_MAPPING
        or metrics.get("operator_digest") != _OPERATOR.digest()
        or metrics.get("query_budget") != _ATTACK.max_queries
        or metrics.get("sampling_temperature") != temperature
        or metrics.get("deterministic_actions") is not False
        or metrics.get("action_histogram") != action_histogram
        or metrics.get("source_model_calls")
        != sum(row.total_target_calls for row in rows)
        or any(row.policy_digest != fold_digest for row in rows)
        or any(row.seed != condition_seed for row in rows)
    ):
        raise ValueError("temperature summary does not match audited raw rows")


def _temperature_one_digest(
    rows: Sequence[ResearchResultRow],
) -> str:
    records = []
    for row in rows:
        record = asdict(row)
        del record["method"]
        records.append(record)
    records.sort(key=lambda row: (str(row["victim_id"]), str(row["sample_id"])))
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _pool_metrics(metrics: Sequence[Mapping[str, object]]) -> dict[str, object]:
    eligible = sum(int(item["eligible"]) for item in metrics)
    successes = sum(int(item["successes"]) for item in metrics)
    return {
        "eligible": eligible,
        "successes": successes,
        "asr": sum(int(item["successes"]) / int(item["eligible"]) for item in metrics)
        / len(metrics),
        "auc": sum(float(item["asr_query_auc"]) for item in metrics) / len(metrics),
        "pooled_asr": successes / eligible,
        "frozen": all(item.get("frozen") is True for item in metrics),
        "source_model_calls": sum(
            int(item.get("source_model_calls", 0)) for item in metrics
        ),
    }


def _require_summary_match(
    recorded: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    if any(
        key not in recorded or not _equivalent(recorded[key], value)
        for key, value in expected.items()
    ):
        raise ValueError(f"{label} is inconsistent with condition metrics")


def _validate_fold(
    fold: Mapping[str, object],
    grouped: Mapping[
        tuple[str, str, str, float],
        Sequence[ResearchResultRow],
    ],
) -> str:
    heldout = fold.get("heldout_family")
    if heldout not in FOLDS:
        raise ValueError("calibration fold held-out family is invalid")
    expected_families = tuple(family for family in FOLDS if family != heldout)
    source_families = tuple(
        require_sequence(fold.get("source_families"), "fold source families")
    )
    fold_digest = digest(
        fold.get("policy_persistent_digest"),
        "fold policy digest",
    )
    digest(fold.get("policy_checkpoint_sha256"), "fold checkpoint digest")
    if (
        fold.get("seed") != 17
        or source_families != expected_families
        or fold.get("complete") is not True
        or fold.get("raw_evidence_audited") is not True
        or fold.get("temperature_one_reproduced") is not True
        or fold.get("target_calls") != 0
        or fold.get("target_evaluation_performed") is not False
    ):
        raise ValueError("calibration fold violates the source-only contract")
    conditions = require_mapping(fold.get("conditions"), "fold conditions")
    expected_condition_keys = {
        f"{source_slice}/{family}"
        for source_slice in _SOURCE_SLICES
        for family in expected_families
    }
    if set(conditions) != expected_condition_keys:
        raise ValueError("fold condition grid is incomplete")

    score_blocks = []
    temperature_blocks = {temperature: [] for temperature in CALIBRATION_TEMPERATURES}
    fold_calls = 0
    condition_index_digests = set()
    temperature_one_digests = []
    for slice_index, source_slice in enumerate(_SOURCE_SLICES):
        for family_index, family in enumerate(expected_families):
            key = f"{source_slice}/{family}"
            condition = require_mapping(conditions[key], f"condition {key}")
            victim_ids = tuple(
                require_sequence(condition.get("victim_ids"), f"{key} victims")
            )
            if (
                condition.get("source_slice") != source_slice
                or condition.get("family") != family
                or condition.get("condition_seed")
                != 800_017 + 10_000 * slice_index + family_index
                or len(victim_ids) != _EXPECTED_VICTIMS[source_slice]
                or len(set(victim_ids)) != len(victim_ids)
                or any(not isinstance(item, str) or not item for item in victim_ids)
            ):
                raise ValueError(f"condition {key} is inconsistent")
            indices_sha = digest(condition.get("indices_sha256"), f"{key} indices")
            condition_index_digests.add(indices_sha)
            score = require_mapping(condition.get("score_greedy"), f"{key} score")
            _validate_score(score, victim_ids=victim_ids, label=f"{key} score")
            score_blocks.append(score)
            temperatures = require_mapping(
                condition.get("temperatures"),
                f"{key} temperatures",
            )
            audits = require_mapping(
                condition.get("raw_row_audits"),
                f"{key} raw audits",
            )
            expected_temperature_keys = {
                str(temperature) for temperature in CALIBRATION_TEMPERATURES
            }
            if (
                set(temperatures) != expected_temperature_keys
                or set(audits) != expected_temperature_keys
            ):
                raise ValueError(f"condition {key} temperature grid is incomplete")
            expected_sample_ids: set[str] | None = None
            for temperature in CALIBRATION_TEMPERATURES:
                raw_rows = tuple(
                    grouped.get(
                        (str(heldout), source_slice, family, temperature),
                        (),
                    )
                )
                expected_count = _EXPECTED_VICTIMS[source_slice] * _SAMPLES_PER_VICTIM
                if len(raw_rows) != expected_count:
                    raise ValueError("raw result row cardinality is invalid")
                cohort = _expected_cohort(
                    raw_rows,
                    family=family,
                    victim_ids=victim_ids,
                    expected_indices_sha=indices_sha,
                )
                if expected_sample_ids is None:
                    expected_sample_ids = cohort
                elif cohort != expected_sample_ids:
                    raise ValueError("temperatures use different raw cohorts")
                metrics = require_mapping(
                    temperatures[str(temperature)],
                    f"{key} temperature {temperature}",
                )
                raw_audit = require_mapping(
                    audits[str(temperature)],
                    f"{key} audit {temperature}",
                )
                _validate_temperature_metrics(
                    raw_rows,
                    metrics,
                    raw_audit,
                    fold_digest=fold_digest,
                    expected_sample_ids=cohort,
                    victim_ids=victim_ids,
                    temperature=temperature,
                    condition_seed=int(condition["condition_seed"]),
                )
                if math.isclose(temperature, 1.0):
                    temperature_one_digests.append(_temperature_one_digest(raw_rows))
                score_asr = int(score["successes"]) / int(score["eligible"])
                score_auc = float(score["asr_query_auc"])
                expected = {
                    "asr": int(metrics["successes"]) / int(metrics["eligible"]),
                    "auc": float(metrics["asr_query_auc"]),
                    "asr_gain_vs_score": (
                        int(metrics["successes"]) / int(metrics["eligible"]) - score_asr
                    ),
                    "auc_gain_vs_score": float(metrics["asr_query_auc"]) - score_auc,
                }
                _require_summary_match(
                    metrics,
                    expected,
                    label=f"{key} temperature {temperature}",
                )
                if metrics.get("eligible_sample_ids_sha256") != score.get(
                    "eligible_sample_ids_sha256"
                ):
                    raise ValueError("temperature and score cohorts do not match")
                score_by_victim = require_mapping(
                    score.get("by_victim"),
                    f"{key} score by victim",
                )
                temperature_by_victim = require_mapping(
                    metrics.get("by_victim"),
                    f"{key} temperature by victim",
                )
                if any(
                    require_mapping(score_by_victim[victim_id], key).get(
                        "eligible_sample_ids_sha256"
                    )
                    != require_mapping(temperature_by_victim[victim_id], key).get(
                        "eligible_sample_ids_sha256"
                    )
                    for victim_id in victim_ids
                ):
                    raise ValueError("per-victim score cohorts do not match")
                temperature_blocks[temperature].append(metrics)
                fold_calls += int(metrics["source_model_calls"])

    pooled_score = _pool_metrics(score_blocks)
    pooled_score["source_model_calls"] = 0
    _require_summary_match(
        require_mapping(fold.get("score_greedy"), "fold score"),
        {
            **pooled_score,
            "reused_from_verified_phase2_evidence": True,
        },
        label="fold score",
    )
    fold_temperatures = require_mapping(
        fold.get("temperatures"),
        "fold temperatures",
    )
    if set(fold_temperatures) != {
        str(temperature) for temperature in CALIBRATION_TEMPERATURES
    }:
        raise ValueError("fold temperature grid is incomplete")
    for temperature, blocks in temperature_blocks.items():
        pooled = _pool_metrics(blocks)
        expected = {
            **pooled,
            "asr_gain_vs_score": float(pooled["asr"]) - float(pooled_score["asr"]),
            "auc_gain_vs_score": float(pooled["auc"]) - float(pooled_score["auc"]),
        }
        _require_summary_match(
            require_mapping(
                fold_temperatures[str(temperature)],
                f"fold temperature {temperature}",
            ),
            expected,
            label=f"fold temperature {temperature}",
        )
    if fold.get("source_model_calls") != fold_calls:
        raise ValueError("fold source-model call count is inconsistent")
    expected_temperature_one = hashlib.sha256(
        "\n".join(sorted(temperature_one_digests)).encode("utf-8")
    ).hexdigest()
    if (
        len(temperature_one_digests) != 4
        or digest(
            fold.get("temperature_one_row_sha256"),
            "fold temperature-one row digest",
        )
        != expected_temperature_one
    ):
        raise ValueError("fold temperature-one row digest is inconsistent")
    if len(condition_index_digests) != 1:
        raise ValueError("fold conditions use different source image cohorts")
    return next(iter(condition_index_digests))


def _validate_trace(
    trace: Mapping[str, object],
    *,
    rows: Sequence[ResearchResultRow],
    fold_digest: str,
) -> None:
    if set(trace) != _TRACE_FIELDS:
        raise ValueError("query trace has an invalid schema")
    sample_id = trace.get("sample_id")
    victim_id = trace.get("victim_id")
    matches = [
        row for row in rows if row.sample_id == sample_id and row.victim_id == victim_id
    ]
    if len(matches) != 1:
        raise ValueError("query trace identity does not match one raw row")
    row = matches[0]
    actions = tuple(require_sequence(trace.get("actions"), "trace actions"))
    events = tuple(require_sequence(trace.get("query_trace"), "query trace events"))
    if (
        trace.get("hidden_target_calls") != 0
        or trace.get("policy_digest_before") != fold_digest
        or trace.get("policy_digest_after") != fold_digest
        or actions != row.action_trace
        or trace.get("clean_correct") is not row.clean_correct
        or trace.get("success") is not row.success
        or trace.get("query_to_success") != row.query_to_success
        or trace.get("total_target_calls") != row.total_target_calls
        or not _equivalent(trace.get("linf"), row.linf)
        or not _equivalent(trace.get("l2"), row.l2)
        or len(events) != row.total_target_calls
    ):
        raise ValueError("query trace does not match its raw result row")
    for offset, raw_event in enumerate(events):
        event = require_mapping(raw_event, f"query trace event {offset}")
        if (
            set(event) != _QUERY_EVENT_FIELDS
            or event.get("call_index") != offset + 1
            or event.get("step") != offset
            or event.get("purpose") != ("initialization" if offset == 0 else "attack")
            or event.get("feedback") != "scores"
            or event.get("error") is not None
            or event.get("sample_id") != sample_id
            or event.get("victim_id") != victim_id
            or isinstance(event.get("predicted_label"), bool)
            or not isinstance(event.get("predicted_label"), int)
            or not 0 <= int(event["predicted_label"]) <= 9
        ):
            raise ValueError("query trace event is invalid")


def _validate_traces(
    traces: Sequence[Mapping[str, object]],
    grouped: Mapping[
        tuple[str, str, str, float],
        Sequence[ResearchResultRow],
    ],
    folds: Sequence[Mapping[str, object]],
) -> None:
    if len(traces) != _EXPECTED_TRACE_ROWS:
        raise ValueError("query trace cardinality is invalid")
    fold_digests = {
        str(fold["heldout_family"]): digest(
            fold.get("policy_persistent_digest"),
            "fold policy digest",
        )
        for fold in folds
    }
    seen = set()
    for index, trace in enumerate(traces):
        heldout = trace.get("heldout_family")
        source_slice = trace.get("source_slice")
        family = trace.get("family")
        temperature = _temperature(trace.get("temperature"), f"trace {index}")
        key = (heldout, source_slice, family, temperature)
        if (
            heldout not in FOLDS
            or source_slice not in _SOURCE_SLICES
            or family not in FOLDS
            or family == heldout
            or trace.get("method") != _method(temperature)
            or key in seen
            or key not in grouped
        ):
            raise ValueError("query traces contain a duplicate or invalid identity")
        seen.add(key)
        _validate_trace(
            trace,
            rows=grouped[key],
            fold_digest=fold_digests[str(heldout)],
        )
    if seen != set(grouped):
        raise ValueError("query trace grid is incomplete")


def validate_calibration_evidence(
    folds: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
    traces: Sequence[Mapping[str, object]],
) -> int:
    """Audit raw records, summaries, traces, and the persisted decision."""

    grouped, source_model_calls = _group_results(results)
    expected_keys = {
        (heldout, source_slice, family, temperature)
        for heldout in FOLDS
        for source_slice in _SOURCE_SLICES
        for family in FOLDS
        if family != heldout
        for temperature in CALIBRATION_TEMPERATURES
    }
    if set(grouped) != expected_keys:
        raise ValueError("raw calibration row grid is incomplete")
    index_digests = {_validate_fold(fold, grouped) for fold in folds}
    if len(index_digests) != 1:
        raise ValueError("calibration folds use different source image cohorts")
    _validate_traces(traces, grouped, folds)
    recomputed = select_calibration_temperature(folds)
    if not _equivalent(decision, recomputed):
        raise ValueError("calibration decision is inconsistent with recomputed folds")
    return source_model_calls


__all__ = ("validate_calibration_evidence",)
