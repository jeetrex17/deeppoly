"""Independent raw-evidence verification and SVG plots for D1."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

from .artifacts import _atomic_text_write, sha256_file
from .phase2_residual_d1 import (
    D1_HELDOUT_FAMILY,
    D1_SOURCE_FAMILIES,
    validate_residual_source_records,
)
from .research_metrics import AttackOutcome, asr_at_budgets, asr_query_auc
from .results import ResearchResultRow


_ROW_FIELDS = {
    "sample_id",
    "victim_id",
    "victim_family",
    "method",
    "threat_model",
    "seed",
    "query_budget",
    "clean_correct",
    "success",
    "query_to_success",
    "total_target_calls",
    "linf",
    "l2",
    "policy_digest",
    "action_trace",
}
_COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed")


def load_verified_jsonl_records(
    path: Path,
    *,
    max_bytes: int = 512 * 1024 * 1024,
) -> tuple[dict[str, object], ...]:
    """Load canonical JSONL only when its SHA-256 sidecar matches."""

    selected = Path(path)
    sidecar = selected.with_suffix(selected.suffix + ".sha256")
    if (
        selected.is_symlink()
        or sidecar.is_symlink()
        or not selected.is_file()
        or not sidecar.is_file()
        or not 0 < selected.stat().st_size <= max_bytes
    ):
        raise ValueError("verified D1 JSONL artifact is missing or unsafe")
    expected = sidecar.read_text().strip()
    if (
        len(expected) != 64
        or sha256_file(selected) != expected
        or not selected.read_bytes().endswith(b"\n")
    ):
        raise ValueError("verified D1 JSONL checksum or framing failed")
    records: list[dict[str, object]] = []
    for line in selected.read_bytes().splitlines(keepends=True):
        try:
            decoded = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("verified D1 JSONL contains invalid JSON") from error
        canonical = (
            json.dumps(
                decoded,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        if line != canonical or not isinstance(decoded, dict):
            raise ValueError("verified D1 JSONL is noncanonical")
        records.append(decoded)
    if not records:
        raise ValueError("verified D1 JSONL cannot be empty")
    return tuple(records)


def _row(record: Mapping[str, object]) -> ResearchResultRow:
    if not _ROW_FIELDS <= set(record):
        raise ValueError("D1 raw row is missing required fields")
    values = {field: record[field] for field in _ROW_FIELDS}
    action_trace = values["action_trace"]
    if not isinstance(action_trace, list):
        raise ValueError("D1 action trace must be a JSON array")
    return ResearchResultRow(
        **{**values, "action_trace": tuple(action_trace)}  # type: ignore[arg-type]
    )


def _summary(rows: Sequence[ResearchResultRow]) -> dict[str, object]:
    eligible_rows = tuple(row for row in rows if row.clean_correct)
    if not eligible_rows:
        raise ValueError("D1 evidence condition has no clean-correct samples")
    outcomes = tuple(
        AttackOutcome(row.clean_correct, row.query_to_success) for row in rows
    )
    budgets = tuple(range(51))
    curve = asr_at_budgets(outcomes, budgets)
    queries = tuple(
        row.query_to_success
        for row in eligible_rows
        if row.query_to_success is not None
    )
    action_counts = Counter(action for row in rows for action in row.action_trace)
    action_total = sum(action_counts.values())
    action_dim = 96
    entropy = (
        -sum(
            (count / action_total) * math.log(count / action_total)
            for count in action_counts.values()
        )
        / math.log(action_dim)
        if action_total
        else 0.0
    )
    return {
        "rows": len(rows),
        "eligible": len(eligible_rows),
        "successes": sum(row.success for row in eligible_rows),
        "asr_at_budgets": {str(key): value for key, value in curve.items()},
        "asr_query_auc": asr_query_auc(curve),
        "mean_queries_to_success": (statistics.fmean(queries) if queries else None),
        "median_queries_to_success": (statistics.median(queries) if queries else None),
        "mean_source_calls": statistics.fmean(row.total_target_calls for row in rows),
        "source_model_calls": sum(row.total_target_calls for row in rows),
        "mean_linf": statistics.fmean(row.linf for row in rows),
        "median_linf": statistics.median(row.linf for row in rows),
        "mean_l2": statistics.fmean(row.l2 for row in rows),
        "median_l2": statistics.median(row.l2 for row in rows),
        "action_histogram": {
            str(action): count for action, count in sorted(action_counts.items())
        },
        "normalized_action_entropy": entropy,
    }


def verify_d1_raw_evidence(
    row_records: Sequence[Mapping[str, object]],
    trace_records: Sequence[Mapping[str, object]],
    *,
    expected_methods: Sequence[str],
) -> dict[str, object]:
    """Fail closed on source isolation, cohorts, calls, traces, and metrics."""

    methods = tuple(expected_methods)
    if not methods or len(set(methods)) != len(methods) or methods[0] != "score_greedy":
        raise ValueError("D1 expected methods are invalid")
    rows = tuple(_row(record) for record in row_records)
    validate_residual_source_records(
        row_records,
        heldout_family=D1_HELDOUT_FAMILY,
    )
    validate_residual_source_records(
        trace_records,
        heldout_family=D1_HELDOUT_FAMILY,
    )
    if {row.method for row in rows} != set(methods) or {
        row.victim_family for row in rows
    } != set(D1_SOURCE_FAMILIES):
        raise ValueError("D1 raw evidence has unexpected methods or families")
    row_keys = Counter(
        (row.method, row.victim_family, row.victim_id, row.sample_id) for row in rows
    )
    trace_keys = Counter(
        (
            trace.get("method"),
            trace.get("victim_family"),
            trace.get("victim_id"),
            trace.get("sample_id"),
        )
        for trace in trace_records
    )
    if row_keys != trace_keys or any(count != 1 for count in row_keys.values()):
        raise ValueError("D1 raw rows and full traces do not match exactly")
    for row in rows:
        if (
            row.query_budget != 50
            or not 1 <= row.total_target_calls <= 50
            or len(row.action_trace) != row.total_target_calls - 1
            or any(
                isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < 96
                for action in row.action_trace
            )
            or row.linf > 8 / 255 + 1e-6
            or row.linf < 0
            or row.l2 < 0
        ):
            raise ValueError("D1 raw row violates the locked attack contract")
    for trace in trace_records:
        query_trace = trace.get("query_trace")
        calls = trace.get("total_target_calls")
        if (
            not isinstance(query_trace, list)
            or isinstance(calls, bool)
            or not isinstance(calls, int)
            or len(query_trace) != calls
            or any(
                not isinstance(event, Mapping)
                or event.get("call_index") != offset
                or event.get("feedback") != "scores"
                or event.get("error") is not None
                or not isinstance(event.get("purpose"), str)
                or not event.get("purpose")
                or (offset == 1 and event.get("purpose") != "initialization")
                or (offset > 1 and event.get("purpose") == "initialization")
                or event.get("victim_id") != trace.get("victim_id")
                or event.get("sample_id") != trace.get("sample_id")
                for offset, event in enumerate(query_trace, start=1)
            )
        ):
            raise ValueError("D1 full query trace violates call accounting")

    by_family: dict[str, dict[str, object]] = {}
    method_cohorts: dict[tuple[str, str], set[tuple[str, str, bool]]] = {}
    for family in D1_SOURCE_FAMILIES:
        by_family[family] = {}
        for method in methods:
            selected = tuple(
                row
                for row in rows
                if row.victim_family == family and row.method == method
            )
            by_family[family][method] = _summary(selected)
            method_cohorts[(family, method)] = {
                (row.victim_id, row.sample_id, row.clean_correct) for row in selected
            }
        if (
            len({frozenset(method_cohorts[(family, method)]) for method in methods})
            != 1
        ):
            raise ValueError("D1 method cohorts are not exactly paired")
    macro = {
        method: {
            "final_asr": statistics.fmean(
                float(
                    by_family[family][method]["asr_at_budgets"]["50"]  # type: ignore[index]
                )
                for family in D1_SOURCE_FAMILIES
            ),
            "asr_query_auc": statistics.fmean(
                float(by_family[family][method]["asr_query_auc"])  # type: ignore[index]
                for family in D1_SOURCE_FAMILIES
            ),
            "normalized_action_entropy": statistics.fmean(
                float(
                    by_family[family][method]["normalized_action_entropy"]  # type: ignore[index]
                )
                for family in D1_SOURCE_FAMILIES
            ),
        }
        for method in methods
    }
    macro_curves = {
        method: {
            str(budget): statistics.fmean(
                float(
                    by_family[family][method]["asr_at_budgets"][str(budget)]  # type: ignore[index]
                )
                for family in D1_SOURCE_FAMILIES
            )
            for budget in range(51)
        }
        for method in methods
    }
    return {
        "verified": True,
        "methods": list(methods),
        "source_families": list(D1_SOURCE_FAMILIES),
        "rows": len(rows),
        "full_query_traces": len(trace_records),
        "hidden_target_calls": 0,
        "query_budget": 50,
        "epsilon": 8 / 255,
        "cohorts_exactly_paired": True,
        "by_family": by_family,
        "macro": macro,
        "macro_asr_at_budgets": macro_curves,
        "rows_sha256": hashlib.sha256(
            json.dumps(
                list(row_records),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "inference_scope": (
            "one fixed seed and fixed visible source victims; descriptive only"
        ),
    }


def verify_d1_recorded_summaries(
    verified: Mapping[str, object],
    conditions: Mapping[str, object],
) -> None:
    """Require recorded condition summaries to equal raw recomputation."""

    raw_by_family = verified.get("by_family")
    methods = verified.get("methods")
    if (
        verified.get("verified") is not True
        or not isinstance(raw_by_family, Mapping)
        or not isinstance(methods, list)
        or set(conditions) != set(D1_SOURCE_FAMILIES)
    ):
        raise ValueError("D1 recorded-summary verification input is invalid")
    for family in D1_SOURCE_FAMILIES:
        condition = conditions[family]
        raw_family = raw_by_family[family]
        if not isinstance(condition, Mapping) or not isinstance(
            raw_family,
            Mapping,
        ):
            raise ValueError("D1 family summaries must be objects")
        recorded_methods = condition.get("methods")
        if not isinstance(recorded_methods, Mapping) or set(recorded_methods) != set(
            methods
        ):
            raise ValueError("D1 recorded method summaries are incomplete")
        for method in methods:
            raw = raw_family[method]
            recorded = recorded_methods[method]
            if not isinstance(raw, Mapping) or not isinstance(recorded, Mapping):
                raise ValueError("D1 method summary is invalid")
            recorded_curve = recorded.get("asr_at_budgets")
            if not isinstance(recorded_curve, Mapping):
                raise ValueError("D1 recorded ASR curve is missing")
            normalized_curve = {
                str(key): float(value) for key, value in recorded_curve.items()
            }
            exact = (
                int(recorded.get("eligible", -1)) == int(raw["eligible"])
                and int(recorded.get("successes", -1)) == int(raw["successes"])
                and normalized_curve == raw["asr_at_budgets"]
                and math.isclose(
                    float(recorded.get("asr_query_auc", math.nan)),
                    float(raw["asr_query_auc"]),
                    abs_tol=1e-12,
                )
                and int(recorded.get("source_model_calls", -1))
                == int(raw["source_model_calls"])
                and math.isclose(
                    float(
                        recorded.get(
                            "normalized_action_entropy",
                            math.nan,
                        )
                    ),
                    float(raw["normalized_action_entropy"]),
                    abs_tol=1e-12,
                )
            )
            if not exact:
                raise ValueError("D1 recorded metrics do not match raw recomputation")


def _svg_line_plot(
    curves: Mapping[str, Mapping[str, object]],
) -> str:
    width, height = 900, 520
    left, right, top, bottom = 85, 30, 45, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_rate = max(
        0.01,
        max(float(value) for curve in curves.values() for value in curve.values()),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" '
        'font-family="sans-serif" font-size="20" font-weight="700">'
        "D1 source attack success by query budget</text>",
    ]
    for tick in range(6):
        value = max_rate * tick / 5
        y = top + plot_height * (1 - tick / 5)
        parts.extend(
            (
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
                f'y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{value:.1%}</text>',
            )
        )
    for tick in range(0, 51, 10):
        x = left + plot_width * tick / 50
        parts.append(
            f'<text x="{x:.1f}" y="{height - bottom + 25}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{tick}</text>'
        )
    for index, (method, curve) in enumerate(curves.items()):
        points = " ".join(
            f"{left + plot_width * budget / 50:.1f},"
            f"{top + plot_height * (1 - float(curve[str(budget)]) / max_rate):.1f}"
            for budget in range(51)
        )
        color = _COLORS[index % len(_COLORS)]
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{left + 180 * index}" y="{height - 18}" '
            f'font-family="sans-serif" font-size="13" fill="{color}">'
            f"{html.escape(method)}</text>"
        )
    parts.extend(
        (
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 42}" '
            'text-anchor="middle" font-family="sans-serif" font-size="14">'
            "Total source-model calls</text>",
            f'<text x="20" y="{top + plot_height / 2:.1f}" '
            'transform="rotate(-90 20 245)" text-anchor="middle" '
            'font-family="sans-serif" font-size="14">Attack success rate</text>',
            "</svg>",
        )
    )
    return "\n".join(parts)


def _svg_bar_plot(macro: Mapping[str, Mapping[str, object]]) -> str:
    width, height = 800, 500
    values = tuple(
        (method, float(metrics["final_asr"])) for method, metrics in macro.items()
    )
    max_rate = max(0.01, max(value for _, value in values))
    bar_width = 120
    gap = 55
    start = (width - len(values) * bar_width - (len(values) - 1) * gap) / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="400" y="32" text-anchor="middle" font-family="sans-serif" '
        'font-size="20" font-weight="700">D1 final source ASR at 50 calls</text>',
        '<line x1="70" y1="420" x2="760" y2="420" stroke="#111827"/>',
    ]
    for index, (method, value) in enumerate(values):
        x = start + index * (bar_width + gap)
        bar_height = 330 * value / max_rate
        y = 420 - bar_height
        color = _COLORS[index % len(_COLORS)]
        parts.extend(
            (
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" '
                f'height="{bar_height:.1f}" fill="{color}"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" '
                'text-anchor="middle" font-family="sans-serif" font-size="14">'
                f"{value:.2%}</text>",
                f'<text x="{x + bar_width / 2:.1f}" y="445" text-anchor="middle" '
                'font-family="sans-serif" font-size="12">'
                f"{html.escape(method)}</text>",
            )
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_d1_evidence_plots(
    output_dir: Path,
    verified: Mapping[str, object],
) -> tuple[Path, Path]:
    """Write deterministic query-efficiency and final-ASR SVG figures."""

    if verified.get("verified") is not True:
        raise ValueError("plots require independently verified D1 evidence")
    curves = verified.get("macro_asr_at_budgets")
    macro = verified.get("macro")
    if not isinstance(curves, Mapping) or not isinstance(macro, Mapping):
        raise ValueError("verified D1 plot metrics are missing")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    curve_path = destination / "asr_by_query.svg"
    final_path = destination / "final_asr.svg"
    _atomic_text_write(curve_path, _svg_line_plot(curves))  # type: ignore[arg-type]
    _atomic_text_write(final_path, _svg_bar_plot(macro))  # type: ignore[arg-type]
    for path in (curve_path, final_path):
        _atomic_text_write(
            path.with_suffix(path.suffix + ".sha256"),
            sha256_file(path) + "\n",
        )
    return curve_path, final_path


__all__ = (
    "load_verified_jsonl_records",
    "verify_d1_raw_evidence",
    "verify_d1_recorded_summaries",
    "write_d1_evidence_plots",
)
