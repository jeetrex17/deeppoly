"""Portable prose and SVG rendering for calibration evidence."""

from __future__ import annotations

from typing import Mapping, Sequence

from .phase1_export_validation import (
    finite_number,
    require_mapping,
    require_sequence,
)
from .phase2_calibration_manifest import CALIBRATION_TEMPERATURES
from .phase2_export_render import _horizontal_grouped_chart


def mean_gain_figure(
    temperature_rows: Sequence[Mapping[str, object]],
) -> str:
    return _horizontal_grouped_chart(
        title="Frozen policy gain versus score greedy by temperature",
        subtitle=(
            "Macro mean across three held-out-family folds. "
            "Positive values favor the frozen learned policy."
        ),
        rows=temperature_rows,
        label_key="temperature_label",
        series=(
            ("mean_asr_gain", "Mean ASR gain", "#2474b5"),
            ("mean_auc_gain", "Mean AUC gain", "#e67e22"),
        ),
        formatter=lambda value: f"{100 * value:+.2f} pp",
        axis_formatter=lambda value: f"{100 * value:+.2f}",
    )


def fold_asr_figure(
    folds: Sequence[Mapping[str, object]],
) -> str:
    chart_rows = []
    for fold in folds:
        temperatures = require_mapping(
            fold.get("temperatures"),
            "fold temperatures",
        )
        chart_rows.append(
            {
                "heldout_family": fold["heldout_family"],
                "score": finite_number(
                    require_mapping(
                        fold.get("score_greedy"),
                        "score metrics",
                    ).get("asr"),
                    "score ASR",
                ),
                **{
                    f"t_{temperature:g}": finite_number(
                        require_mapping(
                            temperatures.get(str(temperature)),
                            f"temperature {temperature}",
                        ).get("asr"),
                        "temperature ASR",
                    )
                    for temperature in CALIBRATION_TEMPERATURES
                },
            }
        )
    colors = (
        "#17324d",
        "#9f7aea",
        "#2474b5",
        "#2f855a",
        "#e67e22",
        "#c53030",
    )
    series = (
        ("score", "Score greedy", colors[0]),
        *tuple(
            (
                f"t_{temperature:g}",
                f"T={temperature:g}",
                colors[index + 1],
            )
            for index, temperature in enumerate(CALIBRATION_TEMPERATURES)
        ),
    )
    return _horizontal_grouped_chart(
        title="Source attack success by omitted family and temperature",
        subtitle=("Each value is a macro mean over four matched source conditions."),
        rows=chart_rows,
        label_key="heldout_family",
        series=series,
        formatter=lambda value: f"{100 * value:.2f}%",
        axis_formatter=lambda value: f"{100 * value:.1f}%",
    )


def evidence_readme(summary: Mapping[str, object]) -> str:
    decision = require_mapping(summary["decision"], "summary decision")
    runtime = require_mapping(summary["runtime"], "summary runtime")
    integrity = require_mapping(summary["integrity"], "summary integrity")
    score = require_mapping(summary["score_greedy"], "summary score greedy")
    temperature_rows = [
        require_mapping(item, "summary temperature")
        for item in require_sequence(
            summary["temperature_summary"],
            "summary temperatures",
        )
    ]
    temperature_table = "\n".join(
        "| "
        f"{float(row['temperature']):g} | "
        f"{100 * float(row['macro_asr']):.3f}% | "
        f"{100 * float(row['mean_asr_gain']):+.3f} | "
        f"{100 * float(row['macro_auc']):.3f}% | "
        f"{100 * float(row['mean_auc_gain']):+.3f} | "
        f"{row['folds_observed_nonnegative_both']} / 3 | "
        f"{row['conditions_observed_nonnegative_both']} / 12 |"
        for row in temperature_rows
    )
    limitations = "\n".join(
        f"- {item}"
        for item in require_sequence(
            summary["limitations"],
            "summary limitations",
        )
    )
    return f"""# CIFAR-10 RTX Phase 2 calibration evidence

## Result

The bounded source-only diagnostic completed all three leave-one-family-out folds in {float(runtime["elapsed_minutes"]):.2f} minutes. It evaluated the frozen Phase 2 checkpoints at temperatures 0.25, 0.50, 0.75, 1.00, and 1.50. Training remained disabled and the hidden-target attack-call count remained **0**.

No temperature matched score-greedy ASR and ASR-query AUC in every fold. The predeclared diagnostic rule therefore stops this five-value global-temperature repair for these frozen seed-17 checkpoints. The diagnostic does not identify the underlying cause. Residual action ranking is an exploratory next candidate, not an established explanation.

| Diagnostic measure | Result |
|---|---:|
| Completed folds | 3 of 3 |
| Temperature-1.0 exact replays | 3 of 3 |
| Raw result rows | {integrity["result_rows"]} |
| Sampled query traces | {integrity["query_traces"]} |
| Source-model calls | {runtime["source_model_calls"]} |
| Hidden-target calls | 0 |
| Qualifying temperatures | 0 |
| Best mean ASR temperature | {decision["best_mean_asr_temperature"]} ({100 * float(decision["best_mean_asr_gain"]):+.2f} points) |
| Best mean AUC temperature | {decision["best_mean_auc_temperature"]} ({100 * float(decision["best_mean_auc_gain"]):+.2f} points) |

Score greedy reached {100 * float(score["macro_asr"]):.3f}% macro ASR and {100 * float(score["macro_auc"]):.3f}% macro AUC.

| Temperature | Macro ASR | ASR gap, points | Macro AUC | AUC gap, points | Folds with observed gaps >= 0 on both | Conditions with observed gaps >= 0 on both |
|---:|---:|---:|---:|---:|---:|---:|
{temperature_table}

Temperature 0.75 improved both metrics in the transformer-held-out fold, and temperature 1.50 improved ASR in the classical-CNN-held-out fold. Neither effect was consistent across all three folds and both metrics. Hidden target victims remained sealed, so this is not a transferability result.

## Statistical scope and limitations

{limitations}

## Figures

![Mean gain by temperature](mean_gain_by_temperature.svg)

![Fold ASR by temperature](fold_asr_by_temperature.svg)

## Evidence contents

| Artifact | Purpose |
|---|---|
| `summary.json` | Decision, aggregate metrics, integrity facts, runtime, and limitations |
| `temperature_summary.csv` | Mean and worst-fold gains for every temperature |
| `fold_metrics.csv` | Score-greedy and frozen-policy metrics for every fold |
| `condition_metrics.csv` | All 72 matched source-condition method summaries |
| `raw_calibration_records.tar.gz` | Exact verified manifest, 9,000 rows, 60 traces, and sidecars |
| `input_checksums.csv` | Hashes for the copied full-resolution inputs |
| `attempt_log_checksums.csv` | Hashes and sizes for retained execution logs |
| `PROVENANCE.md` | Hardware, code identity, and scope |
| `SHA256SUMS` | Hashes for every published file |

## Next decision

Do not run another grid under this D0 protocol. The next exploratory candidate should preserve score greedy as a fallback and learn only a residual action ranker on visible source victims. A short one-fold D1 screen should be completed before any longer replication is authorized.
"""


def provenance(summary: Mapping[str, object]) -> str:
    integrity = require_mapping(summary["integrity"], "summary integrity")
    runtime = require_mapping(summary["runtime"], "summary runtime")
    return f"""# Provenance

This directory was generated from the checksum-verified Phase 2 calibration archive copied from the RTX workstation.

- Calibration manifest SHA-256: `{integrity["manifest_sha256"]}`
- Raw result SHA-256: `{integrity["results_sha256"]}`
- Query-trace SHA-256: `{integrity["query_traces_sha256"]}`
- Source Phase 2 manifest SHA-256: `{integrity["source_manifest_sha256"]}`
- Calibration code digest: `{integrity["calibration_code_digest"]}`
- Git revision: `{integrity["calibration_git_revision"]}`
- GPU: {runtime["gpu_name"]}
- Recorded runtime: {float(runtime["elapsed_seconds"]):.3f} seconds
- Source-model calls: {runtime["source_model_calls"]}
- Training performed: no
- Hidden-target calls: 0

All three temperature-1.0 replays reproduced the verified Phase 2 aggregate metrics and exact per-sample source rows. Every raw calibration row passed the source-only audit. The result is exploratory and does not authorize hidden-target evaluation.
"""


__all__ = (
    "evidence_readme",
    "fold_asr_figure",
    "mean_gain_figure",
    "provenance",
)
