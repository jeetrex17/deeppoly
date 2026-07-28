"""Portable documentation and SVG rendering for Phase 2 evidence."""

from __future__ import annotations

from html import escape
from typing import Callable, Mapping, Sequence


_NAVY = "#17324d"
_BLUE = "#2474b5"
_ORANGE = "#e67e22"
_GREEN = "#2f855a"
_RED = "#c53030"
_GRAY = "#718096"
_GRID = "#d8e0e8"
_BACKGROUND = "#fbfcfe"


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("chart value must be numeric")
    return float(value)


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _seconds(value: float) -> str:
    return f"{value:.1f}s"


def _horizontal_grouped_chart(
    *,
    title: str,
    subtitle: str,
    rows: Sequence[Mapping[str, object]],
    label_key: str,
    series: Sequence[tuple[str, str, str]],
    formatter: Callable[[float], str],
    axis_formatter: Callable[[float], str],
) -> str:
    if not rows or not series:
        raise ValueError("chart requires rows and series")
    values = [
        _number(row[key])
        for row in rows
        for key, _label, _color in series
    ]
    lower = min(0.0, min(values))
    upper = max(0.0, max(values))
    if upper == lower:
        upper = lower + 1.0
    padding = (upper - lower) * 0.08
    lower -= padding
    upper += padding
    width = 1120
    label_width = 360
    plot_width = 660
    top = 104
    row_height = max(54, 25 * len(series) + 18)
    height = top + row_height * len(rows) + 82

    def x_position(value: float) -> float:
        return label_width + (value - lower) * plot_width / (
            upper - lower
        )

    zero_x = x_position(0.0)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title description">'
        ),
        "<title id=\"title\">" + escape(title) + "</title>",
        (
            "<desc id=\"description\">"
            + escape(subtitle)
            + "</desc>"
        ),
        (
            "<style>"
            "text{font-family:Inter,system-ui,sans-serif;fill:#17324d}"
            ".title{font-size:24px;font-weight:700}"
            ".subtitle{font-size:13px;fill:#526579}"
            ".label{font-size:13px;font-weight:600}"
            ".value{font-size:12px;font-weight:600}"
            ".tick{font-size:11px;fill:#60758a}"
            ".legend{font-size:12px}"
            "</style>"
        ),
        (
            f'<rect width="{width}" height="{height}" '
            f'fill="{_BACKGROUND}"/>'
        ),
        (
            f'<text class="title" x="32" y="38">'
            f"{escape(title)}</text>"
        ),
        (
            f'<text class="subtitle" x="32" y="62">'
            f"{escape(subtitle)}</text>"
        ),
    ]
    legend_x = 32
    for _key, label, color in series:
        parts.extend(
            [
                (
                    f'<rect x="{legend_x}" y="76" width="12" '
                    f'height="12" rx="2" fill="{color}"/>'
                ),
                (
                    f'<text class="legend" x="{legend_x + 18}" '
                    f'y="87">{escape(label)}</text>'
                ),
            ]
        )
        legend_x += 24 + len(label) * 7
    for tick_index in range(5):
        value = lower + (upper - lower) * tick_index / 4
        x = x_position(value)
        parts.extend(
            [
                (
                    f'<line x1="{x:.2f}" y1="{top - 8}" '
                    f'x2="{x:.2f}" y2="{height - 48}" '
                    f'stroke="{_GRID}" stroke-width="1"/>'
                ),
                (
                    f'<text class="tick" x="{x:.2f}" '
                    f'y="{height - 26}" text-anchor="middle">'
                    f"{escape(axis_formatter(value))}</text>"
                ),
            ]
        )
    parts.append(
        (
            f'<line x1="{zero_x:.2f}" y1="{top - 8}" '
            f'x2="{zero_x:.2f}" y2="{height - 48}" '
            f'stroke="{_NAVY}" stroke-width="1.5"/>'
        )
    )
    bar_height = 17
    for row_index, row in enumerate(rows):
        row_top = top + row_index * row_height
        parts.append(
            (
                f'<text class="label" x="{label_width - 14}" '
                f'y="{row_top + row_height / 2:.2f}" '
                f'text-anchor="end" dominant-baseline="middle">'
                f"{escape(str(row[label_key]))}</text>"
            )
        )
        for series_index, (key, _label, color) in enumerate(series):
            value = _number(row[key])
            value_x = x_position(value)
            left = min(zero_x, value_x)
            bar_width = max(1.0, abs(value_x - zero_x))
            y = (
                row_top
                + 7
                + series_index * (bar_height + 5)
            )
            text_anchor = "start" if value >= 0 else "end"
            text_x = (
                max(zero_x, value_x) + 6
                if value >= 0
                else min(zero_x, value_x) - 6
            )
            parts.extend(
                [
                    (
                        f'<rect x="{left:.2f}" y="{y:.2f}" '
                        f'width="{bar_width:.2f}" '
                        f'height="{bar_height}" rx="3" '
                        f'fill="{color}"/>'
                    ),
                    (
                        f'<text class="value" x="{text_x:.2f}" '
                        f'y="{y + 12:.2f}" '
                        f'text-anchor="{text_anchor}">'
                        f"{escape(formatter(value))}</text>"
                    ),
                ]
            )
    parts.append("</svg>\n")
    return "".join(parts)


def source_asr_figure(
    method_rows: Sequence[Mapping[str, object]],
) -> str:
    return _horizontal_grouped_chart(
        title="Phase 2 source attack success by method",
        subtitle=(
            "Macro mean across three leave-one-family-out folds, "
            "two source slices, and both visible source families."
        ),
        rows=method_rows,
        label_key="method_label",
        series=(("asr", "Attack success rate", _BLUE),),
        formatter=_percent,
        axis_formatter=_percent,
    )


def gain_figure(
    fold_rows: Sequence[Mapping[str, object]],
) -> str:
    return _horizontal_grouped_chart(
        title="Learned policy gain versus score greedy",
        subtitle=(
            "Negative values favor score greedy. Each fold is the macro "
            "mean over four matched source conditions."
        ),
        rows=fold_rows,
        label_key="omitted_target_family",
        series=(
            ("asr_gain", "ASR gain", _BLUE),
            ("auc_gain", "AUC gain", _ORANGE),
        ),
        formatter=_percent,
        axis_formatter=_percent,
    )


def bc_figure(
    bc_rows: Sequence[Mapping[str, object]],
) -> str:
    return _horizontal_grouped_chart(
        title="Soft behavior-cloning diagnostics",
        subtitle=(
            "Gain over the evaluated-label validation oracle. "
            "Positive values are required by the screen."
        ),
        rows=bc_rows,
        label_key="omitted_target_family",
        series=(
            ("top5_gain", "Top-5 accuracy gain", _GREEN),
            (
                "soft_ce_improvement",
                "Soft cross-entropy improvement",
                _ORANGE,
            ),
        ),
        formatter=lambda value: f"{value:+.4f}",
        axis_formatter=lambda value: f"{value:+.3f}",
    )


def runtime_figure(
    fold_rows: Sequence[Mapping[str, object]],
) -> str:
    return _horizontal_grouped_chart(
        title="Recorded Phase 2 compute by fold",
        subtitle=(
            "Component timers from the verified manifests. Resume-only "
            "orchestration time is excluded."
        ),
        rows=fold_rows,
        label_key="omitted_target_family",
        series=(
            ("bc_seconds", "Soft BC", _BLUE),
            ("ppo_seconds", "PPO", _ORANGE),
            ("source_evaluation_seconds", "Source evaluation", _GRAY),
        ),
        formatter=_seconds,
        axis_formatter=lambda value: f"{value:.0f}s",
    )


def evidence_readme(summary: Mapping[str, object]) -> str:
    promotion = summary["promotion"]
    metrics = promotion["metrics"]
    runtime = summary["runtime"]
    outcome = "PASSED" if promotion["passed"] else "FAILED"
    return f"""# CIFAR-10 RTX Phase 2 Stage B evidence

## Result

The preregistered source-only Stage B screen **{outcome}**. All three leave-one-family-out cells and all 12 matched source conditions completed. The hidden target families remained sealed during each fold, and the total recorded target attack-call count is **0**.

This is a complete negative development result for the tested action-conditioned soft-BC plus PPO candidate. It is not evidence that adversarial attacks transfer across hidden victim families, and it does not authorize target evaluation.

| Locked Stage B measure | Observed | Required |
|---|---:|---:|
| Mean ASR gain over score greedy | {metrics["mean_asr_gain_over_score_greedy"]:+.4f} | at least +0.0100 |
| Mean ASR-query AUC gain | {metrics["mean_auc_gain_over_score_greedy"]:+.4f} | at least +0.0050 |
| Mean soft top-5 gain over validation oracle | {metrics["mean_bc_top5_or_hard_accuracy_gain"]:+.4f} | at least +0.0100 |
| Mean soft cross-entropy improvement | {metrics["mean_bc_loss_improvement_over_validation_oracle"]:+.4f} | at least +0.0200 |
| Conditions positive on both ASR and AUC | {metrics["positive_asr_and_auc_condition_fraction"]:.4f} | at least 0.6700 |
| Strict source gates passed | {promotion["strict_publication_source_gate_passes"]} of 3 | diagnostic only |

The learned policy produced nonzero attacks, but it did not outperform the matched score-greedy control reliably. The soft behavior-cloning representation also failed its validation-oracle tests. Stage C and hidden-target evaluation were therefore not run.

## Figures

![Source ASR by method](source_asr_by_method.svg)

![Gain versus score greedy](gain_vs_score_greedy.svg)

![Behavior-cloning diagnostics](bc_diagnostics.svg)

![Recorded runtime](runtime.svg)

## Evidence contents

| Artifact | Purpose |
|---|---|
| `summary.json` | Machine-readable outcome, aggregate metrics, runtime, and integrity facts |
| `condition_metrics.csv` | Every method result for all 12 source conditions |
| `fold_summary.csv` | Learned and control results grouped by omitted family |
| `bc_diagnostics.csv` | Soft-BC validation results and oracle comparisons |
| `training_blocks.csv` | PPO block diagnostics and recorded source-query use |
| `victim_accuracy.csv` | Clean validation accuracy for every loaded source victim |
| `raw_source_records.tar.gz` | Exact source result rows and sampled query traces |
| `raw_compact_evidence.json.gz` | Portable run manifests, full source metrics, and training diagnostics |
| `input_checksums.csv` | Hashes binding the export to the full local archive |
| `attempt_log_checksums.csv` | Hash and size of each locally retained execution log |
| `SHA256SUMS` | Hashes for every file in this directory |

The raw numerical rows and traces are included. Binary victim and policy checkpoints are not committed to Git. Their SHA-256 identities are preserved in the compact evidence and checksum tables, while the verified full archive remains under `output/rl_transfer/cifar10_rtx_phase2_screen` on the research Mac.

## Runtime and scope

The recorded training and evaluation components total {runtime["recorded_component_minutes"]:.1f} minutes across the three folds on an NVIDIA GeForce RTX 2080 Ti. This short screen used one development policy seed, 600 soft-BC episodes, 600 scheduled PPO episodes, and 100 source-evaluation images per cell.

The result supports a narrow conclusion: under this CIFAR-10 victim bank, action space, query budget, and training budget, the tested candidate did not establish source competence beyond score greedy. A new method must be developed and screened entirely on source data before any confirmatory or target access.
"""


def provenance(summary: Mapping[str, object]) -> str:
    integrity = summary["integrity"]
    environment = summary["environment"]
    return f"""# Provenance

This directory was generated deterministically from the checksum-verified Phase 2 Stage B archive copied from the RTX workstation.

- Screen manifest SHA-256: `{integrity["screen_manifest_sha256"]}`
- Study code digest: `{integrity["study_code_digest"]}`
- Protocol SHA-256: `{integrity["protocol_sha256"]}`
- Verified source runs: {integrity["verified_runs"]}
- Verified sidecars: {integrity["verified_sidecars"]}
- Verified raw result files: {integrity["verified_results_files"]}
- Verified query-trace files: {integrity["verified_trace_files"]}
- Target attack calls: 0
- GPU: {environment.get("cuda_device_name", "recorded in run manifests")}
- CUDA runtime: {environment.get("cuda_runtime", "unknown")}
- PyTorch: {environment.get("torch", "unknown")}
- Git revision used for training: `{environment.get("git_revision", "unknown")}`

Every run was source-only. The family named as the target of a fold was neither constructed nor clean-validated in that fold. The `victim_access_audit` records are included in the compact evidence.

The full local archive also contains the binary checkpoints. This Git bundle excludes those binaries and execution-log contents. It retains checkpoint hashes, exact raw numerical result rows, sampled query traces, dependency pins, and log hashes so the published evidence remains portable and reviewable.
"""
