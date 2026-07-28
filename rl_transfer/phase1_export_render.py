"""Portable Markdown and SVG rendering for Phase 1 evidence."""

from __future__ import annotations

import html
import math
from typing import Mapping, Sequence


LEARNED_METHOD = "gradient_bc_groupdro_ppo_stochastic"
METHOD_LABELS = {
    "gradient_bc_groupdro_ppo_stochastic": "Hybrid BC + GroupDRO + PPO",
    "gradient_bc_only_stochastic": "BC only",
    "ppo_only_stochastic": "PPO only",
    "score_greedy": "Score greedy",
    "random_action": "Random",
    "bandit_action": "Bandit",
}
METHOD_ORDER = tuple(METHOD_LABELS)
MANAGED_OUTPUT_NAMES = frozenset(
    {
        "README.md",
        "PROVENANCE.md",
        "SHA256SUMS",
        "summary.json",
        "dependency_freeze.txt",
        "environment_summary.json",
        "run_summary.csv",
        "condition_metrics.csv",
        "method_summary.csv",
        "input_checksums.csv",
        "raw_compact_evidence.json.gz",
        "raw_source_records.tar.gz",
        "method_performance.svg",
        "heldout_family_asr.svg",
        "bc_diagnostics.svg",
        "runtime.svg",
    }
)


def _format_float(value: float) -> str:
    return f"{value:.8f}"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _mean(records: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records)


def formatted_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            key: _format_float(value) if isinstance(value, float) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def formatted_condition_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            **row,
            "asr": _format_float(float(row["asr"])),
            "asr_query_auc": _format_float(float(row["asr_query_auc"])),
            "normalized_action_entropy": _format_float(
                float(row["normalized_action_entropy"])
            ),
        }
        for row in rows
    ]


def _svg_chart(
    *,
    title: str,
    subtitle: str,
    groups: Sequence[tuple[str, Sequence[tuple[str, float, str]]]],
    x_max: float,
    x_label: str,
) -> str:
    width = 1080
    left = 310
    right = 80
    top = 125
    row_height = 34
    group_gap = 24
    plot_width = width - left - right
    bar_count = sum(len(series) for _, series in groups)
    height = top + bar_count * row_height + len(groups) * group_gap + 80
    escaped_title = html.escape(title)
    escaped_subtitle = html.escape(subtitle)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img">'
        ),
        f"<title>{escaped_title}</title>",
        "<style>"
        "text{font-family:Inter,Arial,sans-serif;fill:#172033}"
        ".title{font-size:28px;font-weight:700}"
        ".subtitle{font-size:15px;fill:#566176}"
        ".label{font-size:14px}"
        ".value{font-size:13px;font-weight:600}"
        ".group{font-size:16px;font-weight:700}"
        ".axis{font-size:12px;fill:#68748a}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="40" y="45">{escaped_title}</text>',
        (f'<text class="subtitle" x="40" y="74">{escaped_subtitle}</text>'),
    ]
    for tick in range(6):
        fraction = tick / 5
        x = left + plot_width * fraction
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" '
            f'y2="{height - 50}" stroke="#e4e8ef"/>'
        )
        parts.append(
            f'<text class="axis" x="{x:.1f}" y="{height - 30}" '
            f'text-anchor="middle">{x_max * fraction:.3f}</text>'
        )
    y = top
    for group_name, series in groups:
        parts.append(
            f'<text class="group" x="40" y="{y + 16}">{html.escape(group_name)}</text>'
        )
        y += group_gap
        for label, value, color in series:
            bounded = max(0.0, min(float(value), x_max))
            bar_width = plot_width * bounded / x_max if x_max else 0
            parts.extend(
                [
                    (
                        f'<text class="label" x="{left - 12}" y="{y + 16}" '
                        f'text-anchor="end">{html.escape(label)}</text>'
                    ),
                    (
                        f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" '
                        f'height="22" rx="4" fill="{color}"/>'
                    ),
                    (
                        f'<text class="value" x="{left + bar_width + 8:.2f}" '
                        f'y="{y + 16}">{value:.4f}</text>'
                    ),
                ]
            )
            y += row_height
        y += group_gap
    parts.append(
        f'<text class="axis" x="{left + plot_width / 2:.1f}" '
        f'y="{height - 8}" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append("</svg>\n")
    return "".join(parts)


def method_figure(methods: Sequence[Mapping[str, object]]) -> str:
    exact = {
        str(record["method"]): record
        for record in methods
        if record["source_slice"] == "combined_source"
    }
    groups = []
    for method in METHOD_ORDER:
        if method not in exact:
            continue
        record = exact[method]
        groups.append(
            (
                METHOD_LABELS[method],
                (
                    ("ASR at 50 queries", float(record["asr"]), "#2563eb"),
                    (
                        "Query-normalized AUC",
                        float(record["asr_query_auc"]),
                        "#0f9f7f",
                    ),
                ),
            )
        )
    maximum = max(
        (
            float(record[key])
            for record in exact.values()
            for key in ("asr", "asr_query_auc")
        ),
        default=0.1,
    )
    return _svg_chart(
        title="Phase 1 source-victim performance",
        subtitle=(
            "Condition-macro metrics across exact and new-instance source slices"
        ),
        groups=groups,
        x_max=max(0.05, math.ceil(maximum * 100) / 100 + 0.01),
        x_label="Rate",
    )


def family_figure(
    families: Sequence[Mapping[str, object]],
) -> str:
    groups = [
        (
            str(record["omitted_target_family"]).replace("_", " ").title(),
            (
                ("Hybrid source ASR", float(record["learned_asr"]), "#2563eb"),
                (
                    "Score-greedy source ASR",
                    float(record["control_asr"]),
                    "#94a3b8",
                ),
            ),
        )
        for record in families
    ]
    maximum = max(
        (
            max(float(record["learned_asr"]), float(record["control_asr"]))
            for record in families
        ),
        default=0.1,
    )
    return _svg_chart(
        title="Source ASR by LOFO split",
        subtitle=(
            "Source attack slices only; held-out attack cohort remained unopened"
        ),
        groups=groups,
        x_max=max(0.05, math.ceil(maximum * 100) / 100 + 0.01),
        x_label="Attack success rate at 50 queries",
    )


def bc_figure(bc: Mapping[str, object]) -> str:
    groups = [
        (
            "Mean across policy runs",
            (
                ("Training accuracy", float(bc["train_accuracy"]), "#2563eb"),
                (
                    "Validation accuracy",
                    float(bc["validation_accuracy"]),
                    "#0f9f7f",
                ),
                (
                    "Validation majority",
                    float(bc["validation_majority_accuracy"]),
                    "#f59e0b",
                ),
                ("Uniform chance", float(bc["uniform_accuracy"]), "#94a3b8"),
            ),
        )
    ]
    maximum = max(value for _, value, _ in groups[0][1])
    return _svg_chart(
        title="Behavior-cloning competence diagnostics",
        subtitle=("Exact 96-action prediction remained near the majority baseline"),
        groups=groups,
        x_max=max(0.05, math.ceil(maximum * 100) / 100 + 0.01),
        x_label="Action prediction accuracy",
    )


def runtime_figure(run_rows: Sequence[Mapping[str, object]]) -> str:
    families = sorted({str(row["omitted_target_family"]) for row in run_rows})
    groups = []
    for family in families:
        selected = [row for row in run_rows if row["omitted_target_family"] == family]
        groups.append(
            (
                family.replace("_", " ").title(),
                (
                    (
                        "Mean total run time",
                        _mean(selected, "elapsed_seconds") / 3600,
                        "#7c3aed",
                    ),
                    (
                        "Mean source evaluation",
                        _mean(
                            selected,
                            "source_evaluation_elapsed_seconds",
                        )
                        / 3600,
                        "#c4b5fd",
                    ),
                ),
            )
        )
    maximum = max(
        (value for _, series in groups for _, value, _ in series),
        default=1.0,
    )
    return _svg_chart(
        title="Phase 1 run time",
        subtitle="Wall-clock time measured by the study runner",
        groups=groups,
        x_max=max(0.5, math.ceil(maximum * 2) / 2 + 0.25),
        x_label="Hours per policy run",
    )


def readme(summary: Mapping[str, object]) -> str:
    outcomes = _mapping(summary["outcomes"], "outcomes")
    combined = _mapping(outcomes["combined_source"], "combined source")
    learned = _mapping(combined[LEARNED_METHOD], "learned summary")
    control_name = str(summary["protocol"]["primary_control"])
    control = _mapping(combined[control_name], "control summary")
    bc = _mapping(summary["behavior_cloning"], "BC summary")
    runtime = _mapping(summary["runtime"], "runtime summary")
    integrity = _mapping(summary["integrity"], "integrity")
    environment = _mapping(
        summary["environment_evidence"],
        "environment evidence",
    )
    dependencies = _mapping(environment["dependencies"], "dependencies")
    run_count = int(summary["protocol"]["policy_runs"])
    return f"""# CIFAR-10 RTX Phase 1 evidence

This directory is the compact, checksum-verifiable evidence bundle for the
source-attack phase of the preregistered cross-victim RL attack study.

## Outcome

The {run_count}-run source grid completed, but the strict source-competence gate
did not pass. The correct study status is `source_learning_failed`. Victim
models from every family, including each fold's held-out family, were fit or
loaded and clean-accuracy validated. The hidden target attack cohort remained
unopened, and held-out attack-evaluation calls were 0.

Across exact and new-instance source-victim evaluations, the hybrid BC +
GroupDRO + PPO policy
reached ASR `{float(learned["asr"]):.4f}` and query-normalized AUC
`{float(learned["asr_query_auc"]):.4f}`. Score greedy reached ASR
`{float(control["asr"]):.4f}` and AUC
`{float(control["asr_query_auc"]):.4f}`. The observed gains were
`{float(learned["asr"]) - float(control["asr"]):.4f}` ASR and
`{float(learned["asr_query_auc"]) - float(control["asr_query_auc"]):.4f}`
AUC. These gains are positive, but below the preregistered per-condition gate.

Behavior cloning also failed its strict competence gate in all
`{int(bc["run_count"])}` runs. Mean validation action accuracy was
`{float(bc["validation_accuracy"]):.4f}`, compared with majority accuracy
`{float(bc["validation_majority_accuracy"]):.4f}` and uniform chance
`{float(bc["uniform_accuracy"]):.4f}`. This supports a representation or
teacher-label learnability problem, not a claim of successful cross-family
transfer.

## Scope and interpretation

- This is a valid negative source-phase result.
- It is not a target-transfer attack result.
- No target ASR, transfer rate, or publication claim should be inferred.
- Phase 2 should improve source learnability and pass a time-bounded source
  screen before any hidden target attack evaluation.

The complete local archive, including checkpoints, remains outside Git. This
bundle includes all checksum-verified source result rows and query traces in a
normalized compressed archive. It excludes model files, machine-specific
paths, unsanitized machine records, and training logs. A checksum-verified,
sanitized dependency freeze is included.

## Files

- `summary.json`: machine-readable study outcome and aggregate statistics.
- `environment_summary.json`: run-start environment fields, dependency hashes,
  code mapping, and the separately scoped post-run GPU audit.
- `dependency_freeze.txt`: sanitized
  `{int(dependencies["package_count"])}`-package transitive dependency freeze.
- `run_summary.csv`: one row per LOFO policy run.
- `condition_metrics.csv`: every source slice, family, method, and run metric.
- `method_summary.csv`: condition-macro and eligible-pooled aggregates.
- `input_checksums.csv`: checksums for verified source artifacts.
- `raw_compact_evidence.json.gz`: aggregate per-victim curves, audits, gates,
  and selected training diagnostics without absolute paths.
- `raw_source_records.tar.gz`: all 30 source result files and all 30 source
  query-trace files under portable run-ID paths.
- `*.svg`: publication-ready vector summaries.
- `PROVENANCE.md`: verification and interpretation record.
- `SHA256SUMS`: hashes for every other file in this directory.

## Reproducibility

From the repository root, after installing the project environment:

```bash
python scripts/export_phase1_evidence.py
```

The source study manifest SHA-256 is
`{integrity["study_manifest_sha256"]}`. All
`{integrity["verified_runs"]}` run manifests, source-evaluation caches, result
rows, and trace files were checksum-verified before export.

Total measured source-phase wall time was
`{float(runtime["source_phase_hours"]):.2f}` hours. Rerunning the exporter with
unchanged inputs produces byte-identical files, including the gzip archive.
"""


def provenance(summary: Mapping[str, object]) -> str:
    integrity = _mapping(summary["integrity"], "integrity")
    environment = _mapping(
        summary["environment_evidence"],
        "environment evidence",
    )
    run_start = _mapping(
        environment["run_start_manifest"],
        "run-start environment",
    )
    dependencies = _mapping(environment["dependencies"], "dependencies")
    code_mapping = _mapping(environment["code_mapping"], "code mapping")
    gpu_audit = _mapping(
        environment["post_run_workstation_audit"],
        "GPU audit",
    )
    return f"""# Provenance and integrity

## Source record

- Study name: `{summary["study_name"]}`
- Study schema: `{summary["study_schema_version"]}`
- Study manifest SHA-256: `{integrity["study_manifest_sha256"]}`
- Verified source runs: `{integrity["verified_runs"]}`
- Verified raw result files: `{integrity["verified_results_files"]}`
- Verified raw trace files: `{integrity["verified_trace_files"]}`
- Study code digest: `{integrity["study_code_digest"]}`
- CUDA runtime recorded at run start: `{run_start["cuda_runtime"]}`
- cuDNN version recorded at run start: `{run_start["cudnn_version"]}`
- NVIDIA driver recorded at run start: `{run_start["nvidia_driver"]}`
- GPU model in the run-start study field: `unknown`
- Git revision in the run-start study field: `unknown`
- PyTorch from the verified freeze: `{dependencies["torch"]}`
- torchvision from the verified freeze: `{dependencies["torchvision"]}`
- Editable-install code mapping: `{code_mapping["editable_install_commit"]}`

The exporter verified each JSON SHA-256 sidecar, reconstructed each run
directory from its 64-character fingerprint, matched the run manifest against
the study manifest, checked raw result and trace hashes against the verified
source-evaluation cache, and required all recorded source audits to pass.

The study's run-start Git field was null or absent. The code revision is mapped
to commit `{code_mapping["editable_install_commit"]}` through the editable
repository pin in the verified dependency freeze. This is a dependency-based
mapping, not a replacement for a missing run-start Git field.

The GPU model `{gpu_audit["gpu_model"]}` comes from an operator-reported
NVIDIA-SMI audit on the same workstation after study completion. It was not
recorded in the run-start study environment field and is labeled separately.

## Data boundary

This bundle was exported only after confirming `target_calls = 0` and
`target_evaluation_performed = false` at study and run level. Victim fitting or
loading and clean-accuracy validation covered every family, including the
held-out family in each fold. The zero-call statement refers specifically to
held-out attack-evaluation calls. The hidden target attack cohort remained
unopened, so the bundle contains no target attack ASR or transfer result.

## Redaction policy

The Git bundle excludes checkpoint binaries, training logs, absolute paths,
unsanitized machine records, and credentials. It includes a sanitized
dependency freeze plus the checksum-verified raw source result rows and query
traces in `raw_source_records.tar.gz`. Archive members use portable run-ID
paths with timestamps, owners, and permissions normalized for deterministic
output. Checkpoint content hashes are retained for provenance, while the local
archive remains authoritative for excluded model files.

`SHA256SUMS` authenticates the contents of this directory. It intentionally
does not hash itself.
"""
