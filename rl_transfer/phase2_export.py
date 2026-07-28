"""Deterministic, Git-safe evidence export for Phase 2 Stage B."""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

from .artifacts import sha256_file
from .paths import resolve_descendant
from .phase1_export_validation import (
    require_mapping,
    validate_portable_value,
    validated_output_directory,
)
from .phase2_export_archive import raw_source_archive
from .phase2_export_render import (
    bc_figure,
    evidence_readme,
    gain_figure,
    provenance,
    runtime_figure,
    source_asr_figure,
)
from .phase2_export_summary import (
    attempt_log_rows,
    build_summary,
    compact_runs,
    environment_summary,
)
from .phase2_export_tables import (
    LEARNED_METHOD,
    condition_rows,
    fold_rows,
    method_rows,
    training_block_rows,
    victim_rows,
)
from .phase2_export_validation import (
    load_bounded_verified_json_with_digest,
    verified_dependency_freeze,
    verified_runs,
    verify_all_sidecars,
)
from .phase2_promotion import SCREEN_CONTROL


_MAX_COMPACT_BYTES = 25 * 1024 * 1024
_MAX_RAW_ARCHIVE_BYTES = 50 * 1024 * 1024
_MANAGED_OUTPUTS = {
    "README.md",
    "PROVENANCE.md",
    "SHA256SUMS",
    "summary.json",
    "environment_summary.json",
    "dependency_freeze.txt",
    "condition_metrics.csv",
    "fold_summary.csv",
    "bc_diagnostics.csv",
    "training_blocks.csv",
    "victim_accuracy.csv",
    "input_checksums.csv",
    "attempt_log_checksums.csv",
    "raw_compact_evidence.json.gz",
    "raw_source_records.tar.gz",
    "source_asr_by_method.svg",
    "gain_vs_score_greedy.svg",
    "bc_diagnostics.svg",
    "runtime.svg",
}


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode("utf-8"))


def _json_bytes(value: object, *, compact: bool = False) -> bytes:
    options: dict[str, object] = {
        "sort_keys": True,
        "allow_nan": False,
    }
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    return (json.dumps(value, **options) + "\n").encode()


def _csv_text(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _formatted(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(
        ("=", "+", "-", "@")
    ):
        return "'" + value
    return value


def _formatted_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {key: _formatted(value) for key, value in row.items()}
        for row in rows
    ]


def _write_checksums(output: Path) -> None:
    paths = sorted(
        (
            path
            for path in output.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        key=lambda path: path.name,
    )
    _write_text(
        output / "SHA256SUMS",
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in paths
        ),
    )


def _compact_evidence(
    *,
    screen: Mapping[str, object],
    screen_manifest_sha: str,
    environment: Mapping[str, object],
    input_checksums: Sequence[Mapping[str, object]],
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    compact = {
        "schema_version": 1,
        "screen": {
            key: screen[key]
            for key in (
                "schema_version",
                "name",
                "status",
                "research_valid",
                "publication_candidate",
                "target_calls",
                "target_evaluation_performed",
                "study_code_digest",
                "protocol_sha256",
                "base_config_digest",
                "dataset_version",
                "config",
                "victim_cache_reuse",
                "screen_promotion_decision",
            )
            if key in screen
        },
        "screen_manifest_sha256": screen_manifest_sha,
        "environment": environment,
        "input_checksums": list(input_checksums),
        "runs": compact_runs(runs),
    }
    validate_portable_value(compact, "compact Phase 2 evidence")
    return compact


def _write_tables(
    output: Path,
    *,
    conditions: Sequence[Mapping[str, object]],
    folds: Sequence[Mapping[str, object]],
    bc_rows: Sequence[Mapping[str, object]],
    blocks: Sequence[Mapping[str, object]],
    victims: Sequence[Mapping[str, object]],
    input_checksums: Sequence[Mapping[str, object]],
    attempt_logs: Sequence[Mapping[str, object]],
) -> None:
    table_specs = (
        ("condition_metrics.csv", conditions),
        ("fold_summary.csv", folds),
        ("bc_diagnostics.csv", bc_rows),
        ("training_blocks.csv", blocks),
        ("victim_accuracy.csv", victims),
        ("input_checksums.csv", input_checksums),
    )
    for filename, rows in table_specs:
        if not rows:
            raise ValueError(f"{filename} cannot be empty")
        _write_text(
            output / filename,
            _csv_text(tuple(rows[0]), _formatted_rows(rows)),
        )
    _write_text(
        output / "attempt_log_checksums.csv",
        _csv_text(
            ("filename", "bytes", "sha256"),
            _formatted_rows(attempt_logs),
        ),
    )


def _write_bundle(
    output: Path,
    *,
    summary: Mapping[str, object],
    environment: Mapping[str, object],
    freeze: str,
    conditions: Sequence[Mapping[str, object]],
    methods: Sequence[Mapping[str, object]],
    folds: Sequence[Mapping[str, object]],
    bc_rows: Sequence[Mapping[str, object]],
    blocks: Sequence[Mapping[str, object]],
    victims: Sequence[Mapping[str, object]],
    input_checksums: Sequence[Mapping[str, object]],
    attempt_logs: Sequence[Mapping[str, object]],
    compressed_compact: bytes,
    raw_archive: bytes,
) -> None:
    _write_text(output / "README.md", evidence_readme(summary))
    _write_text(output / "PROVENANCE.md", provenance(summary))
    _atomic_write(output / "summary.json", _json_bytes(summary))
    _atomic_write(
        output / "environment_summary.json",
        _json_bytes(environment),
    )
    _write_text(output / "dependency_freeze.txt", freeze)
    _write_tables(
        output,
        conditions=conditions,
        folds=folds,
        bc_rows=bc_rows,
        blocks=blocks,
        victims=victims,
        input_checksums=input_checksums,
        attempt_logs=attempt_logs,
    )
    _atomic_write(
        output / "raw_compact_evidence.json.gz",
        compressed_compact,
    )
    _atomic_write(
        output / "raw_source_records.tar.gz",
        raw_archive,
    )
    method_chart_rows = [
        row
        for row in methods
        if row["method"] in {
            LEARNED_METHOD,
            "soft_gradient_bc_action_conditioned_groupdro_ppo",
            SCREEN_CONTROL,
            "bandit_action",
            "random_action",
            "fixed_action",
        }
    ]
    _write_text(
        output / "source_asr_by_method.svg",
        source_asr_figure(method_chart_rows),
    )
    _write_text(
        output / "gain_vs_score_greedy.svg",
        gain_figure(folds),
    )
    _write_text(
        output / "bc_diagnostics.svg",
        bc_figure(bc_rows),
    )
    _write_text(
        output / "runtime.svg",
        runtime_figure(folds),
    )
    _write_checksums(output)


def export_phase2_evidence(
    source_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Verify a complete Phase 2 screen and write portable evidence."""

    source = source_root.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("Phase 2 source root must be a directory")
    output = validated_output_directory(source, output_dir)
    unexpected = {
        path.name
        for path in output.iterdir()
        if path.name not in _MANAGED_OUTPUTS
    }
    if unexpected:
        raise ValueError(
            "evidence output contains unmanaged entries: "
            + ", ".join(sorted(unexpected))
        )
    sidecar_count = verify_all_sidecars(source)
    screen_path = resolve_descendant(
        source,
        "screen_manifest.json",
        label="Phase 2 screen manifest",
    )
    screen, screen_manifest_sha = (
        load_bounded_verified_json_with_digest(
            screen_path,
            label="Phase 2 screen manifest",
        )
    )
    if (
        screen.get("status") != "screen_complete"
        or screen.get("target_evaluation_performed") is not False
        or screen.get("target_calls") != 0
    ):
        raise ValueError(
            "Phase 2 export requires a complete target-free screen"
        )
    runs, input_checksums = verified_runs(source, screen)
    conditions = condition_rows(runs)
    methods = method_rows(conditions)
    folds, bc_rows = fold_rows(runs, conditions)
    blocks = training_block_rows(runs)
    victims = victim_rows(runs)
    attempt_logs = attempt_log_rows(source)
    environment = environment_summary(screen, runs)
    summary = build_summary(
        screen=screen,
        screen_manifest_sha=screen_manifest_sha,
        sidecar_count=sidecar_count,
        verified_runs=runs,
        input_checksums=input_checksums,
        method_rows=methods,
        fold_rows=folds,
        bc_rows=bc_rows,
        environment=environment,
    )
    compact = _compact_evidence(
        screen=screen,
        screen_manifest_sha=screen_manifest_sha,
        environment=environment,
        input_checksums=input_checksums,
        runs=runs,
    )
    compressed_compact = gzip.compress(
        _json_bytes(compact, compact=True),
        compresslevel=9,
        mtime=0,
    )
    if len(compressed_compact) > _MAX_COMPACT_BYTES:
        raise ValueError("compact Phase 2 evidence exceeds Git-safe size")
    raw_archive = raw_source_archive(runs)
    if len(raw_archive) > _MAX_RAW_ARCHIVE_BYTES:
        raise ValueError("raw Phase 2 records exceed Git-safe size")
    screen_environment = require_mapping(
        screen.get("runtime_environment"),
        "runtime environment",
    )
    freeze = verified_dependency_freeze(source, screen_environment)
    _write_bundle(
        output,
        summary=summary,
        environment=environment,
        freeze=freeze,
        conditions=conditions,
        methods=methods,
        folds=folds,
        bc_rows=bc_rows,
        blocks=blocks,
        victims=victims,
        input_checksums=input_checksums,
        attempt_logs=attempt_logs,
        compressed_compact=compressed_compact,
        raw_archive=raw_archive,
    )
    entries = list(output.iterdir())
    if (
        {path.name for path in entries} != _MANAGED_OUTPUTS
        or any(path.is_symlink() or not path.is_file() for path in entries)
    ):
        raise RuntimeError("Phase 2 evidence output set is incomplete")
    return summary
