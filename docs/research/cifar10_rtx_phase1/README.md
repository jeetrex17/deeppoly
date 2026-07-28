# CIFAR-10 RTX Phase 1 evidence

This directory is the compact, checksum-verifiable evidence bundle for the
source-attack phase of the preregistered cross-victim RL attack study.

## Outcome

The 30-run source grid completed, but the strict source-competence gate
did not pass. The correct study status is `source_learning_failed`. Victim
models from every family, including each fold's held-out family, were fit or
loaded and clean-accuracy validated. The hidden target attack cohort remained
unopened, and held-out attack-evaluation calls were 0.

Across exact and new-instance source-victim evaluations, the hybrid BC +
GroupDRO + PPO policy
reached ASR `0.0705` and query-normalized AUC
`0.0325`. Score greedy reached ASR
`0.0561` and AUC
`0.0247`. The observed gains were
`0.0145` ASR and
`0.0078`
AUC. These gains are positive, but below the preregistered per-condition gate.

Behavior cloning also failed its strict competence gate in all
`30` runs. Mean validation action accuracy was
`0.0284`, compared with majority accuracy
`0.0295` and uniform chance
`0.0104`. This supports a representation or
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
  `141`-package transitive dependency freeze.
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
`791140871a987ec400cca083aea9b1192d8e73f2a5e70e5504dcfcae7f85911d`. All
`30` run manifests, source-evaluation caches, result
rows, and trace files were checksum-verified before export.

Total measured source-phase wall time was
`16.26` hours. Rerunning the exporter with
unchanged inputs produces byte-identical files, including the gzip archive.
