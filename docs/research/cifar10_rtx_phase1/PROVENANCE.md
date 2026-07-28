# Provenance and integrity

## Source record

- Study name: `cifar10-rtx-publication`
- Study schema: `1`
- Study manifest SHA-256: `791140871a987ec400cca083aea9b1192d8e73f2a5e70e5504dcfcae7f85911d`
- Verified source runs: `30`
- Verified raw result files: `30`
- Verified raw trace files: `30`
- Study code digest: `5d976361382cbb07c69b03b6dd88312c1238373da41f92c99c3f389e66a5b0b0`
- CUDA runtime recorded at run start: `13.0`
- cuDNN version recorded at run start: `92000`
- NVIDIA driver recorded at run start: `580.126.09`
- GPU model in the run-start study field: `unknown`
- Git revision in the run-start study field: `unknown`
- PyTorch from the verified freeze: `2.13.0`
- torchvision from the verified freeze: `0.28.0`
- Editable-install code mapping: `47bd57e9c6826a9e09203de2adacef64a75ace4e`

The exporter verified each JSON SHA-256 sidecar, reconstructed each run
directory from its 64-character fingerprint, matched the run manifest against
the study manifest, checked raw result and trace hashes against the verified
source-evaluation cache, and required all recorded source audits to pass.

The study's run-start Git field was null or absent. The code revision is mapped
to commit `47bd57e9c6826a9e09203de2adacef64a75ace4e` through the editable
repository pin in the verified dependency freeze. This is a dependency-based
mapping, not a replacement for a missing run-start Git field.

The GPU model `NVIDIA GeForce RTX 2080 Ti` comes from an operator-reported
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
