# Provenance

This directory was generated deterministically from the checksum-verified Phase 2 Stage B archive copied from the RTX workstation.

- Screen manifest SHA-256: `efd96c5775187ac29fbd1453e3d1654d26373fc17b7c0d22b0e4955215a0e054`
- Study code digest: `48f670536cf4251555e110e9447f58ecf2bf8561bec49d67d7eba350dc9e0c78`
- Protocol SHA-256: `d67edae9b11f1ee499927378f64c34952837ef335d4c7db996bc3f9cd4718564`
- Verified source runs: 3
- Verified sidecars: 19
- Verified raw result files: 3
- Verified query-trace files: 3
- Target attack calls: 0
- GPU: NVIDIA GeForce RTX 2080 Ti
- CUDA runtime: 13.0
- PyTorch: 2.13.0+cu130
- Git revision used for training: `1525dc14c617ff7c715dd1bb62f48092d25ee941`

Every run was source-only. The family named as the target of a fold was neither constructed nor clean-validated in that fold. The `victim_access_audit` records are included in the compact evidence.

The full local archive also contains the binary checkpoints. This Git bundle excludes those binaries and execution-log contents. It retains checkpoint hashes, exact raw numerical result rows, sampled query traces, dependency pins, and log hashes so the published evidence remains portable and reviewable.
