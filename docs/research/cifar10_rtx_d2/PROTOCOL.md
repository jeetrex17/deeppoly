# D2 source-only multi-seed GroupDRO protocol

**Lock date:** 29 July 2026

**Status:** Preregistered before D2 training or evaluation

## Research question

Does worst-family-aware behavior cloning produce a residual attack ranker that
does not regress against score greedy on either visible source family across
three independently initialized policies?

D2 is a source-only replication and repair study motivated by the family
sensitivity observed in D1. It does not evaluate transfer to the hidden
`modern_cnn` family and cannot by itself support a hidden-target transfer
claim.

## Locked experiment

- Dataset: authenticated local CIFAR-10 bytes used by the prior RTX study
- Source fold seed: `17`
- Policy seeds: `223`, `227`, and `229`
- Visible source families: `classical_cnn` and `transformer`
- Held-out family: `modern_cnn`
- Device: CUDA workstation only
- Behavior-cloning epochs per seed: `12`
- GroupDRO exponentiated-weight step size: `0.1`
- GroupDRO training images: `600`
- Threshold-selection images: `100`
- Competence-gate images: `100`
- Source-evaluation images: `100`
- Hidden-target calls: exactly `0`
- Hidden-target evaluation: disabled
- PPO episodes in this rung: `0`

The attack operator, query budget, perturbation limit, rollback behavior,
authenticated source victims, model checkpoints, and source-family identities
remain identical to D1 unless this protocol explicitly changes them. Frozen
victims must not be retrained or modified.

## Fresh source roles

All role indices are supplied explicitly to the allocator and recorded with
ordered SHA-256 digests. Candidate order is frozen before execution. The
allocator must reject duplicate indices, an undersized complement, overlap
between roles in the same split, and intersection with any supplied historical
role.

The earlier Phase 2 screen visited the first `600` `policy_train` images. D2
GroupDRO training uses the first `600` candidates from the remaining
`policy_train` complement.

The historical `source_validation` allocation used `500` images for victim
validation, followed by `100` images for BC validation and `100` images for
source evaluation. D2 excludes all `700` of those observed indices. It then
allocates the following roles, in order, from the untouched
`source_validation` complement:

| Role | Split | Images | Permitted use |
| --- | --- | ---: | --- |
| GroupDRO BC training | `policy_train` | 600 | Teacher collection and GroupDRO BC updates |
| Global threshold selection | `source_validation` | 100 | Threshold candidates and fallback selection only |
| Competence gate | `source_validation` | 100 | Teacher-accuracy and loss gates only |
| D2 source evaluation | `source_validation` | 100 | Final paired ASR and query-AUC comparison only |

No result from the competence or evaluation role may change training,
threshold candidates, optimizer settings, epoch count, GroupDRO step size, or
promotion criteria. D1 observations must not be reused as D2 examples or used
to tune this locked protocol.

## GroupDRO behavior cloning

Each policy seed starts from an independent initialization. Training uses the
same score-greedy prior and residual-ranker architecture as D1. The behavior
cloning objective retains the D1 listwise soft cross-entropy and pairwise
ranking components. Family losses are aggregated by GroupDRO with
exponentiated family weights and step size `0.1`. Training runs for exactly
`12` epochs for each of the three locked policy seeds.

Family batches, trajectory weights, accepted-step accounting, gradient
clipping, optimizer settings, teacher temperature, and proposal prior remain
fixed across seeds. Every seed records per-epoch family losses and GroupDRO
weights. The implementation must fail closed on missing families, non-finite
losses, invalid weights, source-call mismatches, or any target access.

## Global fallback threshold

One confidence threshold is selected globally for both visible source
families. Every candidate must report family-specific residual teacher
accuracy and prior accuracy on the threshold role. A candidate is eligible
only when residual accuracy is at least prior accuracy in every visible
family.

The candidate set must contain exactly one finite always-fallback threshold.
That candidate disables residual overrides, has zero residual use, and exactly
reproduces prior accuracy in every family. Among eligible candidates, select
the highest equal-family macro accuracy, then the lowest residual-use
fraction, then the larger threshold.

The production runner should additionally compute paired score-greedy and
residual ASR and query-AUC on threshold-role attacks. A non-fallback threshold
is deployable only if both metrics are non-decreasing in every visible family.
If no non-fallback candidate satisfies the attack-level check, the locked
always-fallback candidate is selected. The threshold role remains separate
from competence and final evaluation.

## Source gates and promotion

All artifact, provenance, cohort, call-accounting, perturbation, checkpoint,
role-disjointness, and zero-target audits must pass.

For every policy seed and every visible source family:

1. the competence role must satisfy the locked teacher-accuracy and soft-loss
   source gates;
2. the selected global threshold must satisfy the family-safe fallback rule;
3. residual ASR on the untouched D2 evaluation role must be at least
   score-greedy ASR; and
4. residual ASR-query AUC on that same role must be at least score-greedy AUC.

The aggregate decision requires exactly six evaluation cells: three policy
seeds times two source families. Missing, duplicate, additional, non-finite,
or out-of-range cells invalidate the decision. A regression in either ASR or
query-AUC in any one cell fails D2, even if the aggregate mean improves.
Across the six cells, mean ASR gain must be at least `0.010` and mean query-AUC
gain must be at least `0.005`. The mean ASR gain and mean query-AUC gain within
each visible family must also both be strictly positive.

A passing D2 result authorizes only a separately preregistered source-only PPO
study. It never authorizes hidden-target access. PPO must not run unless every
D2 source gate passes. A separate human-approved protocol is required before
any one-time hidden-target evaluation.

## Required evidence

The D2 package must preserve:

- ordered role indices and SHA-256 digests;
- the complete forbidden-index manifests and their digests;
- exact code revision, environment identity, dataset digest, source-manifest
  digest, checkpoint digests, victim identities, and all locked settings;
- per-seed and per-family training losses, GroupDRO weights, threshold
  diagnostics, competence metrics, ASR, query-AUC, and paired deltas;
- raw per-image method rows, action traces, query traces, source-call
  accounting, elapsed time, and GPU telemetry;
- one explicit pass or fail value for every source gate;
- exact zero target and hidden-target calls; and
- checksums for all preserved evidence.

Confidence intervals may be reported as descriptive fixed-victim intervals.
The three locked seeds permit seed-level variability to be reported, but they
do not replace a broader victim-family replication.

## Failure policy and claim limit

Any unverifiable identity, overlap, missing role, metric regression, invalid
checkpoint, query-accounting error, non-finite value, or target access makes
the run invalid or negative as appropriate. Gates must not be relaxed after
results are observed.

A positive D2 result would be evidence that the revised source learner is
stable enough for the next source-only rung. It is not evidence of hidden
cross-family transfer. A negative D2 result remains reportable evidence that
the residual learner is family sensitive under the locked source protocol.
