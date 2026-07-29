# D1 source-only residual-ranker protocol

**Lock date:** 29 July 2026

**Status:** Preregistered before the production D1 run
**Purpose:** Test whether a baseline-preserving residual ranker can improve the
strong score-greedy source attack before any larger replication is considered

## Research question

Does a behavior-cloned residual ranker improve attack success and query
efficiency over the deterministic score-greedy proposal order on visible,
held-out source-model instances?

D1 is an exploratory source-development experiment. It is not a hidden-target
transfer evaluation and cannot establish a publication-level transferability
claim.

## Immutable boundaries

- Dataset: CIFAR-10 using the already authenticated local dataset bytes
- Source evidence manifest SHA-256:
  `efd96c5775187ac29fbd1453e3d1654d26373fc17b7c0d22b0e4955215a0e054`
- CIFAR-10 tree SHA-256:
  `c1adf901d7d67ca1df1a1d0d5ae49a079ed82d7ac742568da8041aa60a54f9b7`
- Held-out family: `modern_cnn`
- Visible source families: `classical_cnn` and `transformer`
- Policy seed: `17`
- Teacher episodes: `200`
- D1a source-evaluation images: `50`
- D1b source-evaluation images: `50`
- Query budget: `50` victim calls per method and episode, including
  initialization
- Perturbation budget: \(L_\infty = 8/255\)
- Proposal step: \(2/255\)
- Action space: 4 by 4 patch grid, three channels, and two signs, for 96
  actions
- Operator: rollback on non-improvement
- Device: CUDA workstation only
- Maximum persisted study duration: no later than eight hours, with the inner
  study deadline finishing before the outer watchdog cleanup window
- Hidden-target calls: exactly `0`
- Hidden-target evaluation: disabled

The frozen victim checkpoints are reused after checksum and schema
verification. Victim models must not be retrained, modified, or replaced.
The held-out `modern_cnn` family must not be constructed, loaded, queried, or
used for threshold selection.

## Data and victim roles

The following image roles are pairwise disjoint and their ordered index
digests must be written to the evidence:

| Role | Images | Permitted use |
| --- | ---: | --- |
| Teacher training | 200 | Residual behavior-cloning teacher collection |
| Threshold selection | 50 | Confidence fallback threshold only |
| Competence gate | 50 | Source competence decision only |
| D1a evaluation | 50 | Score-greedy versus frozen BC residual ranker |
| D1b evaluation | 50 | Reserved score-greedy, BC, and PPO comparison |

For each visible family, teacher collection uses the two authenticated exact
source instances. Evaluation uses one separate authenticated
seen-family-new-instance victim. Teacher and evaluation victim identifiers
must be disjoint.

No threshold, optimizer choice, episode count, or method decision may use the
D1a or D1b evaluation rows.

## D1a method

The control is the deterministic score-greedy proposal order. The candidate
adds a recurrent residual score to that prior and falls back to score greedy
when residual confidence does not clear a validation-selected threshold.

The locked candidate settings are:

- recurrent observation dimension: 536
- action-conditioned actor with hidden dimension 128
- score-prior temperature: 24.0
- normalized soft-teacher temperature: 0.5
- teacher decisions per training episode: 12
- teacher decisions per validation episode: 6
- teacher collection block size: two episodes, with deadline checks before and
  after every collection block
- behavior-cloning epochs: 12
- behavior-cloning objective: equal-family, equal-trajectory listwise soft
  cross-entropy plus pairwise logistic loss with weight 0.1 and up to five
  hard negatives per accepted step
- behavior-cloning optimizer: Adam with learning rate 0.0003 and gradient-norm
  clipping at 0.5
- behavior-cloning deadline checks: before each epoch, trajectory, and example,
  before optimizer preparation, and immediately before each optimizer step

The locked confidence-threshold procedure evaluates the candidates 0.0, 0.05,
0.1, 0.2, 0.5, 1.0, and 2.0, plus the smallest floating-point value above the
maximum observed validation confidence as an always-fallback candidate. It
uses accepted steps from the 50-image threshold-selection role only, aggregates
equally by trajectory and then source family, maximizes accuracy, breaks ties
by minimizing residual use and then preferring the larger threshold, and
disables overrides when the selected candidate always falls back. An override
is used when the nonnegative learned-versus-prior logit advantage is greater
than or equal to the selected threshold.

Both methods must use identical source victims, image indices, episode seeds,
attack operator, perturbation limit, and total query budget. D1a must record
all source victim calls, including the initialization query.

The locked dense reward for every BC-aligned source attack and PPO rollout is
the true-class margin reduction multiplied by 5.0, plus a 2.0 terminal success
bonus when applicable, minus a 0.01 query penalty. The margin is the true-class
score minus the largest rival-class score. Non-improving proposals are rolled
back. PPO discounted returns use a factor of 0.98.

## Preregistered D1a gate

D1a passes only when every condition below is true:

1. All artifact, cohort, query, perturbation, frozen-policy, and source-only
   audits pass.
2. Equal-family macro residual accuracy gain over the score prior is strictly
   positive on the competence role.
3. Equal-family macro soft cross-entropy improvement over the prior is
   strictly positive.
4. Residual use on the competence role is nonzero.
5. Accuracy gain and soft cross-entropy improvement are both strictly
   positive in the worst visible family.
6. The confidence threshold was selected only on the threshold-selection
   role.
7. BC residual ASR is at least score-greedy ASR in each visible family.
8. BC residual ASR-query AUC is at least score-greedy AUC in each visible
   family.
9. The residual is used for at least 1 percent of deployment decisions.

These are point-estimate development gates. They are not inferential
non-inferiority tests.

If any condition fails, D1a remains a valid negative result and D1b is
recorded as skipped. The gates must not be relaxed after results are observed.

## Conditional D1b PPO refinement

D1b is authorized only by a passing D1a gate. It trains four resumable PPO
blocks of 50 episodes each, for a maximum of 200 source-only PPO episodes.
Every block must persist a checksummed checkpoint, receipt, parent-receipt
identity, source-call accounting, and zero-target seal.

PPO starts from a distinct clone of the frozen BC policy and uses the same
score prior. The locked PPO controls are Adam learning rate 0.0003, clipped
ratio 0.2, critic-loss weight 0.5, entropy weight 0.0001, gradient-norm
clipping at 0.5, and four optimizer epochs per block update. The source-family
schedule is balanced, starts with equal family mass, and updates family weights
with exponentiated GroupDRO step size 0.1. The rollout seed is 80017 and the
score-prior seed is 50017. Deadline checks occur before and after every PPO
optimizer step, and a failed pre-step check must prevent the mutation.

After PPO, the confidence threshold is selected again using only the locked
threshold-selection role and the same candidate and tie-breaking procedure.
The competence role and reserved D1b evaluation role remain separate from
threshold selection.

The reserved D1b cohort compares:

- `score_greedy`
- `residual_ranker_bc`
- `residual_ranker_bc_ppo`

The frozen BC method must reproduce non-decreasing ASR and AUC against
score-greedy in each visible family. PPO is eligible for selection only if:

1. its macro and worst-family competence gains are strictly positive;
2. its validation-selected residual use is nonzero;
3. it overrides score greedy for at least 1 percent of deployment decisions;
4. its ASR and AUC do not decrease against score greedy in either visible
   family; and
5. its ASR and AUC do not decrease against frozen BC in either visible
   family.

If PPO fails but frozen BC reproduces, frozen BC remains the selected
source-development method. If frozen BC fails to reproduce, no method is
selected. Neither outcome authorizes hidden-target evaluation.

## Required evidence

The completed or bounded partial study must preserve:

- raw per-method, per-victim, per-image result rows;
- query traces and action histories;
- ASR at every query budget from 0 through 50;
- ASR-query AUC;
- per-family and per-victim summaries;
- matched source call counts;
- action distributions, override fractions, fallback fractions, and entropy;
- fixed-victim paired image-bootstrap intervals with 10,000 resamples;
- training, threshold, competence, and PPO-block diagnostics;
- elapsed time, component runtimes, GPU peak memory, and sampled GPU telemetry;
- exact code revision, code digest, dataset digest, source-manifest digest,
  victim identities, runtime identity, configuration, and seeds;
- SHA-256 sidecars and a package-level checksum index;
- explicit pass or fail state for every gate; and
- exact zero values for hidden-target calls and hidden-target evaluations.

Intervals condition on one fixed policy seed and fixed visible source victims.
They are descriptive and must not be presented as across-seed confidence
intervals.

## Failure and recovery policy

The study fails closed when source-only isolation, provenance, cohort
matching, query accounting, checkpoint integrity, role disjointness, runtime
identity, or deadline enforcement cannot be verified.

Resume is permitted only from a verified checkpoint belonging to the same
Git revision, code digest, dataset bytes, source manifest, runtime, seed, and
persisted deadline. Completed blocks are immutable. Automatic retry is limited
to a safely interrupted D1b run with a verified receipt prefix and no partial
final evidence.

The experiment stops no later than the persisted eight-hour deadline. A
deadline-limited partial result is reported as partial, not complete.

## Claim limits and next decision

A positive D1 result supports only a larger source-only replication. It does
not show cross-family attack transfer to hidden targets. A negative D1 result
supports abandoning or revising this residual-ranking candidate without
opening the hidden targets.

No result from this single-fold, single-seed development experiment may be
called publishable. Multi-seed confirmation and a separately authorized,
one-time hidden-target evaluation remain necessary for a positive
transferability claim.
