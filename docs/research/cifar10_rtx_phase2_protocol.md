# Phase 2 source-competence protocol

**Protocol date:** 28 July 2026
**Status:** Locked before the first Phase 2 GPU run
**Scope:** Exploratory source-only screening, followed by a fresh-seed source replication only if the screen passes

## 1. Research objective

Phase 2 tests whether the failure in Phase 1 can be corrected by improving the observation representation, the behavior-cloning objective, the action scorer, and deployment calibration.

Phase 1 completed all 30 source cells. The hybrid policy reached source ASR 0.0705 and ASR-query AUC 0.0325. Score greedy reached ASR 0.0561 and AUC 0.0247. The gains, 0.0145 ASR and 0.0078 AUC, were positive but below the locked source thresholds. Behavior-cloning validation accuracy was 0.0284, below the 0.0295 majority baseline. PPO added only 0.0020 ASR over BC alone, and the 5,000-episode PPO curve did not improve after its early portion.

These results motivate a representation and objective change. They do not justify running the same method longer.

No part of Phase 2 guarantees a positive or publishable finding. A failed screen remains a valid negative result.

## 2. Locked scientific boundary

The following items remain unchanged:

- CIFAR-10 split seed: `20260727`
- victim-bank seed: `1000000`
- leave-one-family-out folds: classical CNN, modern CNN, transformer
- query budget: 50 total model calls, including initialization
- perturbation limit: \(L_\infty = 8/255\)
- proposal step: \(2/255\)
- action grid: 4 by 4 patches, three channels, two signs
- matched rollback-on-non-improvement operator
- primary control: score greedy
- victim architectures, fitting data, victim epochs, and validation gates

All held-out target-family victims remain sealed. Phase 2 commands expose no target or all-phase option. Every Phase 2 manifest must record:

```text
target_evaluation_performed = false
target_calls = 0
research_valid = false
publication_candidate = false
```

A positive exploratory result authorizes only more source-only work.

## 3. Victim reuse and data roles

Phase 2 reuses the Phase 1 victim bank. Before a cell starts, the runner:

1. verifies the Phase 1 study manifest and sidecar against the preregistered SHA-256 value `791140871a987ec400cca083aea9b1192d8e73f2a5e70e5504dcfcae7f85911d`;
2. derives the exact nine-checkpoint victim allowlist and checkpoint SHA-256 values from that authenticated manifest;
3. rejects missing, extra, oversized, checksum-invalid, or contract-invalid cache artifacts;
4. copies only allowlisted files atomically into a separate Phase 2 cache, without hardlinks;
5. records only repository-relative cache and checkpoint identifiers in portable manifests;
6. computes the exact cache fingerprint and selected source-victim IDs required by the Phase 2 cell;
7. preflights the complete expected source-victim checkpoint set before loading any model; and
8. refuses to run if any selected checkpoint is missing, incomplete, or checksum-invalid.

The runner uses an enforced cache-only mode and has no victim-fitting fallback. Each leave-one-family-out cell constructs, loads, and validates only the two source families. The sealed held-out family has zero model-construction, checkpoint-loading, and validation calls. These counts are written into a fail-closed isolation audit.

The 1,000-image source-validation allocation is partitioned as follows:

| Role | Images | Use |
| --- | ---: | --- |
| Victim validation | 500 | Fixed victim-quality gate |
| BC validation | 100 | Representation and distillation diagnostics |
| Development source attack set | 100 | Phase 2 screening |
| Reserved source-promotion set | 200 | One-time evaluation after fresh-seed replication |
| Unused buffer | 100 | Not used for Phase 2 selection |

The 100-image development attack set is a fixed subset of the Phase 1 source-gate role. Both `exact_source` and `seen_family_new_instance` victim slices are development measurements during the short screen. The separate 200-image source-promotion set remains unopened until all method and training choices are frozen.

The 40,000-image victim-fit role, 4,000-image policy-training role, and 1,000-image source-validation role leave 5,000 CIFAR-10 training images unallocated. A deterministic class-balanced 1,000-image final source-test set, with 100 images per class, is preregistered from this untouched complement. It uses split seed `20260727` and allocator `balanced-complement-shuffle-v1`. Its ordered-index SHA-256 digest is:

```text
0a1329758e414723e597c878f417873312db74165a6d59aa467e966b5873810a
```

The indices are sealed from evaluation during Stages A and B and throughout both Stage C rungs and Stage C promotion. They can be evaluated only in the final confirmatory source study. The machine-validated lock is `configs/rl_transfer/cifar10_rtx_phase2_confirmatory_contract.json`.

## 4. Phase 2 method

The trainable candidate changes four items together. It is therefore evaluated as one method package and does not support a causal claim about any single component.

### 4.1 Patch-statistics observation

The Phase 1 patch means are retained. Each patch and channel also receives:

- original pixel standard deviation;
- internal horizontal and vertical edge magnitude;
- absolute perturbation magnitude normalized by epsilon;
- positive action-region headroom; and
- negative action-region headroom.

The deterministic feature order and exact dimension are tested. At grid size 4, the image block has 336 values and the full recurrent observation has 536 values.

### 4.2 Normalized soft gradient distillation

For each owned source-model teacher state, the privileged teacher computes one-step linearized costs for all 96 actions. Costs are normalized by their within-state standard deviation before applying a fixed softmax temperature of `0.50`:

```text
teacher_probability(a) =
    softmax(-(cost(a) - min(cost)) / (std(cost) * 0.50))
```

Normalization makes the teacher distribution invariant to multiplicative gradient scale. The policy is trained with soft cross-entropy. Validation records top-1 and top-5 agreement, soft cross-entropy, KL divergence, teacher entropy, probability regret, and uniform and validation-oracle baselines. The validation oracle is the retrospective empirical best-constant predictor estimated from the evaluated validation teacher-label marginal without smoothing. It is unavailable to the trained or deployed policy and is used only as a diagnostic baseline.

### 4.3 Action-conditioned recurrent actor

The actor scores each action through shared patch-row, patch-column, channel, and sign features. This replaces 96 unrelated output neurons with a shared action scorer while retaining the recurrent encoder, GRU memory, and critic.

### 4.4 Short PPO refinement

The exploratory candidate uses:

```text
BC collection episodes       600
BC validation episodes       100
BC epochs                     25
BC batch size                512
PPO episodes                 600
PPO update block              50
PPO learning rate         0.0003
PPO entropy weight        0.0001
PPO update epochs               6
```

Component ablations are disabled during the short screen to avoid repeating the Phase 1 PPO-only cost. Required ablations return in the final confirmatory source study.

## 5. Stage A: Phase 1 checkpoint temperature diagnostic

Stage A does not train a model. It evaluates the frozen Phase 1 seed-17 hybrid checkpoint in all three folds at sampling temperatures:

```text
0.25, 0.50, 0.75, 1.00, 1.50
```

The same exact-source development samples, victim checkpoints, action operator, episode seeds, and query budget are used for every temperature. Score greedy is evaluated once on the same eligible cohorts. Policy parameters and digests must remain unchanged.

Candidates are ranked by macro ASR gain versus the matched score-greedy control. Candidates within 0.002 of the best macro ASR gain are treated as tied. Tied candidates are ranked by higher macro ASR-query AUC gain versus score greedy, then by lower temperature. A non-default temperature is reported as preferred for this frozen Phase 1 checkpoint only when:

- its macro ASR is at least 0.005 above temperature 1.00;
- its macro AUC is no more than 0.002 below temperature 1.00;
- every fold has normalized action entropy in `[0.10, 0.95]`; and
- all query, sample-alignment, checkpoint, and frozen-policy audits pass.

Otherwise temperature 1.00 remains the diagnostic default. Stage A has a 10-minute scheduling budget. Stage A is a diagnostic of the flat Phase 1 policy only. It does not select a deployment temperature and does not set the temperature of the action-conditioned Phase 2 candidate.

## 6. Stage B: one-seed joint candidate screen

The improved candidate is trained with policy seed 17 in all three folds. Each cell uses the fixed development data, the verified victim cache, 600 BC episodes, 600 PPO episodes, and 100 attack-evaluation images. Stage B evaluation temperature is locked at `1.00`; no Stage A selection is transferred to the new action-conditioned policy.

The runner has a 60-minute scheduling budget and checks the deadline between cells. It checkpoints every PPO block and resumes only when the configuration, code, dataset, cache, and prior manifest fingerprints match.

The exploratory screen passes only if:

- all three fold cells complete;
- all victim and raw-evidence audits pass;
- target calls remain zero;
- mean soft top-5 gain over the validation-oracle baseline is at least 0.01;
- mean soft cross-entropy improvement over the validation-oracle baseline is at least 0.02;
- mean ASR gain over score greedy is at least 0.01;
- mean ASR-query AUC gain over score greedy is at least 0.005; and
- ASR and AUC gains are both positive in at least 9 of the 12 source conditions.

This is a deliberately permissive compute-allocation gate. It is not a publication gate and cannot authorize target evaluation.

If Stage B fails, Phase 2 stops. The method is reported as an unsuccessful source-learning candidate.

## 7. Stage C: fresh-seed source replication

Stage C occurs only if Stage B passes. All representation, teacher, actor, temperature, reward, and optimizer choices are frozen first.

The replication uses fresh policy seeds 151 and 157 across all three folds. These are the first two prime numbers strictly above the maximum Phase 1 seed, 149, and are disjoint from the Phase 1 and Stage B policy seeds. It begins with 600 PPO episodes. A candidate advances to at most 1,200 episodes only if the first rung shows:

- positive mean ASR and AUC gains in every fold;
- macro ASR gain of at least 0.025 over score greedy;
- macro AUC gain of at least 0.010;
- a passing soft-BC representation gate; and
- no regression over the final 200 training episodes.

Failure stops the replication before additional compute is spent.

After the method and episode count are frozen, the reserved 200-image source-promotion set is evaluated exactly once. Promotion requires positive ASR and AUC gains in every fold, macro ASR gain of at least 0.040, macro AUC gain of at least 0.015, valid entropy, unchanged policy digests, complete matched cohorts, and zero target calls.

The promotion set cannot be used to revise the method.

## 8. Final confirmatory source study

Only a passing Stage C promotion result can start the final source study. It uses the ten locked policy seeds:

```text
163, 167, 173, 179, 181, 191, 193, 197, 199, 211
```

These are the next ten prime numbers after the Stage C seeds and are disjoint from every Phase 1, Stage B, and Stage C policy seed. The final study restores BC-only and PPO-only ablations and evaluates the untouched 1,000-image final source-test set. Episode count is capped at the value selected before promotion. The original strong source gate remains in force:

- mean ASR gain over matched controls at least 0.05;
- mean AUC gain at least 0.02;
- positive effects in every held-out fold and source slice;
- policy-seed bootstrap lower bounds above zero;
- exact paired sign-flip \(p \leq 0.05\);
- hybrid ASR at least 0.01 above both component ablations;
- passing behavior-cloning representation gate; and
- complete artifact, cohort, query, perturbation, and zero-target-call audits.

The final source run is performed once. Failed seeds are not replaced and settings are not changed after observing the result.

Only this gate can unlock the existing one-time target evaluation.

## 9. Reproducibility requirements

Every stage records:

- Git revision and worktree status;
- the active protocol SHA-256 value;
- full configuration and configuration digest;
- Python, PyTorch, CUDA, cuDNN, and GPU information;
- dataset tree, split, and role digests;
- victim, policy, optimizer, result, trace, and manifest SHA-256 values;
- every seed and deterministic-runtime setting;
- source and target call counts;
- accepted, rejected, interrupted, and failed cells;
- per-stage wall time and peak GPU memory when available;
- raw per-image results and the fixed query-trace subset; and
- the decision produced by the applicable gate.

Malformed, missing, non-finite, misaligned, or checksum-invalid evidence causes failure. It is never silently excluded.

## 10. Claim limits

The short screen is exploratory and uses one policy seed. Stage C uses only two fresh seeds. Neither stage estimates the final across-seed effect.

The fixed custom victim bank supports a narrow CIFAR-10 claim only. Soft gradient supervision uses privileged gradients on owned source victims during training, while deployment remains score-based and frozen. Better source competence is necessary but not sufficient for transfer to sealed target families.

If Phase 2 or the final source gate fails, the correct conclusion is that the tested method did not establish transferable source competence under this threat model and compute budget.
