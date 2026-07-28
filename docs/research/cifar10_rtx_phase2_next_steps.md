# Deadline-aware next steps after Phase 2

**Decision date:** 28 July 2026
**Current result:** Phase 2 Stage B failed the locked source-only promotion gate
**D0 result:** complete; no temperature matched score-greedy ASR and AUC in every fold
**Hidden-target attack calls:** 0

## What the completed screen tells us

The three-fold screen is complete and internally consistent. The learned stochastic policy achieved 6.01% macro source ASR and 2.68% ASR-query AUC. Score greedy achieved 6.44% ASR and 2.99% AUC, while random action achieved 6.12% ASR and 2.72% AUC. Learned deterministic deployment achieved only 0.83% ASR.

The soft behavior-cloning objective also failed in all three folds:

| Omitted family | Top-5 gain over oracle | Soft CE improvement over oracle |
|---|---:|---:|
| Classical CNN | -0.0473 | -0.0501 |
| Modern CNN | -0.0504 | -0.0482 |
| Transformer | -0.0219 | -0.0482 |

The evidence supports three conclusions.

1. The attack operator can find adversarial examples, so the main problem is not zero attack opportunity.
2. The learned policy does not yield a strong deployment rule. Stochastic sampling performs near random, while deterministic argmax is weak. These observations do not distinguish calibration, representation, or action-ranking failure.
3. More episodes with the same objective are unlikely to solve the problem. PPO block returns and successes do not show a reliable late-training improvement, and the supervised representation gate already fails.

The present Phase 2 result must remain negative. Post hoc tuning cannot relabel it as a pass.

## Fast research path

The next method should be recorded as a new exploratory source-only phase. It must not modify the Phase 2 protocol, artifacts, thresholds, or seed outcome.

### Step 1: frozen-policy calibration diagnostic, completed

The source-only temperature sweep completed on the three frozen Phase 2 checkpoints. It evaluated temperatures 0.25, 0.50, 0.75, 1.00, and 1.50 against the already matched score-greedy rows.

- Recorded GPU time: 12 minutes 21 seconds
- Training required: none
- Purpose: test whether a simple global sampling-temperature change repairs the frozen checkpoints
- Scientific status: exploratory diagnosis only
- Stop condition: if no temperature reaches score-greedy ASR and AUC in every fold, do not spend time on temperature-only fixes

| Method | Macro ASR | Macro AUC | ASR gap | AUC gap |
|---|---:|---:|---:|---:|
| Score greedy | 6.435% | 2.995% | reference | reference |
| Temperature 0.25 | 5.432% | 2.628% | -1.003 points | -0.366 points |
| Temperature 0.50 | 6.106% | 2.752% | -0.329 points | -0.242 points |
| Temperature 0.75 | 6.236% | 2.874% | -0.200 points | -0.120 points |
| Temperature 1.00 | 6.007% | 2.683% | -0.429 points | -0.312 points |
| Temperature 1.50 | 6.321% | 2.810% | -0.114 points | -0.185 points |

No temperature qualified. Temperature 0.75 improved both metrics only in the transformer-held-out fold. All temperatures remained below score greedy on both metrics in the modern-CNN-held-out fold. The stop condition fired, so this D0 global-temperature protocol is closed. This diagnostic does not identify the underlying cause; residual ranking is only the next exploratory candidate.

The exact archive, 9,000 result rows, 60 query traces, tables, figures, and checksums are in the [D0 evidence bundle](cifar10_rtx_phase2_calibration/README.md).

### Step 2: build a baseline-preserving residual ranker

The fastest defensible candidate is a learned residual over the strong score-greedy prior, not another unconstrained 96-way policy.

The candidate should:

1. score actions relative to the score-greedy prior;
2. train with pairwise or listwise ranking on the gradient-teacher candidate set;
3. cache source teacher trajectories so repeated development runs do not regenerate them;
4. use a source-validation confidence gate that falls back to score greedy when the learned residual is uncertain;
5. keep the target family absent from construction, validation, training, and selection;
6. compare score greedy, residual BC only, and residual BC plus PPO on identical samples and query budgets.

This candidate is motivated by the observed failure. The current soft 96-way classifier does not beat a constant validation oracle, while score greedy is already a strong source policy. D0 did not establish that ranking is the cause.

### Step 3: use a compute funnel

Do not launch another long grid immediately.

| Rung | Scope | Approximate GPU time | Advance only if |
|---|---|---:|---|
| D0, complete | Frozen temperature diagnostic, all folds | 12 min 21 s | Failed: stop temperature-only work |
| D1 | One development fold, 200 BC and 200 PPO episodes, 50 images | 8 to 12 min | BC gains are positive and learned ASR/AUC are not below score greedy |
| D2 | All three folds, 600 BC and 600 PPO episodes, 100 images | 30 to 40 min | Mean ASR gain is at least +0.010, mean AUC gain is at least +0.005, and at least 9 of 12 conditions improve both |
| D3 | Fresh seeds 151 and 157, all folds | 60 to 75 min | Gains are positive in every fold and both BC diagnostics pass |

Each rung is resumable and has a hard wall-clock limit. A failed rung stops the candidate. This keeps a weak idea below roughly 15 minutes and a promising full source replication below roughly two hours.

### Step 4: preserve the confirmatory boundary

If D3 passes, freeze the representation, teacher, ranking loss, fallback rule, optimizer, temperature, and episode count before opening any reserved source-promotion data. Do not replace failed seeds, relax thresholds, or inspect hidden targets.

The final source test and one-time target evaluation remain separate. A positive development screen is not a transfer result.

## Submission strategy

Two submission tracks can proceed in parallel.

### Track A: rigorous negative study

The current evidence can support a careful workshop or technical-report submission about failure modes in RL-based cross-victim attack transfer:

- Phase 1 provides a ten-seed source study with matched ablations and controls.
- Phase 2 tests a preregistered corrective architecture and stops after a short failed screen.
- Score greedy remains stronger than the learned policy under the matched threat model.
- Artifact hashes, raw numerical rows, query traces, and zero-target audits are public.

The claim must stay narrow. The experiments show that the tested RL formulations did not establish transferable source competence on the fixed CIFAR-10 victim bank. They do not show that attack transferability is impossible.

### Track B: positive method attempt

Pursue the residual-ranking candidate only through the compute funnel above. If it fails D1 or D2, stop and strengthen Track A. If it passes D3, budget the larger confirmatory study before making a publication-level positive claim.

## Immediate order of work

1. Completed: commit and publish the Phase 2 Stage B evidence bundle.
2. Completed: update the research report with the locked negative Stage B result.
3. Completed: implement and run the frozen temperature diagnostic with a 15-minute limit.
4. Completed: select residual ranking because no global temperature qualified.
5. Current: implement and run the one-fold D1 residual-ranker screen.
6. Stop for explicit approval before any D2 or longer replication job.

This sequence protects the deadline, the sealed target boundary, and the credibility of the final report.
