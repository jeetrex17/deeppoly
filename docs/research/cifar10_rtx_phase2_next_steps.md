# Deadline-aware next steps after Phase 2

**Decision date:** 28 July 2026
**Current result:** Phase 2 Stage B failed the locked source-only promotion gate
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
2. The policy distribution remains poorly calibrated. Stochastic deployment behaves close to random, while deterministic deployment collapses to a weak action choice.
3. More episodes with the same objective are unlikely to solve the problem. PPO block returns and successes do not show a reliable late-training improvement, and the supervised representation gate already fails.

The present Phase 2 result must remain negative. Post hoc tuning cannot relabel it as a pass.

## Fast research path

The next method should be recorded as a new exploratory source-only phase. It must not modify the Phase 2 protocol, artifacts, thresholds, or seed outcome.

### Step 1: frozen-policy calibration diagnostic

Run a source-only temperature sweep on the three completed Phase 2 checkpoints. Evaluate only the learned policy at temperatures 0.25, 0.50, 0.75, 1.00, and 1.50 against the already matched score-greedy rows.

- Expected GPU time: 10 to 15 minutes
- Training required: none
- Purpose: determine whether the gap is mainly calibration or representation
- Scientific status: exploratory diagnosis only
- Stop condition: if no temperature reaches score-greedy ASR and AUC in every fold, do not spend time on temperature-only fixes

This diagnostic cannot rescue Stage B. It can only select the design direction for a new method.

### Step 2: build a baseline-preserving residual ranker

The fastest defensible candidate is a learned residual over the strong score-greedy prior, not another unconstrained 96-way policy.

The candidate should:

1. score actions relative to the score-greedy prior;
2. train with pairwise or listwise ranking on the gradient-teacher candidate set;
3. cache source teacher trajectories so repeated development runs do not regenerate them;
4. use a source-validation confidence gate that falls back to score greedy when the learned residual is uncertain;
5. keep the target family absent from construction, validation, training, and selection;
6. compare score greedy, residual BC only, and residual BC plus PPO on identical samples and query budgets.

This directly addresses the observed failure. The current soft 96-way classifier does not beat a constant validation oracle, while score greedy is already a strong source policy.

### Step 3: use a compute funnel

Do not launch another long grid immediately.

| Rung | Scope | Approximate GPU time | Advance only if |
|---|---|---:|---|
| D0 | Frozen temperature diagnostic, all folds | 10 to 15 min | A calibration effect is consistent enough to guide design |
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

1. Commit and publish the complete Phase 2 evidence bundle.
2. Update the research report with the locked negative Stage B result.
3. Implement the frozen temperature diagnostic with tests and a 15-minute limit.
4. Decide between calibration repair and residual ranking from that diagnostic.
5. Run D1 before authorizing any larger training job.

This sequence protects the deadline, the sealed target boundary, and the credibility of the final report.
