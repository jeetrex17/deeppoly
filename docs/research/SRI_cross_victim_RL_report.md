# Abstract

This report investigates whether a reinforcement-learning attack policy can be trained on several source image classifiers, frozen, and then reused against a classifier from an unseen architecture family without updating the policy parameters. This differs from conventional adversarial-example transfer: the transferred object is a sequential attack policy rather than one perturbation produced for a surrogate model. The main threat model is an untargeted, score-based black-box attack with a strict total budget of 25 target calls, including initialization, and a raw-pixel L-infinity perturbation limit of 8/255.

The project progressed from a two-victim DQN prototype that supports both frozen transfer and cloned continual adaptation to a recurrent actor-critic policy trained with Proximal Policy Optimization (PPO) across source-victim populations. The final exploratory CIFAR-10 study used three victim families - a residual classical CNN, a depthwise modern CNN, and a patch transformer - in leave-one-family-out evaluation. For each held-out family and each of three fresh seeds (17, 29, and 41), the policy trained only on the other two families. Two independently initialized models represented each source family. The target policy was then evaluated without parameter updates against random action selection, an episode-local score bandit, and a custom patch-based score-greedy attack. Across the complete nine-run study, 5,400 policy episodes were scheduled, 3,640 clean-correct sequences were trainable, 91,800 source-model calls were recorded, and 1,859 clean-correct target image/run cases were evaluated. The complete run took 114.4 minutes on an Apple M4 using PyTorch MPS.

All victim-quality gates passed, which rules out the most direct explanation that the attack failed only because the target classifiers were unusable. Nevertheless, stochastic PPO achieved mean final attack success rates of only 1.45%, 1.01%, and 1.65% on held-out classical CNN, modern CNN, and transformer targets, respectively. Random action selection achieved 1.74%, 1.29%, and 1.84%, while score-greedy achieved 13.55%, 15.49%, and 27.20%. The PPO executed-action histogram retained normalized entropy between 0.988 and 0.994 across individual runs, above the prespecified 0.95 ceiling and close to uniform behavior. The fail-closed promotion rule therefore rejected every held-out family.

The result does not show that adversarial transfer is impossible, nor that reinforcement learning can never support transferable attacks. It shows that this specific dense-reward, GroupDRO-weighted recurrent PPO design did not learn a statistically and practically validated cross-victim advantage under the evaluated CIFAR-10 setting. The principal contribution of the completed work is therefore a reproducible, leakage-resistant evaluation harness and an informative negative result. The next decisive experiment is to test whether the policy can first outperform random actions on the source victims, then add behavioral-cloning pretraining from the strong greedy trajectories before repeating cross-family evaluation.

**Keywords:** adversarial examples; black-box attacks; reinforcement learning; PPO; cross-victim transfer; CIFAR-10; query efficiency; continual learning

# Contents

1. Introduction and research question
2. Background and related work
3. Project scope and evolution
4. Proposed and implemented method
5. Experimental protocol
6. Results
7. Discussion
8. Limitations and threats to validity
9. Responsible research considerations
10. Recommended next experiments
11. Conclusion
Appendix A. Reproducibility and artifacts
Appendix B. Implementation map
References

# 1. Introduction and Research Question

## 1.1 Motivation

Image classifiers can be made to change their predictions after small, carefully chosen input perturbations [1, 2]. In a white-box setting, an attacker can compute gradients through the victim. In a black-box setting, the attacker must instead use queries, transferred examples, or a learned procedure. This project focuses on score-based black-box access: the attacker receives class probabilities but never receives victim gradients, weights, intermediate features, training data, or architecture metadata.

Most transfer-attack studies ask whether an adversarial image produced against one model also fools another model. That is adversarial-example transfer. The present project asks a different question: can a reusable sequential decision policy learn attack behavior from several source victims and then operate directly against a new victim? A successful policy could amortize learning over many models and might expose response patterns that persist across architectural changes. A failed policy would still be scientifically informative because it would show where model-specific learning or an unsuitable state/action design prevents reuse.

## 1.2 Primary research question

The primary question is:

> Can one recurrent policy, trained across source classifier families and then frozen, achieve better attack-success/query performance than matched non-learning controls when deployed on completely held-out architecture families?

The corresponding working hypothesis was that family-balanced population training, recurrent query-context inference, calibration-resistant observations, and dense confidence-margin rewards would create reusable attack behavior. The study was deliberately designed so that a negative answer remained valid. No target family was allowed to influence policy optimization during its own held-out fold.

## 1.3 What exactly transfers?

The project distinguishes four settings.

| Setting | Target feedback | Parameter updates | Interpretation |
| --- | --- | --- | --- |
| T0 | None | None | Query-free adversarial-example or generator transfer |
| T1 | Scores/probabilities | None | Frozen policy reuse with recurrent in-episode context; main setting |
| T2 | Top-1 label only | None | Frozen decision-based policy reuse |
| T3 | Declared target feedback | Limited updates | Continual target adaptation; comparison only |

The completed CIFAR-10 results in this report concern T1. Recurrent hidden state changes within an episode, but the policy weights, optimizer state, normalization, action catalog, and hyperparameters remain fixed. The earlier DQN framework also implements T3 by cloning the source agent and updating only the clone on a disjoint adaptation split. That branch demonstrates the intended continual-learning mechanism at smoke-test level, but it was not included as a scientific result in the final M4 study.

## 1.4 Contributions of the completed work

The project produced five concrete contributions:

- An audited score-based attack interface that counts every target call and centrally enforces the query budget.
- A recurrent GRU actor-critic trained with PPO over balanced source-family schedules and family-level GroupDRO weights.
- Strict frozen-policy evaluation with before/after SHA-256 digests covering model parameters, optimizer state, and PPO configuration.
- A complete three-family by three-seed leave-one-family-out CIFAR-10 study with matched random, score-bandit, and score-greedy controls.
- Reproducible configurations, manifests, checkpoints, compact result files, notebooks, visual reports, and an 87.75%-covered test suite.

![Figure 1. Implemented train-freeze-deploy protocol.](figures/submission_protocol.png)

# 2. Background and Related Work

## 2.1 Adversarial and black-box attacks

For a classifier f, clean image x, and true class y, an untargeted adversarial example x_adv must satisfy both f(x_adv) != y and ||x_adv - x||_infinity <= epsilon. The challenge in a score-based black-box setting is to locate such an input with a small number of queries. The total query budget matters because a method that silently performs extra initialization, probing, or failed calls is not comparable to another method under a nominally equal budget.

SimBA demonstrated that simple coordinate or basis-direction proposals can be a strong score-based black-box baseline [3]. It repeatedly proposes a direction and retains changes that improve the attack objective. That principle directly motivated the custom score-greedy control in this project: proposals are randomized per image, move one signed channel/patch direction to the L-infinity boundary, and are retained only when they reduce the true-class confidence margin.

## 2.2 Reinforcement learning for attack search

PPO optimizes a clipped policy-gradient surrogate while learning a value estimate, allowing multiple optimization epochs over collected trajectories [4]. It is attractive here because attack generation is sequential and partially observed: each score response can reveal something about the unknown victim. The recurrent policy can update its hidden state from these transitions while keeping persistent parameters frozen during deployment.

RL-based attacks already exist, so this report does not claim the first use of RL for adversarial attacks. RLAB uses an RL agent and a dual-action design to add or remove distortion in black-box attacks [6]. Adversarial Agents formulates adversarial generation as a Markov decision process and reports learning across images [7]. QTRL uses two training phases and hard-sample mining for black-box adversarial generation [10]. These systems establish that RL can learn attacks, but they do not by themselves answer the exact family-holdout, zero-parameter-update policy-transfer question evaluated here.

## 2.3 RL-assisted adversarial transfer

RL has also been used to improve transferable adversarial examples. L2T learns transformation sequences that improve example transfer [8]. SMER uses reinforcement learning to reweight surrogate ensembles and exploit ensemble diversity [9]. These are important related methods, but their transferred object is still an adversarial example or a surrogate-optimization strategy. The present project instead deploys the same persistent attack policy directly against the held-out victim and lets it react only through allowed black-box responses.

Recent benchmarking work emphasizes that attack categories, hyperparameters, threat models, and imperceptibility constraints must be aligned before methods are compared [11, 12]. This motivated separate T0/T1/T2/T3 terminology, a shared target interface, identical eligible sample identities, equal query endpoints, and fail-closed validation of every aggregate.

## 2.4 Family robustness

GroupDRO reweights predefined groups toward those with larger loss, aiming to improve worst-group behavior rather than only average behavior [5]. In this project the groups are source architecture families. Each block computes family-level episode returns, converts poor returns into higher GroupDRO loss, and updates family weights multiplicatively. Model instances remain nested within their families so that multiple instances do not accidentally turn one family into several independent groups.

## 2.5 Narrowed research gap

The defensible gap is not "RL has never been used for transferable attacks." The narrower question is whether one recurrent direct-attack policy can be population-trained, completely frozen, and evaluated on held-out architecture families under a leakage-resistant score-based protocol. The literature review in the research plan found close work but not an exact match as of 22 July 2026. This remains a provisional "to our knowledge" statement and must be checked again before any external paper submission.

# 3. Project Scope and Evolution

## 3.1 Initial continual-transfer prototype

The first implementation used a DQN attack agent with replay memory, a target network, epsilon-greedy exploration, and signed patch actions. It was designed around two modes requested at the start of the project:

1. **Frozen transfer:** train on a source victim, store a source-policy digest, and evaluate the same policy on a target victim without learning.
2. **Continual transfer:** clone the exact source policy, train the clone on a declared target adaptation subset, and evaluate it on a separate target subset.

The continual branch preserves the original source agent and both victim models, and it records whether the clone began from the source-policy digest. This is an important implementation distinction: continual adaptation must not overwrite the source artifact, and frozen transfer must not be mislabeled when target updates occur.

## 3.2 Recurrent frozen-policy research harness

The main research plan shifted the primary method from the DQN prototype to a recurrent PPO policy. The reason was conceptual as well as practical. A recurrent state can infer victim-response behavior from the sequence of scores without updating weights. The harness added calibrated observations, population training, GroupDRO, per-image traces, target-call auditing, central L-infinity projection, checkpoint hashing, and a formal T1 frozen boundary.

## 3.3 Apple M4 pilot

The first end-to-end CIFAR-10 pilot trained the recurrent policy on classical and modern CNN source families and held out a patch transformer. It ran for 6 minutes 12 seconds on MPS. The policy was frozen correctly, but stochastic PPO reached 5.1% ASR (5/99) while random reached 7.1% (7/99). This pilot exposed three problems: one seed was insufficient, the target accuracy was only 49.5%, and a stronger score-adaptive baseline was missing.

## 3.4 Corrected three-seed study

The next implementation strengthened victim architectures and fitting, added two source instances per family, replaced the sparse reward with dense confidence-margin shaping, added episode-local UCB bandit and score-greedy controls, corrected training offsets across blocks, and introduced a prespecified promotion gate. The final study used a clean Git revision and fresh seeds 17, 29, and 41. Earlier diagnostic outputs superseded by these fixes were excluded from the final claims.

| Milestone | Repository commit | Outcome |
| --- | --- | --- |
| Continual DQN framework | `9605325` | Frozen and cloned-adaptation protocols implemented |
| Frozen recurrent harness | `16bfae6` | Population PPO, audited T1 evaluation, research configs |
| M4 pilot execution | `e0603e1` | End-to-end feasibility result recorded |
| Corrected cross-victim study | `2eb481a` | Dense reward, stronger victims, controls, gates, reproducibility |
| Results publication | `9453730` | Verified reports, figures, and compact artifacts committed |

# 4. Proposed and Implemented Method

## 4.1 Victim population

The research profile contains three CIFAR-10 victim families:

| Family | Implemented architecture | Research profile |
| --- | --- | --- |
| Classical CNN | Residual CNN | Widths 64/128/256; two residual blocks per stage; dropout 0.1 |
| Modern CNN | Depthwise/inverted-residual CNN | Widths 48/72/144/256; 2/2/3 blocks; expansion 3; residual connections; dropout 0.1 |
| Transformer | Patch transformer | 4x4 patch embedding; width 96; depth 3; four heads; CLS pooling; MLP ratio 4; dropout 0.1 |

Each model normalizes CIFAR-10 inputs internally, so perturbations and L-infinity projection remain in raw [0, 1] pixel space. Victims are trained with AdamW, cosine learning-rate decay, random reflected crops, random horizontal flips, cross-entropy loss, and gradient clipping. The full study uses 10,000 victim-training images, 1,000 policy-training images, 1,000 source-validation images, and 300 outer-test images per seed. Splits are class-stratified and carry SHA-256 digests.

For a run that holds out one family, the other two families each contribute two independently seeded model instances. The target family contributes one held-out instance. Checkpoints are reused only if the configuration, data split, fitting code, device backend, training seed, and cache contract match.

## 4.2 State representation

At each query step, the policy receives an eight-dimensional observation:

- normalized rank of the true class;
- normalized predictive entropy;
- normalized change in the true-class score from initialization;
- current true-class margin over the strongest rival;
- fraction of the total query budget remaining;
- normalized previous-action identifier;
- hyperbolic tangent of the previous reward;
- fraction of the episode already consumed.

This representation avoids victim layers and gradients. It uses probabilities, ranks, and normalized changes so that the policy is less dependent on one model's score calibration.

## 4.3 Action space and perturbation constraint

The 32x32 image is partitioned into a 4x4 grid. Each action selects one of 16 patches, one of three color channels, and one sign, yielding 96 discrete actions. The selected patch/channel changes by 2/255 per step. After every action, the candidate is projected into the raw-pixel L-infinity ball with epsilon = 8/255 and clamped to [0, 1].

The score-greedy baseline uses the same 96 patch primitives, but proposes each action directly at the 8/255 boundary and retains it only when the confidence margin improves. Thus, its advantage does not come from a different spatial representation; it comes from a more effective query-driven decision rule.

## 4.4 Recurrent actor-critic and PPO

The policy consists of a linear encoder with tanh activation, a GRUCell with hidden size 96, an actor head over 96 actions, and a scalar critic. PPO uses learning rate 3e-4, clipping ratio 0.2, value-loss weight 0.5, entropy weight 0.01, gradient clipping at 0.5, and four update epochs per block. Returns use discount factor 0.98. Advantages are normalized across collected source-family sequences before each update.

The recurrent hidden state starts at zero for each image. During target evaluation it changes after each observation, but it is ephemeral. The persistent policy digest covers the model state, optimizer state, and PPO configuration; evaluation raises an error if that digest changes.

## 4.5 Dense reward and family-level GroupDRO

Let m_t be the true-class probability minus the largest rival probability after step t. The implemented dense reward is:

`r_t = 5 (m_(t-1) - m_t) + 2 I[success] - 0.01.`

The one-step margin difference telescopes across the episode, so repeated reward requires continued progress rather than merely remaining below the clean confidence. The success bonus favors actual misclassification, while the query penalty makes earlier success preferable.

Source families are scheduled in balanced randomized cycles. Within each family, model instances rotate across episodes. After a block, GroupDRO weights are updated using family loss derived from negative mean return. Sequence weights are normalized by both family weight and the number of trainable sequences from that family.

## 4.6 Baselines

The full study reports six evaluation methods, but the promotion gate focuses on stochastic recurrent PPO and three controls:

| Method | Target feedback | Persistent learning during evaluation | Purpose |
| --- | --- | --- | --- |
| Stochastic recurrent PPO | Scores | None | Main learned policy |
| Random action | Same observation stream | None | Detects near-uniform or ineffective learning |
| Score bandit | Previous reward | Episode-local only | Tests lightweight online action adaptation |
| Score greedy | Current candidate margin | Episode-local only | Strong query-matched patch-search control |

Deterministic PPO and a fixed-action policy were also recorded, but they were not used to rescue the main hypothesis after the stochastic policy failed.

# 5. Experimental Protocol

## 5.1 Leave-one-family-out design

The complete grid contains three held-out target families multiplied by three fresh seeds. In each fold, policy training sees only the two source families. A separate policy checkpoint is trained for every family/seed fold, yielding nine policy checkpoints rather than one universal checkpoint evaluated against all targets. Target evaluation begins after the fold-specific policy is frozen, and the target instance does not influence reward design, early stopping, architecture selection, or hyperparameter selection within that fold.

| Study dimension | Final setting |
| --- | --- |
| Dataset | CIFAR-10 |
| Target families | Classical CNN, modern CNN, transformer |
| Seeds | 17, 29, 41 |
| Source instances | Two independently initialized victims per source family |
| Target instance | One held-out victim per family/seed run |
| Victim training | 10,000 images; 12 epochs; AdamW; augmentation |
| Policy training | 1,000 images; 600 scheduled episodes per run |
| Target evaluation | 300 stratified test images per run; ASR denominator is clean-correct images |
| Threat model | T1 score-based, untargeted, no target-side parameter updates |
| Total query budget | 25 calls including initialization |
| Perturbation | Raw-pixel L-infinity epsilon = 8/255; step = 2/255 |
| Hardware | Apple M4, PyTorch MPS |

## 5.2 Query and denominator alignment

Every method attacks the same 300 test indices for a given run. Attack success rate is computed only over images the held-out victim classifies correctly before attack. A digest of those eligible sample identifiers must match across all methods. The ASR curve is computed from a single full trajectory at budgets 0, 5, 10, and 25 rather than rerunning methods separately at each budget. Initialization is counted as the first target call, leaving at most 24 perturbation proposals, and the maximum recorded target calls may not exceed 25.

## 5.3 Victim-quality gates

Before an attack run can support promotion, every trained instance must clear its family gate: 60% source-validation accuracy for the classical CNN, 50% for the modern CNN, and 40% for the transformer. These thresholds are modest and the resulting models are not state of the art; their purpose is to reject obviously unusable victims in a bounded Mac experiment.

## 5.4 Metrics and promotion rule

The main metrics are final ASR at 25 total calls and ASR/query AUC across the recorded budgets. For each target family, the study calculates Student-t 95% confidence intervals across the three aligned seeds and paired intervals for stochastic PPO minus each control.

The fail-closed promotion rule requires all of the following:

- the exact three-family by three-seed grid is complete;
- every victim-quality gate passes;
- for random, score-bandit, and score-greedy separately, the paired 95% lower bounds for both ASR and AUC differences are positive;
- mean gains are at least 1 percentage point in final ASR and 0.5 percentage points in AUC;
- normalized PPO action entropy for every seed lies between 0.10 and 0.95.

This gate is deliberately strict. It prevents a small unpaired mean difference, a missing fold, a bad victim, or an almost-uniform policy from being presented as evidence of transferable RL behavior.

## 5.5 Reproducibility and integrity controls

The runner records the Git revision, package-code digest, data-split digest, config digest, PyTorch version, platform, device, victim training seeds, checkpoint checksums, and policy digest. It aborts if package code changes during a study. The clean full run used Git revision `2eb481a`, Python 3.13.11, PyTorch 2.13.0, torchvision 0.28.0, macOS 15.7.3 arm64, and MPS. Victim-cache writes use file locks and atomic sidecars; model loads use `weights_only=True` and verify SHA-256 checksums. The final result report recomputes the aggregate and promotion gate from per-run records and records whether the original manifest summary matches.

The verified codebase completed 76 tests with 87.75% line coverage. Tests cover reward calculations, action projection, query accounting, DQN cloning, frozen digests, recurrent PPO updates, source-family scheduling, checkpoint safety, CIFAR splitting, victim building, study aggregation, promotion failure modes, report generation, and MPS integration. PyTorch warned that one MPS operation lacks a deterministic implementation even when deterministic algorithms are requested in warning mode; exact checkpoint hashes therefore identify the artifacts more reliably than seeds alone.

# 6. Results

## 6.1 Pilot result

The initial transformer-holdout pilot established that the entire pipeline could run on the M4. It scheduled 400 policy episodes, obtained 205 trainable episodes, made 5,261 source-model calls, and evaluated 99 clean-correct transformer images. Stochastic PPO achieved 5.1% final ASR, compared with 7.1% for random. This did not support the hypothesis, but it directly motivated the stronger victims, multiple seeds, confidence intervals, and score-adaptive controls used in the final study.

## 6.2 Final-study execution summary

| Quantity | Verified value |
| --- | --- |
| Complete family/seed runs | 9 |
| Scheduled policy episodes | 5,400 |
| Trainable clean-correct sequences | 3,640 |
| Source-model calls | 91,800 |
| Clean-correct target image/run cases | 1,859 |
| Distinct victim instances across seeds | 18 |
| Wall-clock time | 6,864.9 s = 114.4 min |
| Victim-quality gates | 9/9 runs passed |
| Promotion result | Failed for all three target families |

## 6.3 Victim quality

Victim validation accuracies were stable and above their gates. Target-test accuracy varied because each run used a seed-specific stratified outer test subset and independently trained victim.

| Victim family | Validation accuracy range | Gate | Target-test accuracy range |
| --- | --- | --- | --- |
| Classical CNN | 73.4% to 74.4% | 60% | 73.3% to 78.7% |
| Modern CNN | 73.3% to 76.2% | 50% | 70.7% to 79.3% |
| Transformer | 51.1% to 56.3% | 40% | 53.0% to 58.3% |

![Figure 2. Validation and target-test accuracy for all held-out runs.](figures/submission_victim_accuracy.png)

## 6.4 Final attack success rates

The final ASR table reports the across-seed mean and Student-t 95% interval.

| Held-out target | Stochastic PPO | Random | Score bandit | Score greedy |
| --- | --- | --- | --- | --- |
| Classical CNN | 1.45% [0.00, 4.05] | 1.74% [0.00, 4.43] | 1.17% [0.04, 2.30] | 13.55% [6.65, 20.45] |
| Modern CNN | 1.01% [0.00, 2.59] | 1.29% [0.00, 4.95] | 2.50% [0.72, 4.28] | 15.49% [7.05, 23.92] |
| Transformer | 1.65% [0.00, 6.29] | 1.84% [0.00, 5.90] | 4.02% [1.83, 6.21] | 27.20% [18.61, 35.79] |

![Figure 3. Final frozen attack success rate by held-out victim family.](figures/submission_final_asr.png)

The stochastic policy did not establish an advantage over random in any family. It exceeded the score-bandit mean by only 0.28 percentage points on classical CNN targets, but the paired 95% interval crossed zero and the practical-gain threshold was not met. On modern CNN and transformer targets, the score bandit was stronger. Score-greedy was dramatically stronger in every family. Pooled descriptively across all nine runs, stochastic PPO succeeded on 25/1,859 clean-correct image/run cases, compared with 30/1,859 for random, 45/1,859 for bandit, and 333/1,859 for score-greedy; the equal-seed family aggregates above remain the appropriate primary comparison.

## 6.5 ASR/query AUC

| Held-out target | Stochastic PPO | Random | Score bandit | Score greedy |
| --- | --- | --- | --- | --- |
| Classical CNN | 0.81% | 0.96% | 0.79% | 6.52% |
| Modern CNN | 0.53% | 0.62% | 1.48% | 7.17% |
| Transformer | 0.78% | 0.91% | 2.33% | 13.19% |

The AUC result reinforces the final-ASR result. The learned policy was not merely slower and catching up by query 25; it also failed to improve cumulative success over the available query trajectory.

![Figure 4. Mean ASR as the total target-call budget increases.](figures/submission_asr_curves.png)

## 6.6 Per-run outcomes

| Target | Seed | Eligible | PPO successes | Random | Bandit | Greedy |
| --- | --- | --- | --- | --- | --- | --- |
| Classical CNN | 17 | 222 | 3 | 3 | 2 | 23 |
| Classical CNN | 29 | 236 | 6 | 7 | 4 | 35 |
| Classical CNN | 41 | 220 | 1 | 2 | 2 | 34 |
| Modern CNN | 17 | 234 | 4 | 7 | 7 | 45 |
| Modern CNN | 29 | 238 | 2 | 1 | 4 | 30 |
| Modern CNN | 41 | 212 | 1 | 1 | 6 | 31 |
| Transformer | 17 | 159 | 2 | 2 | 5 | 49 |
| Transformer | 29 | 175 | 0 | 1 | 7 | 47 |
| Transformer | 41 | 163 | 6 | 6 | 8 | 39 |

## 6.7 Action entropy and promotion decision

Normalized stochastic-PPO executed-action entropy ranged from 0.988 to 0.994 across runs. Because 1.0 denotes an approximately uniform executed-action histogram, this shows that the policy did not specialize its action selection enough to pass the prespecified diagnostic ceiling of 0.95. This metric describes empirical actions, not the exact categorical entropy of the actor logits. All three families also failed the paired RL-versus-control criteria.

![Figure 5. Stochastic-PPO action entropy by family and seed.](figures/submission_policy_entropy.png)

The failure is not an administrative artifact: the family/seed grid was complete, the result manifest passed internal recomputation, all target methods remained frozen, query and sample identities aligned, and all victims cleared their gates. The promotion gate failed because the empirical behavior did not support the hypothesis.

# 7. Discussion

## 7.1 Main interpretation

The current PPO agent did not learn a transferable advantage. Its final ASR was approximately random, its query-AUC was approximately random, and its action entropy was close to uniform. These three observations point to a learning or representation failure rather than a subtle loss of transfer at deployment.

The stronger score-greedy result is especially informative. All methods used the same patch catalog, perturbation bound, held-out images, and total target-call budget. Greedy's improvement therefore suggests that the local action space contains useful attacks, but the PPO policy did not learn to select them from its observation history. On transformer targets the gap was largest: 1.65% mean ASR for PPO versus 27.20% for score-greedy.

## 7.2 Why more compute is not yet the best response

The victims were adequate for a bounded exploratory experiment, and the complete study already used 91,800 source calls. Simply increasing episode count might reduce optimization noise, but there is no present evidence that the learned policy is moving away from uniform action selection. Scaling the same design to ImageNet would multiply cost before resolving whether PPO can even overfit the source attack task.

The first diagnostic should evaluate each trained policy on (a) its exact source instances, (b) unseen instances from a seen family, and (c) the held-out family. If PPO fails on its training victims, then transfer is not the bottleneck. If it succeeds on source victims but fails within-family holdouts, the policy is instance-specific. If it succeeds within family but fails across family, the project has isolated a genuine architecture-shift boundary.

## 7.3 Likely technical bottlenecks

Several mechanisms could explain the observed entropy and low ASR:

- The 96-way flat action space may be too large for the number of informative source trajectories.
- The observation compresses image and action history into eight values and contains no visual feature representation of which regions are promising.
- PPO receives many weak trajectories because only clean-correct source episodes are trainable and successful source attacks are rare.
- The entropy bonus may be too strong relative to the small, noisy advantage signal.
- Dense margin reduction improves local credit assignment but does not teach the policy the strong accept/reject behavior used by score-greedy.
- Several upgrades were changed together, so the study cannot attribute behavior to reward shaping, victim diversity, recurrence, or GroupDRO separately.

These are hypotheses, not conclusions. Matched ablations are required.

## 7.4 Scientific value of the negative result

A negative result is useful when the protocol is strong enough to rule out easy explanations. Here, policy mutability, query mismatch, inconsistent denominators, incomplete seed grids, and bad victim gates were explicitly checked. The result provides a defensible go/no-go signal: do not claim a transferable RL method yet, and do not scale the unchanged agent. Preserve the harness, use it to falsify improved designs, and report only a positive claim if the prespecified controls are beaten.

# 8. Limitations and Threats to Validity

The following limitations prevent a broad or paper-level conclusion:

- **Dataset scale:** Only CIFAR-10 was executed. The research plan identifies ImageNet-1K as the eventual main benchmark.
- **Victim population:** The victims are custom compact models rather than established pretrained architectures such as VGG, ResNet, ConvNeXt, ViT, and Swin checkpoints.
- **Target multiplicity:** Each family/seed run uses one held-out target instance. Architectural generalization should ultimately treat multiple target models and families as the statistical units.
- **Number of seeds:** Three fresh seeds support an exploratory interval but fall below the five-seed paper plan.
- **Query range:** Only 25 total target calls were studied. The full plan calls for 0, 25, 100, and 500-call blocks.
- **Attack coverage:** The implemented controls include random, bandit, and a custom patch-based SimBA-style greedy method, not the complete Square Attack, SimBA-DCT, NES, transfer, and matched-generator suite.
- **Threat-model coverage:** The reported study covers T1 only. T0, T2, and scientific T3 experiments remain future work.
- **Ablation coverage:** Reward, victim capacity, victim multiplicity, controls, and some reproducibility logic changed between pilot and full study. Their individual causal effects are unknown.
- **Perceptual metrics:** L-infinity and L2 values are recorded, but LPIPS and a full perceptual-quality analysis were not run.
- **MPS reproducibility:** Deterministic algorithms are requested in warning mode, but at least one PyTorch MPS operation is nondeterministic.
- **Statistical depth:** Student-t intervals over three seeds are reported. Hierarchical bootstrap, paired permutation tests, and multiplicity correction from the full plan were not yet appropriate for this small target population.

These limitations mean the correct conclusion is "the current agent failed in this bounded setting," not "RL policies cannot transfer" and not "score-greedy is universally superior."

# 9. Responsible Research Considerations

This repository is intended for defensive robustness research. Experiments should use public benchmark models or systems the researcher owns or is explicitly authorized to test. Target rate limits, monetary costs, and terms of service must be respected. The current local experiments do not query third-party services.

The release boundary should prioritize evaluation code, query accounting, aggregate analysis, limitations, and reproducibility metadata. Raw sample or victim identifiers should be replaced with opaque study IDs before traces are shared. Distribution of pretrained attack policies should receive institutional dual-use review because a reusable cross-victim policy could lower the cost of attacking unknown deployed models.

The result itself should be communicated without sensationalism. The current evidence does not establish a strong attack; it establishes a careful procedure and reveals that the tested RL policy behaved nearly randomly.

# 10. Recommended Next Experiments

## 10.1 Stage 1: learning diagnosis

Add a three-level evaluation for every trained checkpoint:

1. Evaluate on the exact source instances used during training.
2. Evaluate on new independently initialized instances from the same source families.
3. Evaluate on the held-out family as today.

Do not start another nine-run study unless the policy clearly beats random on its own source victims, reduces confidence margin over the episode, and moves below the 0.95 entropy ceiling.

## 10.2 Stage 2: behavioral cloning from greedy trajectories

The score-greedy control provides successful source-victim demonstrations under the same action basis. Generate those trajectories only on source victims, pretrain the recurrent actor to imitate accepted greedy decisions, and then fine-tune with PPO. This tests whether the network and observation pipeline can represent a useful strategy before policy-gradient optimization is asked to discover it from sparse success.

Suggested gates before full cross-family evaluation are:

- imitation action accuracy materially above the 1/96 random level;
- source-victim ASR at least 5 percentage points above random at 25 calls;
- positive paired source-victim AUC difference;
- action entropy below 0.95 without collapsing to one fixed action;
- continued success on unseen instances from a seen family.

## 10.3 Stage 3: hierarchical action design

Replace one 96-way decision with a hierarchy: select region, select channel/sign, then select magnitude. Mask invalid or repeatedly ineffective actions. Add image-region features or a lightweight visual encoder so the policy can associate score changes with where it has perturbed the image. Compare this against the current flat action catalog with matched parameter count and source calls.

## 10.4 Stage 4: matched ablations

Change one factor at a time:

- behavioral cloning only versus cloning plus PPO;
- recurrent versus feed-forward policy;
- raw scores versus ranks and normalized deltas;
- uniform family weighting versus GroupDRO;
- sparse success reward versus dense margin reward;
- current patch actions versus hierarchical patches and DCT directions;
- ordered history versus shuffled history.

Only after a small diagnostic passes should the full three-seed CIFAR grid be repeated. Five seeds, multiple held-out models per family, 100/500-query budgets, established query attacks, robust models, and ImageNet belong after that gate. The MacBook M4 is sufficient for the diagnostic funnel; larger GPU compute becomes justified only after the agent demonstrates source-task learning.

# 11. Conclusion

This project implemented both conceptual branches requested at its start: a frozen transfer protocol and a cloned continual-adaptation protocol. It then developed the frozen branch into a rigorous recurrent PPO study with audited target access, held-out architecture families, victim populations, reproducibility contracts, matched controls, and prespecified promotion rules.

The final M4 study is complete and technically valid as an exploratory result. It does not support the primary positive hypothesis. Stochastic GroupDRO PPO remained close to random action selection, failed every family promotion gate, and was substantially weaker than score-greedy patch search. All victim-quality gates passed, the complete evaluation grid ran, and the policy remained frozen, so the negative result cannot be dismissed as a missing run or target leakage.

The most productive next step is not a larger unchanged training run. It is a decisive learning diagnosis followed by behavioral-cloning pretraining and a better action/state representation. If those changes produce source and within-family gains that survive matched controls, the same harness is ready to test genuine cross-family transfer. Until then, the responsible claim is that a careful zero-update cross-victim evaluation was built and that the current RL design did not learn a transferable advantage.

# Appendix A. Reproducibility and Artifacts

## A.1 Environment setup

```text
uv sync --extra vision --extra analysis --extra notebook --extra test
uv run pytest --cov=rl_transfer --cov-report=term --cov-fail-under=80 -q
```

## A.2 Run the diagnostic and complete study

```text
uv run python -m rl_transfer.cifar_study_cli \
  --config configs/rl_transfer/cifar10_m4_study_quick.json \
  --device auto

uv run python -m rl_transfer.cifar_study_cli \
  --config configs/rl_transfer/cifar10_m4_study.json \
  --device auto
```

## A.3 Regenerate the compact report

```text
uv run python -m rl_transfer.study_report \
  --input output/rl_transfer/cifar10_m4_studies/cifar10-m4-study/study_manifest.json \
  --output-dir docs/research
```

## A.4 Principal artifacts

- `configs/rl_transfer/cifar10_m4_iteration.json`: per-run victim, policy, reward, and attack settings.
- `configs/rl_transfer/cifar10_m4_study.json`: target-family grid, fresh seeds, and promotion thresholds.
- `docs/research/cifar10_m4_study_results.json`: verified compact per-run results and recomputed aggregate.
- `docs/research/cifar10_m4_study_summary.md`: generated visual result summary.
- `notebooks/cifar10_m4_study.ipynb`: interactive inspection of folds, seeds, and promotion gates.
- `MODEL_CARD_RL_ATTACK.md`: intended use, threat model, limitations, and responsible-release guidance.
- `docs/research/rl_cross_victim_research_plan.pdf`: full research and publication plan.

# Appendix B. Implementation Map

| Component | File | Responsibility |
| --- | --- | --- |
| Audited target access | `rl_transfer/audit.py` | Enforces feedback type, total calls, and query trace |
| Attack actions | `rl_transfer/actions.py` | Patch/DCT catalogs and raw-pixel L-infinity projection |
| Recurrent policy | `rl_transfer/recurrent.py` | GRU actor-critic, PPO update, persistent digest |
| Population training | `rl_transfer/research_protocol.py` | Balanced families, victim rotation, GroupDRO, frozen episodes |
| Victim models | `rl_transfer/cifar_models.py` | Residual CNN, depthwise CNN, patch transformer |
| CIFAR runner | `rl_transfer/cifar_pilot.py` | Splits, victim fitting, checkpointing, evaluation |
| Multi-seed study | `rl_transfer/cifar_study.py` | Leave-one-family-out grid, intervals, fail-closed gate |
| Baselines | `rl_transfer/baselines.py` | Fixed, random, and episode-local score bandit |
| Continual DQN protocols | `rl_transfer/protocols.py` | Frozen source policy and cloned target adaptation |
| Report generation | `rl_transfer/study_report.py` | Aggregate recomputation, compact JSON, Markdown, SVG |

# References

[1] C. Szegedy et al. "Intriguing Properties of Neural Networks." ICLR, 2014. https://arxiv.org/abs/1312.6199

[2] I. J. Goodfellow, J. Shlens, and C. Szegedy. "Explaining and Harnessing Adversarial Examples." ICLR, 2015. https://arxiv.org/abs/1412.6572

[3] C. Guo, J. Gardner, Y. You, A. G. Wilson, and K. Q. Weinberger. "Simple Black-box Adversarial Attacks." ICML, PMLR 97:2484-2493, 2019. https://proceedings.mlr.press/v97/guo19a.html

[4] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347

[5] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang. "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization." ICLR, 2020. https://openreview.net/forum?id=ryxGuJrFvS

[6] S. Sarkar et al. "Reinforcement Learning Platform for Adversarial Black-box Attacks with Custom Distortion Filters." AAAI 39(26):27628-27635, 2025. https://doi.org/10.1609/aaai.v39i26.34976

[7] K. Domico, J.-C. Noirot Ferrand, R. Sheatsley, E. Pauley, J. Hanna, and P. McDaniel. "Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning." arXiv:2503.01734, 2025. https://arxiv.org/abs/2503.01734

[8] R. Zhu, Z. Zhang, S. Liang, Z. Liu, and C. Xu. "Learning to Transform Dynamically for Better Adversarial Transferability." CVPR, pp. 24273-24283, 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Zhu_Learning_to_Transform_Dynamically_for_Better_Adversarial_Transferability_CVPR_2024_paper.html

[9] B. Tang, Z. Wang, Y. Bin, Q. Dou, Y. Yang, and H. T. Shen. "Ensemble Diversity Facilitates Adversarial Transferability." CVPR, pp. 24377-24386, 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Ensemble_Diversity_Facilitates_Adversarial_Transferability_CVPR_2024_paper.html

[10] Z. Ma and T. Feng. "Query-Efficient Two-Phase Reinforcement Learning Framework for Black-Box Adversarial Attacks." Symmetry 17(7):1093, 2025. https://doi.org/10.3390/sym17071093

[11] X. Wang et al. "Devling into Adversarial Transferability on Image Classification: Review, Benchmark, and Evaluation." arXiv:2602.23117, 2026. https://arxiv.org/abs/2602.23117

[12] Z. Zhao et al. "Revisiting Transferable Adversarial Image Examples: Attack Categorization, Evaluation Guidelines, and New Insights." arXiv:2310.11850, 2023. https://arxiv.org/abs/2310.11850

[13] A. Krizhevsky. "Learning Multiple Layers of Features from Tiny Images." Technical report, 2009. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
