# Transferability of RL-Based Black-Box Adversarial Attacks on Image Classifiers

## Research Proposal

### Core Question
Do RL-generated adversarial attack policies generalize across different neural network architectures (CNN → ResNet → ViT), or do they overfit to specific architectures?

### Why This Matters
Transferability of adversarial examples is well-studied for gradient-based attacks. RL has also been used for attack augmentation, surrogate weighting, and per-victim black-box attackers. The narrower open question is whether one recurrent direct-attack policy can be trained across source victim families, frozen, and reused without parameter updates on completely held-out architecture families.

### Two required conditions

The experiment separates two meanings of “transfer”:

1. **T1 frozen score-based transfer (primary):** train a recurrent PPO policy across a population of source victim families, freeze its complete persistent state, and evaluate it on an outer held-out family. Its GRU hidden state may change within an episode; parameters, optimizer, normalization, action catalog, and hyperparameters may not.
2. **T3 limited adaptation (comparison):** clone the exact frozen checkpoint and permit a declared target update budget on a disjoint adaptation split. Re-evaluate both source and target families to quantify target gain and source forgetting.

Both branches use the same clean-correct evaluation cohort, raw-pixel L-infinity threat model, query budget, deterministic signed patch actions, and policy/victim state digests. A negative transfer result is valid; the protocol, not a preselected success rate, is the research result.

### Methodology

**Phase 1 - Register victims and lock family holdouts**
- CIFAR-10/100 for development; ImageNet-1K for the main result
- Classical CNN, modern CNN, transformer, and hierarchical-transformer families
- Nested leave-one-family-out evaluation with separate source-family validation

**Phase 2 - Audited black-box environment**
- T1 observations: ranks, entropy, normalized score deltas, remaining budget, and action history
- Architecture-independent patch and DCT action catalogs
- Raw-pixel L-infinity projection after every action
- Initialization and failed calls included in the total query budget

**Phase 3 - Population-train the recurrent attacker**
- Recurrent PPO with hidden-state victim-context inference
- Balanced source-family episodes
- Naive pooled and GroupDRO objectives as paired variants
- Feed-forward DQN retained as an RLAB-style baseline

**Phase 4 - Frozen held-out-family evaluation**
Freeze the complete policy bundle and run it directly on victims from the outer held-out family:

| Metric | Reporting rule |
|--------|----------------|
| ASR/query AUC | Macro-average by target model, then family |
| ASR at fixed budgets | 25, 100, and 500 total calls with 95% intervals |
| Queries to success | Treat failures as censored |
| Distortion | L-infinity, L2, LPIPS, and valid-image rate |

**Phase 5 - Baselines and falsification tests**
- Fixed/random/greedy strategies, matched feed-forward policy, DQN, Square Attack, and SimBA-DCT
- Recurrent versus feed-forward versus shuffled-history controls
- Single-source versus pooled versus GroupDRO training
- T0, T1, T2, and T3 reported as separate threat-model blocks

### Expected Contributions
1. A leakage-resistant study of zero-update cross-victim generalization for a recurrent direct-attack policy
2. Analysis of whether RL policies discover architecture-agnostic or architecture-specific perturbation strategies
3. Query-efficiency comparison across transfer settings
4. Open-source codebase for reproducible RL adversarial attack research
