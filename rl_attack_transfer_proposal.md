# Transferability of RL-Based Black-Box Adversarial Attacks on Image Classifiers

## Research Proposal

### Core Question
Do RL-generated adversarial attack policies generalize across different neural network architectures (CNN → ResNet → ViT), or do they overfit to specific architectures?

### Why This Matters
Transferability of adversarial examples is well-studied for gradient-based attacks (FGSM, PGD, C&W), but **no prior work has tested whether RL-discovered perturbation strategies transfer across architectures**. If RL-learned attacks transfer, they pose a more practical black-box threat. If they don't, it reveals architecture-specific vulnerabilities in RL policies.

### Methodology

**Phase 1 — Train target classifiers (CIFAR-10)**
- Simple CNN (3-layer ConvNet)
- ResNet-18
- ViT-tiny

**Phase 2 — RL attack environment**
- State: current patch noise levels (64 patches = 8×8 grid) + confidence + step count
- Actions: pick which patch to perturb (64 discrete actions)
- Reward: +10 if classifier misclassifies, −0.1 per patch touched, −confidence penalty otherwise
- Episode: up to 20 steps

**Phase 3 — Train RL agent (DQN) on CNN only**
- 500 episodes, ε-greedy (1.0 → 0.01)
- Track success rate, L2 perturbation, queries per episode

**Phase 4 — Transferability testing**
Freeze the trained policy, run inference on all 3 models on held-out images:

| Metric | CNN (surrogate) | ResNet (transfer) | ViT (transfer) |
|--------|:---:|:---:|:---:|
| Attack success rate | | | |
| Avg L2 perturbation | | | |
| Avg queries used | | | |

**Phase 5 — Ablation studies**
- Vary query budget (5/10/20/50)
- Vary perturbation magnitude
- Test JPEG compression robustness

### Expected Contributions
1. First empirical study of RL-based adversarial attack transferability across CNN, ResNet, and ViT
2. Analysis of whether RL policies discover architecture-agnostic or architecture-specific perturbation strategies
3. Query-efficiency comparison across transfer settings
4. Open-source codebase for reproducible RL adversarial attack research

