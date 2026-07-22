# DeepPoly

Adversarial robustness certification for neural networks using the DeepPoly abstract domain (Singh et al., POPL 2019).

This repo implements two approaches: **certified training** (DeepPoly bound propagation) and **adversarial game training** (PGD + DeepPoly verification). Also includes research on attack transferability, speech attacks, and zero-knowledge proof integration for certification.

## Files

### Core certification

- **`deeppoly_mnist.ipynb`** — Trains an MLP on MNIST with a DeepPoly robustness loss. Compares certified accuracy of a baseline (cross-entropy only) vs a DeepPoly-trained model at multiple L∞ epsilon values. Pure PyTorch implementation of the DeepPoly abstract domain (affine layer propagation, ReLU bounds with identity/zero lower-bound selection, epsilon curriculum during training).

- **`adversarial_game.py`** — Adversarial training (PGD, Madry et al. 2018) on a CNN with DeepPoly certification. Trains a CNN with PGD-7, then evaluates with PGD-40 attacks and DeepPoly formal certification. Implements a two-player zero-sum game: Defender (weights) vs Attacker (perturbations). Supports numpy-based DeepPoly verification for conv2d layers (not just FC).

### Attacks & transferability

- **`mnist_rotation_attacks.ipynb`** — Empirical rotation attacks on a simple MLP (grid search over angles, not formal certification).

- **`rl_attack_transfer.ipynb`** — DQN-based patch attack on CIFAR-10 (pretrained ResNet/ViT). Tests whether RL-discovered perturbation policies transfer across architectures (CNN → ResNet → ViT).

- **`speech_adversarial_attack.ipynb`** — PGD attack on a Hugging Face ASR model (waveform perturbation, L∞ bound, CTC loss).

### Research documents

- **`research_areas.md`** — Comprehensive research landscape covering abstract interpretation (DeepPoly, αβ-CROWN), certified training (IBP, SABR, MTL-IBP), randomized smoothing, ZKML (zkLLM, EZKL), game-theoretic certification, speech certification (PROVER), and the unexplored gap between ZK proofs and robustness certification.

- **`research_idea.md`** — Specific proposal for a ZK proof of certified accuracy: use LP certificate verification (check pre-computed DeepPoly bounds against committed weights in a ZK circuit, avoiding expensive ReLU bit-decomposition). Targets CCS/S&P venues.

- **`rl_attack_transfer_proposal.md`** — Research proposal for studying transferability of RL-based black-box adversarial attacks across CNN, ResNet, and ViT architectures.

### Papers (reference PDFs)

- **`resarch-paper/DeepPoly.pdf`** — Original DeepPoly paper (Singh et al., POPL 2019)
- **`resarch-paper/dnn-verifier.pdf`** — Neural network verification survey
- **`resarch-paper/dnn-certifier.pdf`** — Certification methods comparison
- **`resarch-paper/hidden-in-the-noise.pdf`** — Hidden-in-the-noise attack paper
- **`resarch-paper/sacred-bench.pdf`** — Sacred-bench benchmark

## Requirements

torch, torchvision, numpy, pandas, matplotlib, jupyter

For speech: transformers, soundfile, jiwer, librosa

For RL: gym, stable-baselines3 (or compatible DQN implementation)

For ZK: gnark, nova (Go), or EZKL (Python)

## References

- DeepPoly: Singh et al., POPL 2019. https://github.com/eth-sri/eran
- PGD adversarial training: Madry et al., ICLR 2018
- αβ-CROWN: Wang et al., NeurIPS 2021. https://github.com/Verified-Intelligence/alpha-beta-CROWN
- PROVER (speech certification): Ryou et al., CAV 2021
- FairProof (ZK fairness): Yadav et al., ICML 2024
