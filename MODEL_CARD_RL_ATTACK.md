# Model Card: Frozen Cross-Victim RL Attack Policy

## Intended use

This repository is a defensive research harness for studying whether a recurrent black-box attack policy trained on public source classifiers generalizes to held-out victim families. Use it only on models and APIs you own or are authorized to evaluate.

## Primary setting

- T1 score-based black-box attack.
- The policy may update its recurrent hidden state within one image episode.
- Policy parameters, optimizer state, observation normalization, action catalog, and hyperparameters remain frozen on held-out victims.
- The target initialization call and every subsequent call count toward the total query budget.

T0 query-free, T2 label-only, and T3 limited target adaptation must be reported separately. T3 is a comparison, not evidence of zero-update policy transfer.

## Inputs and outputs

Inputs are raw images in `[0, 1]`, true labels, allowed score feedback, and an audited remaining-query budget. Actions are signed patch primitives projected centrally into the configured L-infinity ball. Results include per-image success, first-success query, all target calls, distortion, action trace, victim family, seed, and policy digests.

## Training and evaluation boundaries

Source victims are sampled by architecture family. Outer-test families may not affect training, source validation, early stopping, reward design, architecture choices, or hyperparameters. The full study uses nested leave-one-family-out evaluation; the local synthetic smoke run is explicitly not research-valid.

## Limitations

- The included offline smoke victims are toy models and cannot establish transferability.
- The DQN implementation is a pilot/baseline; recurrent PPO plus family-robust population training is the primary method.
- Square Attack, SimBA-DCT, ImageNet execution, robust-model checkpoints, LPIPS, and hierarchical statistical analysis require optional dependencies and compute not bundled with the smoke run.
- A fixed action pattern may match a learned policy; action-frequency and history-shuffling controls are therefore mandatory before making an RL-specific claim.

## Responsible release

Publish evaluation code, aggregate analysis, limitations, costs, and query accounting. Replace raw sample and victim identifiers with opaque study IDs before sharing traces. Distribution of pretrained attack policies should undergo institutional dual-use review.
