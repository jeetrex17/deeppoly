# CIFAR-10 cross-victim RL study

Status: exploratory (`research_valid: false`). Runtime: **114.4 minutes**.

![Frozen ASR by held-out family](figures/cifar10_m4_study_asr.svg)

| Held-out target | Seed | Time | Val. acc. / gate | Test acc. | Eligible | RL stochastic | Random | Score bandit | Score greedy | Victim gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| classical_cnn | 17 | 27.5m | 74.2% / ≥60.0% | 74.0% | 222 | 3/222 (1.4%) | 3/222 (1.4%) | 2/222 (0.9%) | 23/222 (10.4%) | pass |
| classical_cnn | 29 | 29.0m | 74.4% / ≥60.0% | 78.7% | 236 | 6/236 (2.5%) | 7/236 (3.0%) | 4/236 (1.7%) | 35/236 (14.8%) | pass |
| classical_cnn | 41 | 26.0m | 74.0% / ≥60.0% | 73.3% | 220 | 1/220 (0.5%) | 2/220 (0.9%) | 2/220 (0.9%) | 34/220 (15.5%) | pass |
| modern_cnn | 17 | 7.2m | 73.9% / ≥50.0% | 78.0% | 234 | 4/234 (1.7%) | 7/234 (3.0%) | 7/234 (3.0%) | 45/234 (19.2%) | pass |
| modern_cnn | 29 | 7.5m | 73.3% / ≥50.0% | 79.3% | 238 | 2/238 (0.8%) | 1/238 (0.4%) | 4/238 (1.7%) | 30/238 (12.6%) | pass |
| modern_cnn | 41 | 7.0m | 76.2% / ≥50.0% | 70.7% | 212 | 1/212 (0.5%) | 1/212 (0.5%) | 6/212 (2.8%) | 31/212 (14.6%) | pass |
| transformer | 17 | 3.4m | 53.2% / ≥40.0% | 53.0% | 159 | 2/159 (1.3%) | 2/159 (1.3%) | 5/159 (3.1%) | 49/159 (30.8%) | pass |
| transformer | 29 | 3.5m | 51.3% / ≥40.0% | 58.3% | 175 | 0/175 (0.0%) | 1/175 (0.6%) | 7/175 (4.0%) | 47/175 (26.9%) | pass |
| transformer | 41 | 3.4m | 56.3% / ≥40.0% | 54.3% | 163 | 6/163 (3.7%) | 6/163 (3.7%) | 8/163 (4.9%) | 39/163 (23.9%) | pass |

## Across-seed mean ASR

| Held-out target | RL stochastic | Random | Score bandit | Score greedy |
| --- | ---: | ---: | ---: | ---: |
| classical_cnn | 1.45% [0.00, 4.05] | 1.74% [0.00, 4.43] | 1.17% [0.04, 2.30] | 13.55% [6.65, 20.45] |
| modern_cnn | 1.01% [0.00, 2.59] | 1.29% [0.00, 4.95] | 2.50% [0.72, 4.28] | 15.49% [7.05, 23.92] |
| transformer | 1.65% [0.00, 6.29] | 1.84% [0.00, 5.90] | 4.02% [1.83, 6.21] | 27.20% [18.61, 35.79] |

## Interpretation

The study completed 9 run(s) across 3 held-out target family/families and 3 unique seed(s). The promotion gate did not pass because at least one fold failed the paired RL-versus-control criterion; at least one fold failed the per-seed RL entropy criterion. All reported target attacks use frozen deployment and a shared total query budget; the results remain exploratory and should not be treated as a publication claim.
