# DeepPoly & RL Attack Transferability

Research code for two related robustness directions:

- **Certified robustness:** DeepPoly bound propagation and adversarial-game training for neural networks.
- **Attack transferability:** a recurrent reinforcement-learning attacker trained on frozen source model families and evaluated, without further learning, on a held-out victim family.

The transferability work asks a specific question: can an RL attack policy learn reusable attack behavior from multiple source architectures, then transfer to an unseen target architecture under a fixed query budget?

> Current status: the corrected M4/CIFAR-10 study completed all **9 family/seed folds**. Every victim-quality gate passed, but the RL promotion gate failed: stochastic PPO was near-random and did not establish a statistically and practically validated advantage over the controls. This is a useful negative exploratory result, not a paper-ready superiority claim.

## M4 CIFAR-10 pilot results

The completed pilot trained the attacker only on a Classical CNN and a Modern CNN. It then froze the attacker and evaluated it against a held-out Patch Transformer (ViT-like) victim.

![Frozen-policy cross-victim transfer pipeline](docs/research/figures/cifar10_m4_pilot_pipeline.svg)

### Experiment setup

| Item | Pilot value |
| --- | --- |
| Dataset | CIFAR-10 |
| Hardware | Apple M4 using PyTorch MPS |
| Wall-clock time | 6 min 12 s |
| Source victim families | Classical CNN and Modern CNN |
| Held-out target family | Patch Transformer |
| RL policy | Recurrent PPO with GroupDRO source-family weighting |
| Policy training | 400 scheduled episodes, 205 trainable episodes, 5,261 source-model calls |
| Target attack budget | 25 total target queries, \(L_\infty\) epsilon = 8/255 |
| Evaluation denominator | 99 clean-correct target images |

All victim models cleared the deliberately modest pilot-quality gate before attack evaluation.

![Victim validation accuracy](docs/research/figures/cifar10_m4_pilot_victim_accuracy.svg)

| Victim family | Validation accuracy | Pilot gate |
| --- | ---: | ---: |
| Classical CNN | 66.3% | 60% |
| Modern CNN | 53.4% | 50% |
| Patch Transformer | 42.5% | 40% |

### Frozen transfer result

![Attack success rate by query budget](docs/research/figures/cifar10_m4_pilot_asr_by_query.svg)

| Method on held-out Patch Transformer | Successful attacks | ASR at 25 queries | ASR/query AUC |
| --- | ---: | ---: | ---: |
| Fixed action | 1 / 99 | 1.0% | 0.9% |
| Frozen RL policy, deterministic | 2 / 99 | 2.0% | 1.8% |
| Frozen RL policy, stochastic | 5 / 99 | 5.1% | 2.5% |
| Random action | 7 / 99 | 7.1% | 3.9% |

The policy **was frozen** during target evaluation: its SHA-256 digest was unchanged before and after deployment. However, stochastic RL reached 5.1% ASR while random reached 7.1%, so this experiment does **not** demonstrate an RL transfer advantage. The right next step is stronger target/victim fitting followed by multiple seeds and confidence intervals—not a stronger claim from one pilot.

Read the full visual report in [docs/research/cifar10_m4_pilot_summary.md](docs/research/cifar10_m4_pilot_summary.md). The committed raw aggregate is [cifar10_m4_pilot_results.json](docs/research/cifar10_m4_pilot_results.json).

## Upgraded three-seed cross-victim study

The upgraded iteration directly addresses the first pilot's failure modes:

- dense confidence-margin rewards replace sparse true-class-score rewards;
- two independently initialized victims are rotated inside each source family;
- GroupDRO remains family-level and records calls for every source model;
- all three leave-one-family-out directions are supported;
- query-matched random and score-bandit controls are joined by a custom patch-based, SimBA-style score-greedy control;
- three-seed aggregation uses paired Student-t intervals, minimum practical gains, per-seed entropy checks, and a fail-closed promotion gate;
- every promotion input is revalidated for frozen policy digests, identical eligible samples, exact query budgets, and internally consistent ASR/AUC;
- victim checkpoints are shared across folds with fitting-code/backend contracts, independent fit seeds, checksum verification, and writer locks;
- every policy block records family/instance eligibility, successes, returns, margin reductions, GroupDRO losses, weights, and source calls.

The full clean-revision run used an Apple M4 with PyTorch MPS and finished in **1 h 54 min**:

![Full three-seed frozen ASR by target family](docs/research/figures/cifar10_m4_study_asr.svg)

| Study item | Completed value |
| --- | --- |
| Held-out folds | Classical CNN, Modern CNN, Transformer |
| Fresh seeds | 17, 29, 41 |
| Independent victim instances | 18 across all seeds; two per source family and one held-out target per run |
| Policy training | 5,400 scheduled episodes; 3,640 trainable sequences; 91,800 source-model calls |
| Frozen target evaluation | 1,859 clean-correct image/run cases |
| Threat model | 25 total score queries, including initialization; L∞ = 8/255 |
| Promotion result | **Fail** in every held-out family; all victim-quality gates passed |

Victim fitting was materially stronger and stable across seeds:

| Victim family | Validation accuracy range | Gate | Target-test accuracy range |
| --- | ---: | ---: | ---: |
| Classical CNN | 73.4–74.4% | 60% | 73.3–78.7% |
| Modern CNN | 73.3–76.2% | 50% | 70.7–79.3% |
| Transformer | 51.1–56.3% | 40% | 53.0–58.3% |

Mean final attack success rate across the three fresh seeds:

| Held-out target | Stochastic RL | Random | Score bandit | Score greedy |
| --- | ---: | ---: | ---: | ---: |
| Classical CNN | 1.45% | 1.74% | 1.17% | **13.55%** |
| Modern CNN | 1.01% | 1.29% | 2.50% | **15.49%** |
| Transformer | 1.65% | 1.84% | 4.02% | **27.20%** |

The result is clear for this setup: dense-reward GroupDRO PPO did **not** learn a transferable advantage. Its normalized action entropy stayed near random (0.988–0.994), while the custom patch-based score-greedy attack was consistently much stronger under the same total query budget. This argues for changing the RL state/action/reward design or testing matched ablations before spending on larger compute—not for claiming that transfer attacks fail generally.

See the [full visual report](docs/research/cifar10_m4_study_summary.md), [verified compact results](docs/research/cifar10_m4_study_results.json), and [one-seed diagnostic](docs/research/cifar10_m4_study_quick_summary.md). Results generated before the score-greedy, RNG-provenance, and rollout-offset fixes are intentionally excluded because they are scientifically superseded.

## Run it locally

### Install

```bash
uv sync --extra vision --extra analysis --extra notebook --extra test
```

### Validate the codebase

```bash
uv run pytest -q
```

### Fast M4 feasibility run

This is the short configuration for verifying the full pipeline on a Mac. It is intentionally too small for scientific conclusions.

```bash
uv run python -m rl_transfer.cifar_cli \
  --config configs/rl_transfer/cifar10_m4_quick.json \
  --device auto
```

### Bounded M4 pilot

```bash
uv run python -m rl_transfer.cifar_cli \
  --config configs/rl_transfer/cifar10_m4_pilot.json \
  --device auto
```

The runner detects MPS automatically, checkpoints every victim and policy block, and resumes only when the configuration, deterministic split, and package-code fingerprint match. Run artifacts and checkpoints are written to `output/rl_transfer/` and are intentionally not committed.

### Three-fold diagnostic

```bash
uv run python -m rl_transfer.cifar_study_cli \
  --config configs/rl_transfer/cifar10_m4_study_quick.json \
  --device auto
```

### Three-seed study

This is the longer resumable run. It evaluates all three held-out families for fresh seeds 17, 29, and 41; seed 7 is reserved for development diagnostics.

```bash
uv run python -m rl_transfer.cifar_study_cli \
  --config configs/rl_transfer/cifar10_m4_study.json \
  --device auto
```

### Notebook and report

- [notebooks/cifar10_mac_pilot.ipynb](notebooks/cifar10_mac_pilot.ipynb) — train or inspect an M4 run interactively.
- [notebooks/cifar10_m4_study.ipynb](notebooks/cifar10_m4_study.ipynb) — inspect cross-fold, multi-seed aggregates and promotion gates.
- [docs/research/cifar10_m4_pilot_summary.md](docs/research/cifar10_m4_pilot_summary.md) — rendered figures and result interpretation.
- [docs/research/cifar10_m4_study_summary.md](docs/research/cifar10_m4_study_summary.md) — corrected nine-fold result with per-seed time, victim accuracy, and control ASR.
- Regenerate the figures after recording a new manifest:

```bash
uv run python -m rl_transfer.pilot_report \
  --input docs/research/cifar10_m4_pilot_results.json \
  --output-dir docs/research

uv run python -m rl_transfer.study_report \
  --input output/rl_transfer/cifar10_m4_studies/cifar10-m4-study/study_manifest.json \
  --output-dir docs/research
```

## Repository guide

| Area | Where to look |
| --- | --- |
| DeepPoly MNIST certification | [deeppoly_mnist.ipynb](deeppoly_mnist.ipynb) |
| PGD + DeepPoly adversarial game | [adversarial_game.py](adversarial_game.py) |
| Cross-victim RL attack harness | [rl_transfer/](rl_transfer) |
| M4 pilot configuration | [configs/rl_transfer/](configs/rl_transfer) |
| Legacy DQN transfer prototype | [rl_attack_transfer.ipynb](rl_attack_transfer.ipynb) |
| Research plan | [docs/research/rl_cross_victim_research_plan.pdf](docs/research/rl_cross_victim_research_plan.pdf) |
| Broader research directions | [research_areas.md](research_areas.md), [research_idea.md](research_idea.md), and [rl_attack_transfer_proposal.md](rl_attack_transfer_proposal.md) |
| Speech attack prototype | [speech_adversarial_attack.ipynb](speech_adversarial_attack.ipynb) |

## Research protocol notes

The RL attacker is never allowed to train on, tune against, or update during evaluation on the held-out target family. Every target call—including initialization and failed attempts—counts toward the total target-query budget. Lower-budget results are computed from each same full attack trajectory rather than by giving separate runs more chances.

The upgraded study still uses one dataset and one held-out target instance per family/seed. Broader architecture populations, nested family-level model selection, matched component ablations, and more seeds are required before making a general transferability claim.

The upgraded study is still exploratory: it changes reward shaping, victim population size, and family weighting together, so it cannot attribute any improvement to one component without matched ablations. MPS determinism is requested in warning mode, but some operators remain nondeterministic; checkpoint hashes are therefore the exact artifact identity even when fit seeds match.

## References

- DeepPoly: Singh et al., POPL 2019 — <https://github.com/eth-sri/eran>
- PGD adversarial training: Madry et al., ICLR 2018
- \(\alpha\beta\)-CROWN: Wang et al., NeurIPS 2021
