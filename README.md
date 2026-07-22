# DeepPoly & RL Attack Transferability

Research code for two related robustness directions:

- **Certified robustness:** DeepPoly bound propagation and adversarial-game training for neural networks.
- **Attack transferability:** a recurrent reinforcement-learning attacker trained on frozen source model families and evaluated, without further learning, on a held-out victim family.

The transferability work asks a specific question: can an RL attack policy learn reusable attack behavior from multiple source architectures, then transfer to an unseen target architecture under a fixed query budget?

> Current status: the M4/CIFAR-10 run is an **exploratory feasibility pilot**, not a paper-ready result. It validates the frozen cross-victim workflow end-to-end, but its RL attacker did not beat the random-action control.

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

### Notebook and report

- [notebooks/cifar10_mac_pilot.ipynb](notebooks/cifar10_mac_pilot.ipynb) — train or inspect an M4 run interactively.
- [docs/research/cifar10_m4_pilot_summary.md](docs/research/cifar10_m4_pilot_summary.md) — rendered figures and result interpretation.
- Regenerate the figures after recording a new manifest:

```bash
uv run python -m rl_transfer.pilot_report \
  --input docs/research/cifar10_m4_pilot_results.json \
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

The current pilot uses one dataset, one held-out model per family, and one seed. It should be expanded to multiple architecture instances, nested family-level selection, repeated seeds, and uncertainty reporting before making a general transferability claim.

## References

- DeepPoly: Singh et al., POPL 2019 — <https://github.com/eth-sri/eran>
- PGD adversarial training: Madry et al., ICLR 2018
- \(\alpha\beta\)-CROWN: Wang et al., NeurIPS 2021
