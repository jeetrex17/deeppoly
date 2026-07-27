"""Build the checked-in RTX research notebook deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap


OUTPUT = Path("notebooks/cifar10_rtx_publication_study.ipynb")


def _source(value: str) -> list[str]:
    return textwrap.dedent(value).strip("\n").splitlines(keepends=True)


def _markdown(value: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(value),
    }


def _code(value: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(value),
    }


def build_notebook() -> dict[str, object]:
    cells = [
        _markdown(
            """
            # Frozen cross-victim attack policy study on an RTX GPU

            Authors: Jeetraj and Prashit

            This notebook is the locked execution and reporting surface for a CIFAR-10 study of frozen-parameter, target-query policy reuse. A source-trained recurrent attack policy may update its hidden state from target scores within one image, but its parameters and optimizer state remain fixed.

            The runner, not this notebook, owns the only `publication_candidate` decision. A positive decision supports only frozen-parameter, target-query policy reuse on the fixed custom CIFAR-10 victim bank. It does not guarantee acceptance by a venue and does not establish universal attack transferability.
            """
        ),
        _markdown(
            """
            ## Prespecified research compact

            | Item | Locked decision |
            |---|---|
            | Primary endpoint | Macro ASR at 50 total victim calls for the hybrid policy minus operator-matched score greedy |
            | Threat model | Untargeted score access, raw-pixel L-infinity 8/255, initialization included in 50 calls |
            | Attack operator | One 2/255 patch-channel proposal per call, central projection, margin-based rollback |
            | Main learner | CW-logit gradient teacher on owned source victims, sequence filtered hindsight imitation, then GroupDRO PPO |
            | Component ablations | Gradient-BC only and PPO only, both evaluated from frozen checkpoints |
            | Independent unit | Ten policy seeds crossed with one fixed victim bank |
            | Victim suite | Three families and three fixed instances per family |
            | Target design | Three leave-one-family-out folds and 1,000 fixed test images |
            | Disjoint validation roles | 500 victim-quality images, 100 BC-validation images, 200 source-gate images |
            | Source gate | Exact source instances plus unseen instances from each seen source family |
            | Primary inference | One two-sided exact sign-flip test on ten policy-seed macro differences |
            | Practical threshold | At least 3 percentage points macro ASR gain |
            | Secondary endpoint | Exact ASR-by-query AUC with a 1 percentage point practical threshold |
            | Fail-closed checks | Clean victims, source competence, component checkpoints, perturbation bound, calls, cohorts, operator, frozen digests |

            Do not change thresholds after inspecting target results. A changed configuration is a new experiment and needs a new study name.
            """
        ),
        _code(
            """
            import hashlib
            import json
            import os
            from pathlib import Path
            import subprocess
            import sys
            import time

            from rl_transfer.gpu_reporting import (
                create_timestamped_export_directory,
                load_verified_runtime_freeze,
                load_verified_study_manifest,
                resolve_verified_result_rows,
            )
            from rl_transfer.paths import resolve_descendant

            def find_repo_root(start: Path) -> Path:
                for candidate in (start.resolve(), *start.resolve().parents):
                    if (candidate / 'pyproject.toml').is_file() and (candidate / 'rl_transfer').is_dir():
                        return candidate
                raise RuntimeError('Run this notebook from inside the repository')

            REPO_ROOT = find_repo_root(Path.cwd())
            CONFIG = REPO_ROOT / 'configs/rl_transfer/cifar10_rtx_publication.json'
            DEVICE = 'cuda'

            RUN_BENCHMARK = True
            RUN_SOURCE_PHASE = False
            RUN_FULL_STUDY = False
            RUN_ERROR_ANALYSIS = True
            EXPORT_ARTIFACTS = False

            def environment_flag(name: str, default: bool) -> bool:
                value = os.environ.get(name)
                if value is None:
                    return default
                if value not in {'0', '1'}:
                    raise ValueError(f'{name} must be 0 or 1')
                return value == '1'

            RUN_BENCHMARK = environment_flag('RL_RUN_BENCHMARK', RUN_BENCHMARK)
            RUN_SOURCE_PHASE = environment_flag('RL_RUN_SOURCE_PHASE', RUN_SOURCE_PHASE)
            RUN_FULL_STUDY = environment_flag('RL_RUN_FULL_STUDY', RUN_FULL_STUDY)
            RUN_ERROR_ANALYSIS = environment_flag('RL_RUN_ERROR_ANALYSIS', RUN_ERROR_ANALYSIS)
            EXPORT_ARTIFACTS = environment_flag('RL_EXPORT_ARTIFACTS', EXPORT_ARTIFACTS)

            STUDY_DIR = resolve_descendant(
                REPO_ROOT,
                REPO_ROOT / 'output/rl_transfer/cifar10_rtx_publication/cifar10-rtx-publication',
                label='publication study directory',
            )
            MANIFEST_PATH = resolve_descendant(
                STUDY_DIR, 'study_manifest.json', label='study manifest'
            )
            REPORT_ROOT = resolve_descendant(
                STUDY_DIR, 'paper_artifacts', label='paper artifact root'
            )
            EXPORT_DIR = None
            print('Config: configs/rl_transfer/cifar10_rtx_publication.json')
            print('Manifest: output/rl_transfer/cifar10_rtx_publication/cifar10-rtx-publication/study_manifest.json')
            """
        ),
        _markdown(
            """
            ## Environment

            Use Python 3.12 or newer and a CUDA build of PyTorch. Python 3.12 is the reference environment for the pinned package set. From the repository root:

            ```text
            python -m pip install -r requirements/rtx-publication.txt
            python -m pip install -e . --no-deps
            ```

            The locked runner requires clean committed protocol code, configs, requirements, tests, and notebook generator. Notebook execution outputs are excluded because Jupyter autosave changes them. The runner captures code, configuration, checkpoint, split, CUDA, cuDNN, and package evidence.

            Phase flags can be set before launching Jupyter without editing this tracked notebook:

            ```text
            RL_RUN_SOURCE_PHASE=1 jupyter lab
            RL_RUN_FULL_STUDY=1 jupyter lab
            ```
            """
        ),
        _code(
            """
            required = ('torch', 'torchvision', 'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn')
            missing = []
            versions = {}
            for package in required:
                try:
                    module = __import__(package)
                    versions[package] = getattr(module, '__version__', 'unknown')
                except ImportError:
                    missing.append(package)
            PROTOCOL_PATHS = (
                'rl_transfer',
                'configs/rl_transfer',
                'requirements',
                'tests',
                'scripts/build_gpu_research_notebook.py',
                'pyproject.toml',
            )
            git_status = subprocess.run(
                [
                    'git',
                    'status',
                    '--porcelain=v1',
                    '--untracked-files=all',
                    '--',
                    *PROTOCOL_PATHS,
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            PROTOCOL_TREE_CLEAN = not bool(git_status.stdout)
            print(json.dumps({'versions': versions, 'protocol_tree_clean': PROTOCOL_TREE_CLEAN}, indent=2, sort_keys=True))
            if missing:
                raise RuntimeError(f'Missing packages: {missing}')
            """
        ),
        _markdown(
            """
            ## Validate the locked protocol

            Validation checks the ten policy seeds, fixed victim seed, three target instances, disjoint data-role budget, component ablations, explicit CUDA device, query operator, safe paths, and resampling budget. It does not train or query a victim.
            """
        ),
        _code(
            """
            from rl_transfer.cifar_config import MacPilotConfig
            from rl_transfer.gpu_config import RTXPublicationConfig

            study_config = RTXPublicationConfig.from_json(CONFIG)
            base_config_path = (REPO_ROOT / study_config.base_config).resolve()
            base_config = MacPilotConfig.from_json(base_config_path)
            validation = subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'rl_transfer.gpu_study_cli',
                    '--config',
                    str(CONFIG),
                    '--validate-only',
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            print(validation.stdout)
            config_digest = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
            action_count = base_config.grid_size ** 2 * 3 * 2
            observation_count = 8 + 2 * action_count + 2 * base_config.grid_size ** 2 * 3
            display({
                'config_sha256': config_digest,
                'policy_seeds': list(study_config.seeds),
                'fixed_victim_seed': study_config.victim_seed,
                'target_instances_per_family': study_config.target_instances_per_family,
                'action_count': action_count,
                'observation_count': observation_count,
                'component_ablations': base_config.train_ablation_policies,
            })
            """
        ),
        _markdown(
            """
            ## CUDA preflight and call-volume estimate

            A 12 GB RTX card is sufficient for the listed models because victims train sequentially. Wall time is dominated by sequential single-image attack calls. The benchmark measures all three victim families and uses the slowest rate for the estimate. Victim fitting time is reported separately by the completed manifests.
            """
        ),
        _code(
            """
            from rl_transfer.gpu_preflight import (
                benchmark_single_image_calls,
                cuda_preflight,
                estimate_study_calls,
                estimate_wall_time_hours,
            )

            preflight = cuda_preflight(minimum_memory_gib=10.0)
            call_plan = estimate_study_calls(CONFIG)
            print(json.dumps(preflight, indent=2, sort_keys=True))
            print(json.dumps(call_plan, indent=2, sort_keys=True))
            benchmark = None
            if RUN_BENCHMARK and preflight['passed']:
                benchmark = benchmark_single_image_calls(device=DEVICE, calls=500, warmup=50)
                estimated_hours = estimate_wall_time_hours(
                    call_plan['total_upper_bound'],
                    benchmark['calls_per_second'],
                    overhead_multiplier=1.75,
                )
                print(json.dumps(benchmark, indent=2, sort_keys=True))
                print(f'Conservative attack-call estimate: {estimated_hours:.1f} hours')
            elif not preflight['passed']:
                print('CUDA preflight did not pass. Move the notebook to the RTX environment before training.')
            """
        ),
        _markdown(
            """
            ## Phase 1: source learning only

            This phase trains 30 hybrid policies and 30 PPO-only ablations. The fixed victim bank is fitted once and reused by checkpoint digest. It evaluates exact source victims and unseen instances from seen source families. No target-family attack call is allowed before the full source grid passes.
            """
        ),
        _code(
            """
            def run_study_phase(phase: str) -> None:
                if not PROTOCOL_TREE_CLEAN:
                    raise RuntimeError('Commit protocol-critical code and configs before the locked run')
                subprocess.run(
                    [
                        sys.executable,
                        '-m',
                        'rl_transfer.gpu_study_cli',
                        '--config',
                        str(CONFIG),
                        '--phase',
                        phase,
                    ],
                    cwd=REPO_ROOT,
                    check=True,
                )

            if RUN_SOURCE_PHASE:
                if not preflight['passed']:
                    raise RuntimeError('CUDA preflight must pass before source training')
                run_study_phase('source')
            else:
                print('Source training is disabled. Set RUN_SOURCE_PHASE = True on the RTX kernel.')
            """
        ),
        _code(
            """
            source_manifest = load_verified_study_manifest(MANIFEST_PATH, STUDY_DIR) if MANIFEST_PATH.is_file() else None
            if source_manifest is None:
                print('No source manifest yet.')
            else:
                source_gate = source_manifest.get('source_competence_gate', {})
                print(f"Status: {source_manifest.get('status')}")
                print(f"Source gate passed: {source_gate.get('passed', False)}")
                print(f"Target evaluation performed: {source_manifest.get('target_evaluation_performed', False)}")
                print(f"Recorded target calls: {source_manifest.get('target_calls', 0)}")
                failures = source_gate.get('failures', [])
                if failures:
                    display(failures[:20])
            """
        ),
        _markdown(
            """
            ## Phase 2: locked target evaluation

            Enable this phase only after Phase 1 reports a passed source gate. The runner recomputes source competence from the current frozen checkpoints before each target run. Hidden state may adapt within one image, but parameters and optimizer state must retain identical digests.
            """
        ),
        _code(
            """
            if RUN_FULL_STUDY:
                if not preflight['passed']:
                    raise RuntimeError('CUDA preflight must pass before target evaluation')
                run_study_phase('all')
            else:
                print('Target evaluation is disabled. Set RUN_FULL_STUDY = True only for the locked run.')
            """
        ),
        _code(
            """
            study = load_verified_study_manifest(MANIFEST_PATH, STUDY_DIR) if MANIFEST_PATH.is_file() else None
            if study is None:
                print('No study manifest is available.')
            else:
                print(f"Status: {study.get('status')}")
                print(f"Publication candidate: {study.get('publication_candidate', False)}")
                print(f"Claim scope: {study.get('claim_scope', 'no target claim')}")
                print(f"Recorded source plus target hours: {study.get('total_recorded_elapsed_seconds', study.get('elapsed_seconds', 0)) / 3600:.2f}")
            """
        ),
        _markdown(
            """
            ## Authoritative evidence decision

            The table below renders the runner-owned decision. The notebook does not create a second promotion flag. Families and victims are repeated measurements within a policy seed. The exact primary test uses ten seed-level macro differences.
            """
        ),
        _code(
            """
            import pandas as pd

            evidence_rows = []
            confirmatory_gate = {}
            if study is not None:
                source_gate = study.get('source_competence_gate', {})
                confirmatory_gate = study.get('confirmatory_gate', {})
                evidence_rows = [
                    {'check': 'source grid complete', 'passed': source_gate.get('grid_complete', False)},
                    {'check': 'source competence', 'passed': source_gate.get('passed', False)},
                    {'check': 'target grid and raw audits', 'passed': confirmatory_gate.get('grid_complete', False)},
                    {'check': 'primary exact seed-level test', 'passed': bool(
                        confirmatory_gate.get('primary', {}).get('exact_sign_flip_pvalue') is not None
                        and confirmatory_gate['primary']['exact_sign_flip_pvalue'] <= 0.05
                    )},
                    {'check': 'authoritative target gate', 'passed': confirmatory_gate.get('passed', False)},
                    {'check': 'publication candidate', 'passed': study.get('publication_candidate', False)},
                ]
            evidence_table = pd.DataFrame(evidence_rows)
            display(evidence_table)
            if confirmatory_gate:
                display(pd.DataFrame([confirmatory_gate.get('primary', {})]))
                display(pd.DataFrame([confirmatory_gate.get('secondary_query_efficiency', {})]))
                display(pd.DataFrame.from_dict(
                    confirmatory_gate.get('policy_seed_differences', {}),
                    orient='index',
                ).rename_axis('policy_seed').reset_index())
            """
        ),
        _markdown(
            """
            ## Seed-level metrics and component attribution

            The primary comparison is the stochastic hybrid against matched score greedy. Random action, UCB, deterministic decoding, BC only, and PPO only are secondary controls. The hybrid must have positive mean ASR differences over both component ablations before the final gate can pass.
            """
        ),
        _code(
            """
            import numpy as np
            from rl_transfer.statistics import bootstrap_interval

            LEARNED = 'gradient_bc_groupdro_ppo_stochastic'
            PRIMARY_CONTROL = 'score_greedy'
            SECONDARY_CONTROLS = (
                'random_action',
                'bandit_action',
                'gradient_bc_only_stochastic',
                'ppo_only_stochastic',
            )
            seed_rows = []
            if study is not None and study.get('status') == 'complete':
                for run in study['runs']:
                    for method, metrics in run['evaluation'].items():
                        curve = {int(key): float(value) for key, value in metrics['asr_at_budgets'].items()}
                        victim_asr = [
                            float(victim['asr_at_budgets'][str(base_config.query_budget)])
                            if str(base_config.query_budget) in victim['asr_at_budgets']
                            else float(victim['asr_at_budgets'][base_config.query_budget])
                            for victim in metrics.get('by_victim', {}).values()
                        ]
                        seed_rows.append({
                            'target_family': run['target_family'],
                            'policy_seed': run['seed'],
                            'victim_seed': run['victim_seed'],
                            'method': method,
                            'eligible': metrics['eligible'],
                            'final_asr': float(np.mean(victim_asr)) if victim_asr else curve[max(curve)],
                            'asr_query_auc': metrics['asr_query_auc'],
                            'action_entropy': metrics['normalized_action_entropy'],
                        })
            seed_metrics = pd.DataFrame(seed_rows)
            display(seed_metrics.head(24))

            secondary_rows = []
            if not seed_metrics.empty:
                for family in study_config.target_families:
                    family_frame = seed_metrics[seed_metrics.target_family == family]
                    learned = family_frame[family_frame.method == LEARNED].set_index('policy_seed')
                    for control in (PRIMARY_CONTROL, *SECONDARY_CONTROLS):
                        compared = family_frame[family_frame.method == control].set_index('policy_seed')
                        paired = learned.join(compared, lsuffix='_learned', rsuffix='_control', validate='one_to_one')
                        differences = tuple(paired.final_asr_learned - paired.final_asr_control)
                        interval = bootstrap_interval(
                            differences,
                            samples=study_config.bootstrap_samples,
                            seed=study_config.split_seed + len(secondary_rows) + 10,
                        )
                        secondary_rows.append({
                            'target_family': family,
                            'control': control,
                            'policy_seeds': len(differences),
                            'mean_asr_difference': float(np.mean(differences)),
                            'bootstrap_ci_low': interval[0],
                            'bootstrap_ci_high': interval[1],
                            'confirmatory_test': control == PRIMARY_CONTROL,
                        })
            secondary_statistics = pd.DataFrame(secondary_rows)
            display(secondary_statistics)
            """
        ),
        _markdown(
            """
            ## Publication figures

            Figures show every policy-seed point, uncertainty, exact query curves, source behavior, component ablations, victim accuracy, action entropy, and invocation time. Plotting never re-queries a victim.
            """
        ),
        _code(
            """
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_theme(style='whitegrid', context='talk')
            figure_paths = []

            def prepare_export_dir():
                global EXPORT_DIR
                if not EXPORT_ARTIFACTS:
                    return None
                if EXPORT_DIR is None:
                    stamp = time.strftime('%Y%m%d_%H%M%S', time.gmtime())
                    EXPORT_DIR = create_timestamped_export_directory(
                        STUDY_DIR, REPORT_ROOT, stamp
                    )
                return EXPORT_DIR

            def finish_figure(fig, stem):
                fig.tight_layout()
                export_dir = prepare_export_dir()
                if export_dir is not None:
                    for suffix in ('png', 'pdf'):
                        path = export_dir / f'{stem}.{suffix}'
                        fig.savefig(path, dpi=240, bbox_inches='tight')
                        figure_paths.append(path)
                plt.show()

            source_metric_rows = []
            victim_rows = []
            runtime_rows = []
            if study is not None and study.get('status') == 'complete':
                for run in study['source_runs']:
                    for slice_name, families in run['source_evaluation'].items():
                        for family, methods in families.items():
                            for method in (LEARNED, PRIMARY_CONTROL):
                                curve = {int(key): float(value) for key, value in methods[method]['asr_at_budgets'].items()}
                                source_metric_rows.append({
                                    'slice': slice_name,
                                    'source_family': family,
                                    'policy_seed': run['seed'],
                                    'method': method,
                                    'final_asr': curve[max(curve)],
                                })
                seen_victims = set()
                for run in study['runs']:
                    runtime_rows.append({
                        'target_family': run['target_family'],
                        'policy_seed': run['seed'],
                        'target_evaluation_hours': run.get('target_evaluation_elapsed_seconds', 0) / 3600,
                    })
                    for victim_id, accuracy in run['target_test_accuracy_by_victim'].items():
                        if victim_id not in seen_victims:
                            seen_victims.add(victim_id)
                            victim_rows.append({
                                'target_family': run['target_family'],
                                'victim_id': victim_id,
                                'accuracy': accuracy,
                            })

                selected = seed_metrics[seed_metrics.method.isin(
                    (LEARNED, PRIMARY_CONTROL, 'gradient_bc_only_stochastic', 'ppo_only_stochastic')
                )]
                fig, ax = plt.subplots(figsize=(11, 5.5))
                sns.pointplot(
                    data=selected,
                    x='target_family',
                    y='final_asr',
                    hue='method',
                    errorbar=('ci', 95),
                    dodge=0.45,
                    markers='o',
                    ax=ax,
                )
                sns.stripplot(
                    data=selected,
                    x='target_family',
                    y='final_asr',
                    hue='method',
                    dodge=True,
                    alpha=0.35,
                    legend=False,
                    ax=ax,
                )
                ax.set(xlabel='Held-out family', ylabel='Macro ASR at 50 calls', title='Frozen target attack success by policy seed')
                finish_figure(fig, 'target_final_asr')

                curve_rows = []
                for run in study['runs']:
                    for method in (LEARNED, PRIMARY_CONTROL):
                        for budget, value in run['evaluation'][method]['asr_at_budgets'].items():
                            curve_rows.append({
                                'target_family': run['target_family'],
                                'policy_seed': run['seed'],
                                'method': method,
                                'budget': int(budget),
                                'asr': float(value),
                            })
                curve_frame = pd.DataFrame(curve_rows)
                fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
                for axis, family in zip(axes, study_config.target_families):
                    sns.lineplot(
                        data=curve_frame[curve_frame.target_family == family],
                        x='budget',
                        y='asr',
                        hue='method',
                        errorbar=('ci', 95),
                        ax=axis,
                    )
                    axis.set_title(family.replace('_', ' '))
                    axis.set(xlabel='Total victim calls', ylabel='ASR')
                finish_figure(fig, 'target_query_curves')

                source_metrics = pd.DataFrame(source_metric_rows)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.pointplot(
                    data=source_metrics,
                    x='slice',
                    y='final_asr',
                    hue='method',
                    errorbar=('ci', 95),
                    dodge=0.35,
                    ax=ax,
                )
                ax.set(xlabel='Source evaluation role', ylabel='ASR at 50 calls', title='Source competence before target access')
                finish_figure(fig, 'source_competence')

                victim_frame = pd.DataFrame(victim_rows)
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.stripplot(data=victim_frame, x='target_family', y='accuracy', size=9, ax=ax)
                ax.set(xlabel='Victim family', ylabel='Clean test accuracy', title='Fixed held-out victim bank')
                finish_figure(fig, 'victim_accuracy')

                entropy_frame = seed_metrics[seed_metrics.method == LEARNED]
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                sns.stripplot(data=entropy_frame, x='target_family', y='action_entropy', ax=axes[0])
                axes[0].set(xlabel='Held-out family', ylabel='Normalized entropy', title='Hybrid executed-action entropy')
                runtime_frame = pd.DataFrame(runtime_rows)
                sns.boxplot(data=runtime_frame, x='target_family', y='target_evaluation_hours', ax=axes[1])
                sns.stripplot(data=runtime_frame, x='target_family', y='target_evaluation_hours', color='black', alpha=0.5, ax=axes[1])
                axes[1].set(xlabel='Held-out family', ylabel='Hours', title='Original target evaluation time')
                finish_figure(fig, 'entropy_and_runtime')
            else:
                source_metrics = pd.DataFrame()
                victim_frame = pd.DataFrame()
                runtime_frame = pd.DataFrame()
                print('Figures require a complete target study.')
            """
        ),
        _markdown(
            """
            ## Raw-row audit and qualitative trace examples

            The runner has already made these audits part of the authoritative gate. This cell independently streams the saved rows without loading full image tensors. Manifest-controlled paths are resolved inside the study run directory before use.
            """
        ),
        _code(
            """
            from collections import Counter

            audit_rows = []
            outcome_counter = Counter()
            examples = {}
            if RUN_ERROR_ANALYSIS and study is not None and study.get('status') == 'complete':
                allowed_runs = (STUDY_DIR / 'runs').resolve()
                for run in study['runs']:
                    run_dir = Path(run['run_dir'])
                    if not run_dir.is_absolute():
                        run_dir = REPO_ROOT / run_dir
                    result_path = resolve_verified_result_rows(
                        run_dir, allowed_runs, STUDY_DIR
                    )
                    with result_path.open() as handle:
                        for line in handle:
                            row = json.loads(line)
                            outcome = (
                                'clean_error'
                                if not row['clean_correct']
                                else 'attack_success'
                                if row['success']
                                else 'attack_failure'
                            )
                            key = (run['target_family'], row['method'], outcome)
                            outcome_counter[key] += 1
                            example_key = (run['target_family'], row['method'], outcome)
                            if example_key not in examples:
                                examples[example_key] = {
                                    'target_family': run['target_family'],
                                    'policy_seed': run['seed'],
                                    'method': row['method'],
                                    'outcome': outcome,
                                    'victim_id': row['victim_id'],
                                    'sample_id': row['sample_id'],
                                    'query_to_success': row['query_to_success'],
                                    'total_calls': row['total_target_calls'],
                                    'linf': row['linf'],
                                    'first_actions': row['action_trace'][:10],
                                }
                audit_rows = [
                    {
                        'target_family': run['target_family'],
                        'policy_seed': run['seed'],
                        **run['evaluation_audit'],
                    }
                    for run in study['runs']
                ]
            audit_table = pd.DataFrame(audit_rows)
            outcome_table = pd.DataFrame([
                {
                    'target_family': family,
                    'method': method,
                    'outcome': outcome,
                    'count': count,
                }
                for (family, method, outcome), count in outcome_counter.items()
            ])
            example_table = pd.DataFrame(examples.values())
            display(audit_table)
            display(outcome_table)
            display(example_table.head(24))
            """
        ),
        _markdown(
            """
            ## Export the paper evidence bundle

            Export is opt-in and creates a timestamped directory, so rerunning the notebook does not overwrite prior evidence. The bundle includes the authoritative manifest, configs, package freeze, seed metrics, secondary analyses, audits, trace examples, and every generated PNG and PDF.
            """
        ),
        _code(
            """
            exported = {}
            if EXPORT_ARTIFACTS and study is not None and study.get('status') == 'complete':
                export_dir = prepare_export_dir()
                seed_metrics.to_csv(export_dir / 'seed_level_metrics.csv', index=False)
                secondary_statistics.to_csv(export_dir / 'secondary_statistics.csv', index=False)
                evidence_table.to_csv(export_dir / 'evidence_checklist.csv', index=False)
                audit_table.to_csv(export_dir / 'raw_run_audits.csv', index=False)
                outcome_table.to_csv(export_dir / 'outcome_counts.csv', index=False)
                example_table.to_json(export_dir / 'qualitative_trace_examples.json', orient='records', indent=2)
                (export_dir / 'study_manifest.json').write_text(
                    json.dumps(study, indent=2, sort_keys=True, allow_nan=False)
                )
                (export_dir / 'study_config.json').write_text(
                    json.dumps(json.loads(CONFIG.read_text()), indent=2, sort_keys=True)
                )
                (export_dir / 'base_config.json').write_text(
                    json.dumps(json.loads(base_config_path.read_text()), indent=2, sort_keys=True)
                )
                runtime_freeze, freeze = load_verified_runtime_freeze(
                    study, STUDY_DIR, REPO_ROOT
                )
                (export_dir / 'pip_freeze.txt').write_text(freeze)
                decision = {
                    'study_name': study['name'],
                    'status': study['status'],
                    'publication_candidate': study.get('publication_candidate', False),
                    'claim_scope': study.get('claim_scope'),
                    'source_competence_gate': study.get('source_competence_gate'),
                    'confirmatory_gate': study.get('confirmatory_gate'),
                    'study_code_digest': study.get('study_code_digest'),
                    'elapsed_seconds': study.get('elapsed_seconds'),
                    'total_recorded_elapsed_seconds': study.get('total_recorded_elapsed_seconds'),
                    'exported_at_unix': time.time(),
                }
                (export_dir / 'decision_record.json').write_text(
                    json.dumps(decision, indent=2, sort_keys=True, allow_nan=False)
                )
                exported = {
                    path.name: str(path.relative_to(REPO_ROOT))
                    for path in sorted(export_dir.iterdir())
                    if path.is_file()
                }
                print(json.dumps(exported, indent=2, sort_keys=True))
            else:
                print('Set EXPORT_ARTIFACTS = True after a complete study to write a new evidence bundle.')
            """
        ),
        _markdown(
            """
            ## Interpretation

            - If the source gate fails, report a source-learning failure and do not interpret target transfer.
            - If the hybrid does not beat BC only, PPO did not add measurable transfer value in this protocol.
            - If the hybrid does not beat PPO only, the privileged source warm start did not add measurable transfer value.
            - If the hybrid beats random or UCB but not matched score greedy, do not claim superiority over the primary control.
            - A passed final gate supports a narrow frozen-parameter, target-query reuse result on this fixed custom victim suite.
            - Do not call this query-free adversarial-example transfer or continual learning.
            - Replication on standard external checkpoints and a second dataset remains necessary before a broad practical claim.
            """
        ),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (CUDA)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = (
        json.dumps(build_notebook(), indent=1, ensure_ascii=False)
        + "\n"
    )
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != encoded:
            raise SystemExit(f"{OUTPUT} is not up to date")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(encoded)


if __name__ == "__main__":
    main()
