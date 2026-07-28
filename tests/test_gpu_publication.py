import copy
import hashlib
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from rl_transfer.cifar_data import disjoint_balanced_subsets
from rl_transfer.cifar_pilot import MacPilotConfig
from rl_transfer.cifar_source_evaluation import source_evidence
from rl_transfer.gpu_config import RTXPublicationConfig
from rl_transfer.gpu_study import (
    _source_grid_gate,
    run_gpu_study_from_datasets,
)
from rl_transfer.config import AttackConfig
from rl_transfer.imitation import (
    BehaviorCloneStep,
    behavior_clone_policy,
    collect_best_of_k_demonstrations,
    collect_gradient_demonstrations,
)
from rl_transfer.features import patch_image_features
from rl_transfer.operator import AttackOperatorContract, choose_attack_transition
from rl_transfer.paths import resolve_within_repository
from rl_transfer.recurrent import RecurrentAttackPolicy
from rl_transfer.research_protocol import calibration_resistant_observation
from rl_transfer.source_gates import SourceGateThresholds, summarize_source_competence


def _method_metrics(
    asr: float,
    auc: float,
    *,
    digest: str = "operator-digest",
    eligible_digest: str = "eligible-digest",
    entropy: float = 0.6,
) -> dict[str, object]:
    victim_metric = {
        "eligible": 50,
        "successes": round(asr * 50),
        "asr_at_budgets": {"0": 0.0, "50": asr},
        "asr_query_auc": auc,
        "eligible_sample_ids_sha256": eligible_digest,
    }
    return {
        "eligible": 100,
        "successes": round(asr * 100),
        "asr_at_budgets": {"0": 0.0, "50": asr},
        "asr_query_auc": auc,
        "normalized_action_entropy": entropy,
        "query_budget": 50,
        "max_total_target_calls": 50,
        "initialization_included": True,
        "eligible_sample_ids_sha256": eligible_digest,
        "policy_digest_before": "policy-digest",
        "policy_digest_after": "policy-digest",
        "operator_digest": digest,
        "frozen": True,
        "by_victim": {
            "victim-0": victim_metric,
            "victim-1": dict(victim_metric),
            "victim-2": dict(victim_metric),
        },
    }


def _source_slice(learned_asr: float = 0.25) -> dict[str, object]:
    return {
        "classical_cnn": {
            "groupdro_recurrent_ppo_stochastic": _method_metrics(
                learned_asr,
                learned_asr / 2,
            ),
            "random_action": _method_metrics(0.10, 0.05),
            "bandit_action": _method_metrics(0.11, 0.055),
            "score_greedy": _method_metrics(0.16, 0.08),
        },
        "modern_cnn": {
            "groupdro_recurrent_ppo_stochastic": _method_metrics(
                learned_asr,
                learned_asr / 2,
            ),
            "random_action": _method_metrics(0.10, 0.05),
            "bandit_action": _method_metrics(0.11, 0.055),
            "score_greedy": _method_metrics(0.16, 0.08),
        },
    }


class CUDAPublicationConfigurationTests(unittest.TestCase):
    def test_pilot_config_accepts_cuda(self) -> None:
        payload = json.loads(
            Path("configs/rl_transfer/cifar10_m4_iteration.json").read_text()
        )
        config = MacPilotConfig(**{**payload, "device": "cuda"})
        self.assertEqual(config.device, "cuda")

    def test_cifar_clis_accept_explicit_cuda(self) -> None:
        from rl_transfer import cifar_cli, cifar_study_cli

        cases = (
            (cifar_cli, "run_cifar_pilot"),
            (cifar_study_cli, "run_cifar_study"),
        )
        for module, runner_name in cases:
            with self.subTest(module=module.__name__):
                with (
                    mock.patch.object(
                        module, runner_name, return_value={"status": "ok"}
                    ) as runner,
                    mock.patch.object(
                        sys,
                        "argv",
                        ["entrypoint", "--config", "fixture.json", "--device", "cuda"],
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    module.main()
                self.assertEqual(runner.call_args.kwargs["device"], "cuda")

    def test_committed_rtx_config_is_prespecified_and_fail_closed(self) -> None:
        config = RTXPublicationConfig.from_json(
            Path("configs/rl_transfer/cifar10_rtx_publication.json")
        )
        self.assertFalse(config.research_valid)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(
            set(config.target_families),
            {"classical_cnn", "modern_cnn", "transformer"},
        )
        self.assertGreaterEqual(len(config.seeds), 10)
        self.assertEqual(len(config.seeds), len(set(config.seeds)))
        self.assertTrue(config.require_source_gate)
        self.assertGreaterEqual(config.target_instances_per_family, 3)
        self.assertGreaterEqual(config.source_holdout_instances_per_family, 1)
        self.assertTrue(config.resume)
        self.assertEqual(config.split_seed, 20260727)
        self.assertGreater(config.victim_seed, max(config.seeds))

        self.assertEqual(config.replicate_unit, "policy_seed")
        self.assertEqual(config.primary_control, "score_greedy")
        self.assertEqual(config.primary_metric, "asr_at_50")
        self.assertFalse(Path(config.output_dir).is_absolute())
        self.assertNotIn("..", Path(config.output_dir).parts)

        base = MacPilotConfig.from_json(Path(config.base_config))
        self.assertEqual(base.device, "cuda")
        self.assertEqual(base.query_budget, 50)
        self.assertAlmostEqual(base.step_size, 2 / 255)
        self.assertTrue(base.rollback_on_non_improvement)
        self.assertTrue(base.action_history_features)
        self.assertTrue(base.image_patch_features)
        self.assertTrue(base.train_ablation_policies)
        attack = AttackConfig(
            epsilon=base.epsilon,
            step_size=base.step_size,
            grid_size=base.grid_size,
            max_queries=base.query_budget,
            reward_mode=base.reward_mode,
            margin_reward_scale=base.margin_reward_scale,
            terminal_success_bonus=base.terminal_success_bonus,
            query_penalty=base.query_penalty,
            rollback_on_non_improvement=base.rollback_on_non_improvement,
            action_history_features=base.action_history_features,
            image_patch_features=base.image_patch_features,
        )
        self.assertEqual(
            attack.recurrent_observation_dim,
            8 + 2 * attack.action_dim + 2 * base.grid_size * base.grid_size * 3,
        )
        self.assertEqual(base.behavior_cloning_teacher, "gradient")
        self.assertGreater(base.behavior_cloning_episodes, 0)
        self.assertGreater(base.behavior_cloning_epochs, 0)

    def test_repository_allowlist_rejects_symlinked_root(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repository_directory,
            tempfile.TemporaryDirectory() as outside_directory,
        ):
            repository = Path(repository_directory)
            (repository / "output").mkdir()
            (repository / "output" / "rl_transfer").symlink_to(
                Path(outside_directory),
                target_is_directory=True,
            )
            with (
                mock.patch(
                    "rl_transfer.paths.REPOSITORY_ROOT",
                    repository,
                ),
                self.assertRaises(ValueError),
            ):
                resolve_within_repository(
                    "output/rl_transfer/study",
                    allowed_directory="output/rl_transfer",
                    label="study",
                )

    def test_publication_config_rejects_underpowered_or_unsafe_designs(self) -> None:
        payload = json.loads(
            Path("configs/rl_transfer/cifar10_rtx_publication.json").read_text()
        )
        cases = (
            ("seeds", [1, 2, 3]),
            ("seeds", [1, 1, 2, 3, 4]),
            ("target_families", ["classical_cnn", "modern_cnn"]),
            ("require_source_gate", False),
            ("target_instances_per_family", 1),
            ("source_holdout_instances_per_family", 0),
            ("research_valid", True),
            ("output_dir", "../escape"),
            ("seeds", list(range(9))),
            ("primary_control", "random_action"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    RTXPublicationConfig(**{**payload, field: value})

    def test_bc_validation_and_source_gate_use_disjoint_stratified_indices(
        self,
    ) -> None:
        class BalancedDataset:
            def __init__(self) -> None:
                self.targets = [label for label in range(10) for _ in range(50)]

            def __getitem__(self, index):
                return torch.zeros((3, 32, 32)), self.targets[index]

        dataset = BalancedDataset()
        indices = tuple(range(500))
        bc_indices, source_indices = disjoint_balanced_subsets(
            dataset,
            indices,
            (100, 200),
        )
        self.assertEqual(len(bc_indices), 100)
        self.assertEqual(len(source_indices), 200)
        self.assertFalse(set(bc_indices) & set(source_indices))
        for subset in (bc_indices, source_indices):
            counts = {
                label: sum(dataset[index][1] == label for index in subset)
                for label in range(10)
            }
            self.assertEqual(len(set(counts.values())), 1)


class SharedAttackOperatorTests(unittest.TestCase):
    def test_margin_rollback_is_immutable_and_accepts_only_improvements(self) -> None:
        current = torch.zeros((1, 2, 2))
        proposal = torch.ones((1, 2, 2)) * 0.1
        current_before = current.clone()
        proposal_before = proposal.clone()

        worse = choose_attack_transition(
            current,
            proposal,
            current_margin=0.2,
            proposal_margin=0.2,
            success=False,
            rollback_on_non_improvement=True,
        )
        better = choose_attack_transition(
            current,
            proposal,
            current_margin=0.2,
            proposal_margin=0.1,
            success=False,
            rollback_on_non_improvement=True,
        )

        self.assertFalse(worse.accepted)
        self.assertTrue(torch.equal(worse.image, current))
        self.assertTrue(better.accepted)
        self.assertTrue(torch.equal(better.image, proposal))
        self.assertTrue(torch.equal(current, current_before))
        self.assertTrue(torch.equal(proposal, proposal_before))
        self.assertIsNot(worse.image, current)
        self.assertIsNot(better.image, proposal)

    def test_operator_contract_digest_changes_for_scientific_confounds(self) -> None:
        reference = AttackOperatorContract(
            epsilon=8 / 255,
            step_size=2 / 255,
            grid_size=4,
            rollback_on_non_improvement=True,
        )
        self.assertEqual(reference.digest(), copy.deepcopy(reference).digest())
        variants = (
            AttackOperatorContract(8 / 255, 8 / 255, 4, True),
            AttackOperatorContract(8 / 255, 2 / 255, 4, False),
            AttackOperatorContract(8 / 255, 2 / 255, 2, True),
        )
        self.assertTrue(all(item.digest() != reference.digest() for item in variants))


class BehaviorCloningTests(unittest.TestCase):
    def test_behavior_cloning_uses_only_accepted_actions(self) -> None:
        policy = RecurrentAttackPolicy(
            observation_dim=8,
            action_dim=2,
            hidden_dim=8,
            seed=3,
        )
        observation = np.asarray((0.0, 0.1, 0.2, 0.3, 0.4, -1.0, 0.0, 0.0))
        steps = (
            BehaviorCloneStep(observation, action=1, accepted=True),
            BehaviorCloneStep(observation, action=0, accepted=False),
            BehaviorCloneStep(observation, action=0, accepted=False),
            BehaviorCloneStep(observation, action=0, accepted=False),
        )

        with torch.inference_mode():
            logits_before, _, _ = policy(
                torch.as_tensor(observation, dtype=torch.float32),
                policy.initial_state(),
            )
            probability_before = float(logits_before.softmax(0)[1])
        metrics = behavior_clone_policy(policy, steps, epochs=30, seed=7)
        with torch.inference_mode():
            logits_after, _, _ = policy(
                torch.as_tensor(observation, dtype=torch.float32),
                policy.initial_state(),
            )
            probability_after = float(logits_after.softmax(0)[1])

        self.assertGreater(probability_after, probability_before)
        self.assertEqual(metrics["accepted_steps"], 1)
        self.assertEqual(metrics["rejected_steps"], 3)
        self.assertTrue(math.isfinite(float(metrics["final_loss"])))

    def test_behavior_cloning_fails_closed_without_accepted_actions(self) -> None:
        policy = RecurrentAttackPolicy(8, 2, hidden_dim=8, seed=5)
        before = policy.persistent_digest()
        steps = (BehaviorCloneStep(np.zeros(8, dtype=np.float32), 0, accepted=False),)
        with self.assertRaises(ValueError):
            behavior_clone_policy(policy, steps, epochs=2, seed=7)
        self.assertEqual(before, policy.persistent_digest())

    def test_best_of_k_teacher_uses_source_queries_and_emits_accepted_steps(
        self,
    ) -> None:
        class MeanVictim(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(
                    torch.zeros(()),
                    requires_grad=False,
                )

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                means = images.mean(dim=(1, 2, 3)) + 0 * self.anchor
                return torch.stack((means, 1 - means), dim=1)

        config = AttackConfig(
            epsilon=0.1,
            step_size=0.05,
            grid_size=1,
            max_queries=3,
            reward_mode="margin_delta",
            rollback_on_non_improvement=True,
            action_history_features=True,
        )
        steps, metrics = collect_best_of_k_demonstrations(
            {"source": (("source-0", MeanVictim()),)},
            ((torch.full((3, 4, 4), 0.7), 0),),
            config,
            episodes=1,
            candidates=config.action_dim,
            decisions=1,
            seed=11,
        )
        self.assertEqual(len(steps), 1)
        self.assertTrue(steps[0].accepted)
        self.assertEqual(len(steps[0].observation), config.recurrent_observation_dim)
        self.assertEqual(metrics["source_calls"], 1 + config.action_dim)
        self.assertEqual(metrics["accepted_steps"], 1)
        self.assertEqual(
            metrics["operator_digest"],
            AttackOperatorContract.from_config(config).digest(),
        )

    def test_action_history_observation_has_a_stable_declared_dimension(self) -> None:
        config = AttackConfig(
            grid_size=1,
            action_history_features=True,
            image_patch_features=True,
        )
        scores = torch.tensor((0.7, 0.2, 0.1))
        image = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4) / 48
        image_features = patch_image_features(image, image, grid_size=1)
        observation = calibration_resistant_observation(
            scores,
            0,
            scores,
            1.0,
            None,
            config.action_dim,
            0.0,
            0.0,
            np.zeros(config.action_dim),
            np.zeros(config.action_dim),
            image_features,
        )
        self.assertEqual(observation.shape, (config.recurrent_observation_dim,))
        self.assertTrue(np.isfinite(observation).all())

    def test_gradient_teacher_produces_global_source_action_labels(self) -> None:
        class MeanVictim(nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                means = images.mean(dim=(1, 2, 3))
                return torch.stack((means, 1 - means), dim=1)

        config = AttackConfig(
            epsilon=0.1,
            step_size=0.05,
            grid_size=1,
            max_queries=3,
            reward_mode="margin_delta",
            rollback_on_non_improvement=True,
            action_history_features=True,
            image_patch_features=True,
        )
        steps, metrics = collect_gradient_demonstrations(
            {"source": (("source-0", MeanVictim()),)},
            ((torch.full((3, 4, 4), 0.7), 0),),
            config,
            episodes=1,
            decisions=1,
            seed=13,
        )
        self.assertEqual(len(steps), 1)
        self.assertTrue(steps[0].accepted)
        self.assertEqual(
            metrics["teacher"],
            "source_privileged_cw_logit_gradient",
        )
        self.assertEqual(metrics["source_calls"], 2)
        self.assertEqual(len(steps[0].observation), config.recurrent_observation_dim)


class SourceCompetenceGateTests(unittest.TestCase):
    def test_gate_accepts_soft_action_conditioned_method_identity(
        self,
    ) -> None:
        soft_method = (
            "soft_gradient_bc_action_conditioned_groupdro_ppo_stochastic"
        )
        evaluation = {
            "exact_source": _source_slice(),
            "seen_family_new_instance": _source_slice(0.22),
        }
        for source_slice in evaluation.values():
            for methods in source_slice.values():
                methods[soft_method] = methods.pop(
                    "groupdro_recurrent_ppo_stochastic"
                )

        gate = summarize_source_competence(evaluation)

        self.assertTrue(gate["passed"])
        for source_slice in gate["slices"].values():
            for family in source_slice["families"].values():
                self.assertEqual(family["learned_method"], soft_method)

    def test_gate_requires_exact_and_unseen_source_instance_slices(self) -> None:
        evaluation = {
            "exact_source": _source_slice(),
            "seen_family_new_instance": _source_slice(0.22),
        }
        gate = summarize_source_competence(
            evaluation,
            SourceGateThresholds(
                minimum_asr_gain=0.05,
                minimum_auc_gain=0.02,
            ),
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(
            set(gate["slices"]),
            {"exact_source", "seen_family_new_instance"},
        )

    def test_gate_fails_closed_for_missing_or_confounded_source_evidence(self) -> None:
        valid = {
            "exact_source": _source_slice(),
            "seen_family_new_instance": _source_slice(),
        }
        invalid_cases = []
        missing = copy.deepcopy(valid)
        del missing["seen_family_new_instance"]
        invalid_cases.append(missing)
        weak = copy.deepcopy(valid)
        weak["exact_source"]["classical_cnn"]["groupdro_recurrent_ppo_stochastic"] = (
            _method_metrics(0.10, 0.05)
        )
        invalid_cases.append(weak)
        confounded = copy.deepcopy(valid)
        confounded["seen_family_new_instance"]["modern_cnn"]["random_action"][
            "operator_digest"
        ] = "different"
        invalid_cases.append(confounded)
        not_frozen = copy.deepcopy(valid)
        not_frozen["exact_source"]["modern_cnn"]["groupdro_recurrent_ppo_stochastic"][
            "frozen"
        ] = False
        invalid_cases.append(not_frozen)
        empty = copy.deepcopy(valid)
        empty["exact_source"]["modern_cnn"]["random_action"]["eligible"] = 0
        invalid_cases.append(empty)

        for index, evaluation in enumerate(invalid_cases):
            with self.subTest(case=index):
                gate = summarize_source_competence(
                    evaluation,
                    SourceGateThresholds(),
                )
                self.assertFalse(gate["passed"])

    def test_study_source_grid_gate_blocks_any_early_target_access(self) -> None:
        config = RTXPublicationConfig.from_json(
            Path("configs/rl_transfer/cifar10_rtx_publication.json")
        )
        runs = []
        for family in config.target_families:
            for seed in config.seeds:
                runs.append(
                    {
                        "status": "source_complete",
                        "target_family": family,
                        "seed": seed,
                        "target_evaluation_performed": False,
                        "target_calls": 0,
                        "victim_seed": config.victim_seed,
                        "victim_bank_digest": "a" * 64,
                        "validation_roles_disjoint": True,
                        "victim_accuracy_gate": {"passed": True},
                        "source_competence_gate": {"passed": True},
                        "policy": {
                            "checkpoint_sha256": hashlib.sha256(
                                f"{family}:{seed}".encode()
                            ).hexdigest(),
                            "persistent_digest": (f"policy:{family}:{seed}"),
                            "training": {
                                "behavior_cloning": {
                                    "enabled": True,
                                    "gate": {"passed": True},
                                }
                            },
                        },
                    }
                )
        self.assertTrue(_source_grid_gate(runs, config)["passed"])
        leaked = copy.deepcopy(runs)
        leaked[0]["target_evaluation_performed"] = True
        leaked[0]["target_calls"] = 1
        blocked = _source_grid_gate(leaked, config)
        self.assertFalse(blocked["passed"])
        self.assertTrue(any("target" in failure for failure in blocked["failures"]))
        changed_bank = copy.deepcopy(runs)
        changed_bank[0]["victim_bank_digest"] = "b" * 64
        self.assertFalse(_source_grid_gate(changed_bank, config)["passed"])
        repeated_policy = copy.deepcopy(runs)
        repeated_policy[1]["policy"]["checkpoint_sha256"] = repeated_policy[0][
            "policy"
        ]["checkpoint_sha256"]
        self.assertFalse(_source_grid_gate(repeated_policy, config)["passed"])

    def test_failed_source_grid_never_invokes_target_phase(self) -> None:
        config = RTXPublicationConfig.from_json(
            Path("configs/rl_transfer/cifar10_rtx_publication.json")
        )

        def failed_source_run(derived, *_args, **kwargs):
            self.assertFalse(kwargs["evaluate_target"])
            return {
                "status": "source_complete",
                "target_family": derived.target_family,
                "seed": derived.seed,
                "target_evaluation_performed": False,
                "target_calls": 0,
                "victim_accuracy_gate": {"passed": True},
                "source_competence_gate": {"passed": False},
                "policy": {
                    "training": {
                        "behavior_cloning": {
                            "enabled": True,
                            "gate": {"passed": True},
                        }
                    }
                },
            }

        with (
            mock.patch(
                "rl_transfer.gpu_study.run_cifar_pilot_from_datasets",
                side_effect=failed_source_run,
            ) as runner,
            mock.patch("rl_transfer.gpu_study._write_json"),
            mock.patch("rl_transfer.gpu_study._code_digest", return_value="digest"),
        ):
            result = run_gpu_study_from_datasets(
                config,
                mock.Mock(),
                mock.Mock(),
                dataset_version="fixture",
                phase="all",
            )
        self.assertEqual(
            runner.call_count,
            len(config.seeds) * len(config.target_families),
        )
        self.assertEqual(result["status"], "source_learning_failed")
        self.assertFalse(result["target_evaluation_performed"])
        self.assertEqual(result["target_calls"], 0)

    def test_cached_source_pass_boolean_is_ignored_and_gate_is_recomputed(
        self,
    ) -> None:
        attack = AttackConfig(
            grid_size=1,
            max_queries=2,
            rollback_on_non_improvement=True,
        )
        policy = RecurrentAttackPolicy(
            attack.recurrent_observation_dim,
            attack.action_dim,
            hidden_dim=8,
            seed=3,
        )
        populations = {
            "family": (("victim", nn.Identity()),),
        }
        holdout = {
            "family": (("holdout", nn.Identity()),),
        }
        evaluation = {"placeholder": {}}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with (
                mock.patch(
                    "rl_transfer.cifar_source_evaluation.evaluate_methods",
                    return_value=([], [], evaluation),
                ),
                mock.patch(
                    "rl_transfer.cifar_source_evaluation._audit_slices",
                    return_value={"passed": True},
                ),
                mock.patch(
                    "rl_transfer.cifar_source_evaluation.summarize_source_competence",
                    return_value={"passed": False},
                ),
            ):
                first = source_evidence(
                    policy=policy,
                    additional_policies={},
                    source_victims=populations,
                    source_holdout_victims=holdout,
                    samples=((torch.zeros((3, 4, 4)), 0),),
                    indices=(0,),
                    attack=attack,
                    seed=1,
                    main_method_prefix="groupdro_recurrent_ppo",
                    trace_samples_per_method=0,
                    thresholds=SourceGateThresholds(),
                    run_dir=run_dir,
                    binding={"fingerprint": "locked"},
                    resume=True,
                    report=lambda _message: None,
                )
            self.assertFalse(first.gate["passed"])
            cache_path = run_dir / "source_evaluation.json"
            tampered = json.loads(cache_path.read_text())
            tampered["source_competence_gate"] = {"passed": True}
            cache_path.write_text(json.dumps(tampered, indent=2, sort_keys=True))
            cache_path.with_suffix(".json.sha256").write_text(
                hashlib.sha256(cache_path.read_bytes()).hexdigest() + "\n"
            )
            with (
                mock.patch(
                    "rl_transfer.cifar_source_evaluation.evaluate_methods"
                ) as evaluator,
                mock.patch(
                    "rl_transfer.cifar_source_evaluation._audit_slices",
                    return_value={"passed": True},
                ),
                mock.patch(
                    "rl_transfer.cifar_source_evaluation.summarize_source_competence",
                    return_value={"passed": False},
                ) as summarizer,
            ):
                resumed = source_evidence(
                    policy=policy,
                    additional_policies={},
                    source_victims=populations,
                    source_holdout_victims=holdout,
                    samples=((torch.zeros((3, 4, 4)), 0),),
                    indices=(0,),
                    attack=attack,
                    seed=1,
                    main_method_prefix="groupdro_recurrent_ppo",
                    trace_samples_per_method=0,
                    thresholds=SourceGateThresholds(),
                    run_dir=run_dir,
                    binding={"fingerprint": "locked"},
                    resume=True,
                    report=lambda _message: None,
                )
            evaluator.assert_not_called()
            summarizer.assert_called_once()
            self.assertTrue(resumed.cache_resumed)
            self.assertFalse(resumed.gate["passed"])
