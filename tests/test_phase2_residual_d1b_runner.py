from __future__ import annotations

import itertools
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from torch import nn

from rl_transfer.phase2_residual_d1 import ResidualD1Request
from rl_transfer.phase2_residual_d1b import (
    D1B_METHODS,
    ResidualD1BEvaluationInputs,
    ResidualD1BResult,
)
from rl_transfer.phase2_residual_d1b_policy import (
    ResidualD1BEvaluationPayload,
)
from rl_transfer.phase2_residual_d1b_runner import (
    _existing_manifest,
    _evaluate_d1b_source,
    run_residual_d1b_from_datasets,
)
from rl_transfer.phase2_residual_d1b_verification import (
    d1b_block_records,
    verify_complete_d1b_children,
)
from rl_transfer.phase2_residual_d1b_artifacts import canonical_json_digest
from rl_transfer.recurrent import PPOConfig, RecurrentAttackPolicy
from rl_transfer.residual_ranker import ResidualRankerPolicy
from rl_transfer.results import ResearchResultRow
from rl_transfer.verified_artifacts import load_verified_json, write_verified_json


SOURCE_FAMILIES = ("classical_cnn", "transformer")


def _request(root: Path) -> ResidualD1Request:
    source = root / "source"
    source.mkdir()
    source_manifest = source / "screen_manifest.json"
    source_manifest.write_text('{"sealed":true}\n')
    data = root / "data"
    data.mkdir()
    return ResidualD1Request(
        source_manifest=source_manifest,
        source_root=source,
        output_dir=root / "study" / "d1a",
        data_root=data,
    )


def _d1a(*, passed: bool) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 3,
        "name": "phase2-d1a-residual-ranker-bc",
        "status": "complete",
        "heldout_family": "modern_cnn",
        "source_families": list(SOURCE_FAMILIES),
        "seed": 17,
        "bc_episodes": 200,
        "ppo_episodes_completed": 0,
        "training_performed": passed,
        "teacher_completed": passed,
        "target_calls": 0,
        "hidden_target_calls": 0,
        "target_evaluation_performed": False,
        "target_evaluation_available": False,
        "authorizes_hidden_target_evaluation": False,
        "d1_decision": {
            "passed": passed,
            "eligible_for_d1b_source_only_ppo": passed,
            "authorizes_hidden_target_evaluation": False,
        },
    }
    if passed:
        manifest["checkpoint"] = {
            "name": "residual_ranker_bc.pt",
            "sha256": "b" * 64,
            "persistent_digest": "c" * 64,
        }
    return manifest


def _policy(seed: int) -> ResidualRankerPolicy:
    return ResidualRankerPolicy(
        RecurrentAttackPolicy(
            12,
            96,
            hidden_dim=8,
            seed=seed,
            config=PPOConfig(update_epochs=1),
            actor_mode="action_conditioned",
            action_grid_size=4,
        ),
        confidence_threshold=0.2,
    )


def _evaluation_inputs() -> ResidualD1BEvaluationInputs:
    bc = _policy(1)
    ppo = _policy(2)
    indices = tuple(range(350, 400))
    payload = ResidualD1BEvaluationPayload(
        source_victims={
            family: ((f"{family}-eval", nn.Identity()),) for family in SOURCE_FAMILIES
        },
        source_samples=tuple(("sample", 0) for _ in indices),
        sample_ids=indices,
        attack_config="attack",
    )
    return ResidualD1BEvaluationInputs(
        cohort=payload,
        sample_ids=indices,
        source_families=SOURCE_FAMILIES,
        methods=D1B_METHODS,
        seed=17,
        prior_seed=50_017,
        query_budget=50,
        bc_policy=bc,
        ppo_policy=ppo,
        bc_policy_digest=bc.persistent_digest(),
        ppo_policy_digest=ppo.persistent_digest(),
        threshold_selection={
            "selection_role": "d1b_threshold_selection_only",
            "threshold": 0.2,
            "accepted_steps": 10,
            "overrides_enabled": True,
            "target_calls": 0,
        },
        competence_gate={
            "target_mode": "all_soft",
            "accepted_steps": 10,
            "target_calls": 0,
        },
    )


def _family_output(
    family: str,
) -> tuple[
    dict[str, object],
    list[ResearchResultRow],
    list[dict[str, object]],
]:
    victim_id = f"{family}-eval"
    rows = [
        ResearchResultRow(
            sample_id=f"cifar10:{family}:{victim_id}:350",
            victim_id=victim_id,
            victim_family=family,
            method=method,
            threat_model="T1",
            seed=900_017,
            query_budget=50,
            clean_correct=True,
            success=True,
            query_to_success=6,
            total_target_calls=6,
            linf=8 / 255,
            l2=0.1,
            policy_digest=method,
            action_trace=(0, 1, 2, 3, 4),
        )
        for method in D1B_METHODS
    ]
    traces = [
        {
            "method": row.method,
            "sample_id": row.sample_id,
            "victim_id": victim_id,
            "victim_family": family,
            "family": family,
            "total_target_calls": 6,
            "query_trace": [],
            "target_calls": 0,
            "hidden_target_calls": 0,
        }
        for row in rows
    ]
    methods = {
        method: {
            "eligible": 1,
            "successes": 1,
            "asr_query_auc": 0.9,
            "source_model_calls": 6,
            **(
                {}
                if method == "score_greedy"
                else {
                    "learned_override_decisions": 2,
                    "score_fallback_decisions": 3,
                }
            ),
        }
        for method in D1B_METHODS
    }
    return (
        {
            "audit": {"passed": True},
            "methods": methods,
            "source_model_calls": 18,
            "target_calls": 0,
            "hidden_target_calls": 0,
        },
        rows,
        traces,
    )


class ResidualD1BRunnerTests(unittest.TestCase):
    def test_resume_rejects_terminal_or_promoted_partial_artifacts(self) -> None:
        cases = ("failed", "running_with_final_evidence", "orphan_block")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                request = _request(root)
                output = root / "study" / "d1b"
                output.mkdir(parents=True)
                d1a = _d1a(passed=True)
                if case == "orphan_block":
                    (output / "ppo_block_050.pt").write_bytes(b"partial")
                else:
                    write_verified_json(
                        output / "d1b_manifest.json",
                        {
                            "schema_version": 3,
                            "status": ("failed" if case == "failed" else "running"),
                            "request_sha256": request.digest(),
                            "d1a_manifest_digest": canonical_json_digest(d1a),
                            "target_calls": 0,
                            "hidden_target_calls": 0,
                            "target_evaluation_performed": False,
                            "hidden_target_evaluation_performed": False,
                            "target_evaluation_available": False,
                            "authorizes_hidden_target_evaluation": False,
                        },
                    )
                    if case == "running_with_final_evidence":
                        (output / "source_results.jsonl").write_text("{}\n")

                with self.assertRaisesRegex(
                    ValueError,
                    "terminal|partial|manifest",
                ):
                    _existing_manifest(output, request, d1a)

    def test_block_records_rejects_rebound_checkpoint_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_verified_json(
                output / "ppo_block_050.receipt.json",
                {
                    "schema_version": 1,
                    "name": "phase2-d1b-residual-ranker-ppo-receipt",
                    "checkpoint": {
                        "name": "../outside.pt",
                        "sha256": "a" * 64,
                    },
                    "core_metadata": {},
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "identity|checkpoint|filename",
            ):
                d1b_block_records(output)

    def test_complete_verifier_rejects_rebound_artifact_filenames(self) -> None:
        manifest: dict[str, object] = {
            "status": "complete",
            "evaluation_role": "d1b_evaluation",
            "checkpoint": {
                "name": "residual_ranker_ppo.pt",
                "sha256": "a" * 64,
            },
            "results_sha256": "b" * 64,
            "query_traces_sha256": "c" * 64,
            "figures": {
                "asr_by_query.svg": "d" * 64,
                "final_asr.svg": "e" * 64,
            },
            "source_evaluation": {},
            "raw_evidence_verification": {"verified": True},
            "ppo_blocks": [
                {
                    "endpoint": endpoint,
                    "checkpoint": (
                        "../outside.pt"
                        if endpoint == 50
                        else f"ppo_block_{endpoint:03d}.pt"
                    ),
                    "checkpoint_sha256": "f" * 64,
                    "receipt": f"ppo_block_{endpoint:03d}.receipt.json",
                    "receipt_sha256": "1" * 64,
                }
                for endpoint in (50, 100, 150, 200)
            ],
            "target_calls": 0,
            "hidden_target_calls": 0,
            "target_evaluation_performed": False,
            "hidden_target_evaluation_performed": False,
            "target_evaluation_available": False,
            "authorizes_hidden_target_evaluation": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("rl_transfer.phase2_residual_d1b_verification._verified_child"),
                patch(
                    "rl_transfer.phase2_residual_d1b_verification."
                    "load_verified_jsonl_records",
                    return_value=[],
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_verification."
                    "verify_d1_raw_evidence",
                    return_value={"verified": True},
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_verification."
                    "verify_d1_recorded_summaries"
                ),
            ):
                with self.assertRaisesRegex(ValueError, "filename"):
                    verify_complete_d1b_children(Path(directory), manifest)

    def test_nonpassing_d1a_writes_skip_without_loading_models_or_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            output = root / "study" / "d1b"
            with patch(
                "rl_transfer.phase2_residual_d1b_runner.load_d1_source_context"
            ) as context_loader:
                result = run_residual_d1b_from_datasets(
                    request,
                    output,
                    _d1a(passed=False),
                    "train",
                    "test",
                    dataset_version="synthetic",
                    dataset_content_sha256="e" * 64,
                    runtime_environment={"environment_sha256": "f" * 64},
                    deadline_check=lambda: None,
                    progress=lambda _: None,
                )

            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["ppo_episodes_completed"], 0)
            self.assertEqual(result["target_calls"], 0)
            self.assertFalse(context_loader.called)
            self.assertEqual(
                load_verified_json(output / "d1b_manifest.json"),
                result,
            )

    def test_evaluation_uses_reserved_cohort_and_three_exact_pairings(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            inputs = _evaluation_inputs()

            def fake_plot(
                destination: Path,
                verified: object,
            ) -> tuple[Path, Path]:
                del verified
                paths = (
                    destination / "asr_by_query.svg",
                    destination / "final_asr.svg",
                )
                for path in paths:
                    path.write_text("<svg/>")
                    path.with_suffix(path.suffix + ".sha256").write_text(
                        "0" * 64 + "\n"
                    )
                return paths

            family_outputs = [_family_output(family) for family in SOURCE_FAMILIES]
            with (
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "evaluate_residual_policy_cohort",
                    side_effect=family_outputs,
                ) as evaluator,
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.verify_d1_raw_evidence",
                    return_value={
                        "verified": True,
                        "macro": {},
                        "macro_asr_at_budgets": {},
                    },
                ) as raw_verifier,
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "verify_d1_recorded_summaries"
                ) as summary_verifier,
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.write_d1_evidence_plots",
                    side_effect=fake_plot,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.paired_source_statistics",
                    side_effect=lambda rows, **kwargs: {
                        "row_methods": sorted({row.method for row in rows}),
                        **kwargs,
                    },
                ) as paired,
                patch(
                    "rl_transfer.phase2_residual_d1b_runner."
                    "residual_d1b_selection_decision",
                    return_value={
                        "passed": True,
                        "selected_method": "residual_ranker_bc_ppo",
                        "target_calls": 0,
                        "authorizes_hidden_target_evaluation": False,
                    },
                ) as decision,
            ):
                evidence = _evaluate_d1b_source(
                    output,
                    inputs,
                    deadline_check=lambda: None,
                    progress=lambda _: None,
                )

            self.assertEqual(evaluator.call_count, 2)
            for offset, family in enumerate(SOURCE_FAMILIES):
                call = evaluator.call_args_list[offset]
                self.assertEqual(
                    set(call.kwargs["policies"]),
                    {"residual_ranker_bc", "residual_ranker_bc_ppo"},
                )
                self.assertEqual(call.kwargs["family"], family)
                self.assertIs(
                    call.kwargs["samples"],
                    inputs.cohort.source_samples,
                )
                self.assertEqual(call.kwargs["indices"], inputs.sample_ids)
                self.assertEqual(call.kwargs["seed"], 900_017 + offset)
            raw_verifier.assert_called_once()
            self.assertEqual(
                raw_verifier.call_args.kwargs["expected_methods"],
                D1B_METHODS,
            )
            summary_verifier.assert_called_once()
            self.assertEqual(paired.call_count, 3)
            comparisons = {
                (
                    call.kwargs["control_method"],
                    call.kwargs["learned_method"],
                )
                for call in paired.call_args_list
            }
            self.assertEqual(
                comparisons,
                {
                    ("score_greedy", "residual_ranker_bc"),
                    ("score_greedy", "residual_ranker_bc_ppo"),
                    ("residual_ranker_bc", "residual_ranker_bc_ppo"),
                },
            )
            decision.assert_called_once()
            self.assertEqual(
                evidence["decision"]["selected_method"],
                "residual_ranker_bc_ppo",
            )
            self.assertEqual(evidence["source_model_calls"], 36)
            self.assertTrue((output / "source_results.jsonl").is_file())
            self.assertTrue((output / "source_query_traces.jsonl").is_file())

    def test_passing_d1a_runs_the_complete_d1b_lifecycle_and_resumes_immutably(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            output = root / "study" / "d1b"
            d1a = _d1a(passed=True)
            attack = SimpleNamespace(
                recurrent_observation_dim=12,
                action_dim=96,
            )
            context = SimpleNamespace(
                config=SimpleNamespace(attack_config=lambda: attack)
            )
            roles = SimpleNamespace(digest="d" * 64)
            verified = SimpleNamespace(
                manifest_digest=canonical_json_digest(d1a),
                checkpoint_sha256="b" * 64,
                bc_policy_digest="c" * 64,
            )
            resume_loads = 0
            saved_block_metrics: list[dict[str, object]] = []

            class FakeStore:
                def __init__(self, binding: object) -> None:
                    self.binding = binding

                def load_resume_state(self) -> object:
                    nonlocal resume_loads
                    resume_loads += 1
                    if resume_loads == 1:
                        return None
                    return SimpleNamespace(
                        completed_episodes=200,
                        blocks=(object(), object(), object(), object()),
                    )

                def save_block(
                    self,
                    policy: object,
                    metadata: object,
                    metrics: dict[str, object],
                ) -> str:
                    del policy, metadata
                    saved_block_metrics.append(metrics)
                    return "synthetic-receipt"

                def load_block(self, receipt: object) -> str:
                    self.assert_receipt(receipt)
                    return "synthetic-loaded-block"

                @staticmethod
                def assert_receipt(receipt: object) -> None:
                    if receipt != "synthetic-receipt":
                        raise AssertionError("unexpected synthetic receipt")

            evaluation_inputs = _evaluation_inputs()

            def fake_core(
                supplied_d1a: object,
                bundle: object,
                supplied_roles: object,
                dependencies: object,
                **kwargs: object,
            ) -> ResidualD1BResult:
                del kwargs
                self.assertEqual(
                    canonical_json_digest(supplied_d1a),
                    verified.manifest_digest,
                )
                self.assertIs(supplied_roles, roles)
                self.assertIs(
                    dependencies.verify_d1a(supplied_d1a, bundle),
                    verified,
                )
                metrics = dependencies.train_ppo_block()
                receipt = dependencies.save_block_checkpoint(
                    evaluation_inputs.ppo_policy,
                    {"block": 1},
                )
                self.assertEqual(
                    dependencies.load_block_checkpoint(receipt),
                    "synthetic-loaded-block",
                )
                self.assertEqual(metrics["source_calls"], 200)
                return ResidualD1BResult(
                    manifest={
                        "schema_version": 3,
                        "name": "phase2-d1b-residual-ranker-ppo",
                        "status": "complete",
                        "ppo_episodes_completed": 200,
                        "source_model_calls": 200,
                        "target_calls": 0,
                        "hidden_target_calls": 0,
                        "target_evaluation_performed": False,
                        "hidden_target_evaluation_performed": False,
                        "target_evaluation_available": False,
                        "authorizes_hidden_target_evaluation": False,
                    },
                    resume_state=None,
                    evaluation_inputs=evaluation_inputs,
                )

            evidence = {
                "source_evaluation": {"verified": True},
                "paired_uncertainty": {"paired": True},
                "raw_evidence_verification": {"verified": True},
                "results_sha256": "e" * 64,
                "query_traces_sha256": "f" * 64,
                "figures": {
                    "asr_by_query.svg": "1" * 64,
                    "final_asr.svg": "2" * 64,
                },
                "decision": {
                    "passed": True,
                    "selected_method": "residual_ranker_bc_ppo",
                    "authorizes_hidden_target_evaluation": False,
                },
                "source_model_calls": 36,
            }
            clock_values = itertools.count()
            with (
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.load_d1_source_context",
                    return_value=context,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner._cache_binding",
                    return_value=("binding", "protocol"),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner"
                    ".load_residual_teacher_cache",
                    return_value=object(),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.build_d1b_source_roles",
                    return_value=roles,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.verify_d1a_artifacts",
                    return_value=verified,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.ResidualD1BBlockStore",
                    FakeStore,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner"
                    ".existing_residual_ppo_block",
                    return_value={
                        "episodes": 50,
                        "source_calls": 200,
                        "target_calls": 0,
                        "hidden_target_calls": 0,
                    },
                ) as trainer,
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.run_residual_d1b",
                    side_effect=fake_core,
                ) as core_runner,
                patch(
                    "rl_transfer.phase2_residual_d1b_runner._calibrated_checkpoint",
                    return_value={
                        "name": "residual_ranker_ppo.pt",
                        "sha256": "3" * 64,
                        "persistent_digest": "4" * 64,
                        "metadata_sha256": "5" * 64,
                    },
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner._evaluate_d1b_source",
                    return_value=evidence,
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner.d1b_block_records",
                    return_value=[
                        {"endpoint": endpoint} for endpoint in (50, 100, 150, 200)
                    ],
                ),
                patch(
                    "rl_transfer.phase2_residual_d1b_runner"
                    ".verify_complete_d1b_children"
                ) as complete_verifier,
                patch(
                    "rl_transfer.phase2_residual_d1b_runner._gpu_memory_record",
                    return_value=None,
                ),
            ):
                first = run_residual_d1b_from_datasets(
                    request,
                    output,
                    d1a,
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    dataset_version="synthetic",
                    dataset_content_sha256="6" * 64,
                    runtime_environment={"environment_sha256": "7" * 64},
                    deadline_check=lambda: None,
                    progress=lambda _: None,
                    clock=lambda: float(next(clock_values)),
                )
                second = run_residual_d1b_from_datasets(
                    request,
                    output,
                    d1a,
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    dataset_version="synthetic",
                    dataset_content_sha256="6" * 64,
                    runtime_environment={"environment_sha256": "7" * 64},
                    deadline_check=lambda: None,
                    progress=lambda _: None,
                    clock=lambda: float(next(clock_values)),
                )

            self.assertEqual(first["status"], "complete")
            self.assertEqual(second, first)
            self.assertEqual(first["ppo_episodes_completed"], 200)
            self.assertEqual(first["source_model_calls"], 236)
            self.assertEqual(first["target_calls"], 0)
            self.assertEqual(first["hidden_target_calls"], 0)
            self.assertEqual(len(first["ppo_blocks"]), 4)
            self.assertEqual(
                load_verified_json(output / "d1b_manifest.json")["status"],
                "complete",
            )
            trainer.assert_called_once()
            core_runner.assert_called_once()
            self.assertEqual(
                saved_block_metrics,
                [
                    {
                        "episodes": 50,
                        "source_calls": 200,
                        "target_calls": 0,
                        "hidden_target_calls": 0,
                    }
                ],
            )
            self.assertEqual(complete_verifier.call_count, 2)


if __name__ == "__main__":
    unittest.main()
