from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rl_transfer.config import AttackConfig
from rl_transfer.imitation import BehaviorCloneStep
from rl_transfer.phase2_residual_d1 import (
    ResidualCacheBinding,
    ResidualD1Request,
)
from rl_transfer.phase2_residual_d1_cache import ResidualTeacherCache
from rl_transfer.phase2_residual_d1_teacher import (
    _collect_teacher_blocks,
    _teacher_examples,
)


SOURCE_FAMILIES = ("classical_cnn", "transformer")


def _step(family: str, *, role: str = "train") -> BehaviorCloneStep:
    return BehaviorCloneStep(
        observation=(0.0,) * 12,
        action=1,
        accepted=True,
        trajectory_id=f"bc-gradient-source:{family}:{role}:0",
        step_index=0,
        action_distribution=(0.0, 1.0),
    )


class ResidualD1TeacherLifecycleTests(unittest.TestCase):
    def test_block_collection_audits_family_victim_calls_and_deadline(
        self,
    ) -> None:
        victims = {
            family: ((f"{family}-teacher", nn.Identity()),)
            for family in SOURCE_FAMILIES
        }
        steps = tuple(_step(family) for family in SOURCE_FAMILIES)
        metrics = {
            "source_calls": 4,
            "source_calls_by_family": {
                "classical_cnn": 2,
                "transformer": 2,
            },
            "gradient_evaluations": 2,
            "accepted_steps": 2,
        }
        deadline_calls = 0
        progress_messages: list[str] = []

        def deadline() -> None:
            nonlocal deadline_calls
            deadline_calls += 1

        with (
            patch(
                "rl_transfer.phase2_residual_d1_teacher.balanced_family_schedule",
                return_value=SOURCE_FAMILIES,
            ) as scheduler,
            patch(
                "rl_transfer.phase2_residual_d1_teacher"
                ".collect_gradient_demonstrations",
                return_value=(steps, metrics),
            ) as collector,
        ):
            collected, audit = _collect_teacher_blocks(
                victims=victims,
                samples=(
                    (torch.zeros(3, 4, 4), 0),
                    (torch.zeros(3, 4, 4), 1),
                ),
                config=AttackConfig(
                    epsilon=8 / 255,
                    step_size=2 / 255,
                    grid_size=4,
                    max_queries=50,
                ),
                episodes=2,
                decisions=2,
                seed=17,
                role="train",
                deadline_check=deadline,
                progress=progress_messages.append,
            )

        self.assertEqual(deadline_calls, 2)
        self.assertEqual(
            progress_messages,
            ["[d1] collecting train teacher block 1/1"],
        )
        scheduler.assert_called_once_with(SOURCE_FAMILIES, 2, 17)
        collector.assert_called_once()
        self.assertEqual(len(collected), 2)
        self.assertTrue(
            all(
                step.trajectory_id.startswith("d1-train-block-0:") for step in collected
            )
        )
        self.assertAlmostEqual(collected[0].observation[4], 49 / 50)
        self.assertAlmostEqual(collected[0].observation[7], 1 / 50)
        self.assertEqual(audit["source_calls"], 4)
        self.assertEqual(
            audit["scheduled_episodes_by_family"],
            {"classical_cnn": 1, "transformer": 1},
        )
        self.assertEqual(
            audit["source_calls_by_family"],
            {"classical_cnn": 2, "transformer": 2},
        )
        self.assertEqual(
            audit["source_calls_by_victim"],
            {
                "classical_cnn-teacher": 2,
                "transformer-teacher": 2,
            },
        )
        self.assertEqual(audit["target_calls"], 0)
        self.assertEqual(audit["hidden_target_calls"], 0)
        self.assertFalse(audit["hidden_target_evaluation_performed"])
        self.assertFalse(audit["authorizes_hidden_target_evaluation"])

    def test_teacher_examples_create_three_roles_then_return_bound_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source_manifest = source / "screen_manifest.json"
            source_manifest.write_text("{}\n")
            data = root / "data"
            data.mkdir()
            request = ResidualD1Request(
                source_manifest=source_manifest,
                source_root=source,
                output_dir=root / "study" / "d1a",
                data_root=data,
            )
            attack = SimpleNamespace(
                action_dim=96,
                recurrent_observation_dim=12,
            )
            context = SimpleNamespace(
                config=SimpleNamespace(attack_config=lambda: attack),
                source_families=SOURCE_FAMILIES,
                teacher_victims={
                    family: ((f"{family}-teacher", nn.Identity()),)
                    for family in SOURCE_FAMILIES
                },
                train_samples=tuple(object() for _ in range(200)),
                threshold_samples=tuple(object() for _ in range(50)),
                competence_samples=tuple(object() for _ in range(50)),
            )
            binding = ResidualCacheBinding(
                source_manifest_sha256="1" * 64,
                dataset_content_sha256="2" * 64,
                victim_cache_digest="3" * 64,
                request_sha256="4" * 64,
            )
            protocol = {"schema": "synthetic-source-only"}
            role_outputs = (
                (
                    (_step("classical_cnn", role="train"),),
                    {
                        "role": "train",
                        "source_calls": 2,
                        "target_calls": 0,
                        "hidden_target_calls": 0,
                        "hidden_target_evaluation_performed": False,
                        "authorizes_hidden_target_evaluation": False,
                    },
                ),
                (
                    (_step("classical_cnn", role="threshold"),),
                    {
                        "role": "threshold_selection",
                        "source_calls": 2,
                        "target_calls": 0,
                        "hidden_target_calls": 0,
                        "hidden_target_evaluation_performed": False,
                        "authorizes_hidden_target_evaluation": False,
                    },
                ),
                (
                    (_step("transformer", role="competence"),),
                    {
                        "role": "competence_gate",
                        "source_calls": 2,
                        "target_calls": 0,
                        "hidden_target_calls": 0,
                        "hidden_target_evaluation_performed": False,
                        "authorizes_hidden_target_evaluation": False,
                    },
                ),
            )

            def persist(
                output_dir: Path,
                **kwargs: object,
            ) -> ResidualTeacherCache:
                self.assertEqual(output_dir, request.output_dir)
                created = kwargs["create"]()
                self.assertIsInstance(created, ResidualTeacherCache)
                return replace(
                    created,
                    examples_sha256="5" * 64,
                    metadata_sha256="6" * 64,
                    reused=False,
                )

            def deadline() -> None:
                return None

            def progress(message: str) -> None:
                del message

            with (
                patch(
                    "rl_transfer.phase2_residual_d1_teacher._cache_binding",
                    return_value=(binding, protocol),
                ),
                patch(
                    "rl_transfer.phase2_residual_d1_teacher._collect_teacher_blocks",
                    side_effect=role_outputs,
                ) as collector,
                patch(
                    "rl_transfer.phase2_residual_d1_teacher"
                    ".load_or_create_residual_teacher_cache",
                    side_effect=persist,
                ) as cache_loader,
            ):
                train, threshold, competence, manifest = _teacher_examples(
                    request,
                    context,
                    deadline_check=deadline,
                    progress=progress,
                )

            self.assertEqual(train, role_outputs[0][0])
            self.assertEqual(threshold, role_outputs[1][0])
            self.assertEqual(competence, role_outputs[2][0])
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(manifest["heldout_family"], "modern_cnn")
            self.assertEqual(manifest["source_families"], list(SOURCE_FAMILIES))
            self.assertEqual(manifest["examples_sha256"], "5" * 64)
            self.assertEqual(manifest["metadata_sha256"], "6" * 64)
            self.assertFalse(manifest["cache_reused"])
            self.assertEqual(manifest["target_calls"], 0)
            self.assertEqual(manifest["hidden_target_calls"], 0)
            self.assertFalse(manifest["hidden_target_evaluation_performed"])
            self.assertFalse(manifest["authorizes_hidden_target_evaluation"])
            for role in manifest["roles"].values():
                self.assertEqual(role["hidden_target_calls"], 0)
                self.assertFalse(role["hidden_target_evaluation_performed"])
                self.assertFalse(role["authorizes_hidden_target_evaluation"])
            self.assertEqual(collector.call_count, 3)
            self.assertEqual(
                [call.kwargs["role"] for call in collector.call_args_list],
                ["train", "threshold_selection", "competence_gate"],
            )
            self.assertEqual(
                [call.kwargs["episodes"] for call in collector.call_args_list],
                [200, 50, 50],
            )
            self.assertEqual(
                [call.kwargs["decisions"] for call in collector.call_args_list],
                [12, 6, 6],
            )
            cache_loader.assert_called_once()
            self.assertEqual(cache_loader.call_args.kwargs["action_dim"], 96)
            self.assertEqual(
                cache_loader.call_args.kwargs["observation_dim"],
                12,
            )


if __name__ == "__main__":
    unittest.main()
