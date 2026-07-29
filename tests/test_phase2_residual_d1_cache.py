from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest import mock

from rl_transfer.imitation import BehaviorCloneStep
from rl_transfer.phase2_residual_d1 import ResidualCacheBinding
from rl_transfer.phase2_residual_d1_cache import (
    ResidualTeacherCache,
    ResidualTeacherCachePaths,
    load_or_create_residual_teacher_cache,
    load_residual_teacher_cache,
    residual_teacher_protocol_digest,
    write_residual_teacher_cache,
)


_ACTION_DIM = 3
_OBSERVATION_DIM = 2
_ROLES = ("train", "threshold_selection", "competence_gate")
_SOURCE_FAMILIES = ("classical_cnn", "transformer")


def _protocol() -> dict[str, object]:
    return {
        "schema": "phase2-d1-residual-teacher-v2",
        "request_sha256": "4" * 64,
        "code_digest": "5" * 64,
        "train_decisions": 12,
        "validation_decisions": 6,
        "bc_epochs": 12,
        "soft_temperature": 0.5,
        "prior_temperature": 24.0,
        "operator_digest": "6" * 64,
        "role_indices_sha256": {
            "train": "7" * 64,
            "threshold_selection": "8" * 64,
            "competence_gate": "9" * 64,
            "source_holdout_evaluation": "a" * 64,
            "source_ppo_evaluation": "b" * 64,
        },
        "teacher_victim_ids": {
            "classical_cnn": ["classical-teacher-0"],
            "transformer": ["transformer-teacher-0"],
        },
        "evaluation_victim_ids": {
            "classical_cnn": ["classical-evaluation-0"],
            "transformer": ["transformer-evaluation-0"],
        },
        "victim_cache_digest": "3" * 64,
    }


def _binding(protocol: dict[str, object]) -> ResidualCacheBinding:
    return ResidualCacheBinding(
        source_manifest_sha256="1" * 64,
        dataset_content_sha256="2" * 64,
        victim_cache_digest="3" * 64,
        request_sha256=residual_teacher_protocol_digest(protocol),
    )


def _step(role: str, family: str = "classical_cnn") -> BehaviorCloneStep:
    return BehaviorCloneStep(
        (0.25, 0.75),
        action=1,
        accepted=True,
        trajectory_id=(f"d1-{role}-block-0:bc-gradient-source:{family}:0"),
        step_index=0,
        action_distribution=(0.1, 0.8, 0.1),
    )


def _role_metrics(role: str) -> dict[str, object]:
    return {
        "role": role,
        "episodes": 1,
        "decisions_per_episode": 12 if role == "train" else 6,
        "steps": 1,
        "accepted_steps": 1,
        "source_calls": 2,
        "gradient_evaluations": 1,
        "scheduled_episodes_by_family": {
            "classical_cnn": 1,
            "transformer": 0,
        },
        "source_calls_by_family": {
            "classical_cnn": 2,
            "transformer": 0,
        },
        "scheduled_episodes_by_victim": {
            "classical-teacher-0": 1,
            "transformer-teacher-0": 0,
        },
        "source_calls_by_victim": {
            "classical-teacher-0": 2,
            "transformer-teacher-0": 0,
        },
        "victim_diagnostics": {
            "classical-teacher-0": {
                "scheduled_episodes": 1,
                "source_calls": 2,
            },
            "transformer-teacher-0": {
                "scheduled_episodes": 0,
                "source_calls": 0,
            },
        },
        "target_calls": 0,
        "hidden_target_calls": 0,
        "hidden_target_evaluation_performed": False,
        "authorizes_hidden_target_evaluation": False,
    }


def _cache() -> ResidualTeacherCache:
    protocol = _protocol()
    return ResidualTeacherCache(
        binding=_binding(protocol),
        protocol=protocol,
        heldout_family="modern_cnn",
        source_families=_SOURCE_FAMILIES,
        train_steps=(_step("train"),),
        threshold_steps=(_step("threshold_selection"),),
        competence_steps=(_step("competence_gate"),),
        role_metrics={role: _role_metrics(role) for role in _ROLES},
    )


def _rewrite_checksum(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def _load(root: Path, cache: ResidualTeacherCache | None = None):
    expected = cache or _cache()
    return load_residual_teacher_cache(
        root,
        expected_binding=expected.binding,
        expected_protocol=expected.protocol,
        action_dim=_ACTION_DIM,
        observation_dim=_OBSERVATION_DIM,
    )


class ResidualTeacherCacheRoundTripTests(unittest.TestCase):
    def test_round_trip_is_checksum_verified_immutable_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = _cache()

            written = write_residual_teacher_cache(
                root,
                original,
                action_dim=_ACTION_DIM,
                observation_dim=_OBSERVATION_DIM,
            )
            loaded = _load(root, original)

            self.assertFalse(written.reused)
            self.assertTrue(loaded.reused)
            self.assertEqual(loaded.binding, original.binding)
            self.assertEqual(dict(loaded.protocol), dict(original.protocol))
            self.assertEqual(loaded.train_steps, original.train_steps)
            self.assertEqual(
                loaded.threshold_steps,
                original.threshold_steps,
            )
            self.assertEqual(
                loaded.competence_steps,
                original.competence_steps,
            )
            self.assertRegex(loaded.examples_sha256 or "", r"^[0-9a-f]{64}$")
            self.assertRegex(loaded.metadata_sha256 or "", r"^[0-9a-f]{64}$")
            paths = ResidualTeacherCachePaths(root)
            for path in paths.artifacts:
                self.assertTrue(path.is_file(), path)
            metadata = json.loads(paths.metadata.read_text())
            records = [
                json.loads(line) for line in paths.examples.read_text().splitlines()
            ]
            source_only_payloads = (
                metadata,
                *metadata["roles"].values(),
                *records,
            )
            for payload in source_only_payloads:
                self.assertEqual(payload["hidden_target_calls"], 0)
                self.assertFalse(payload["hidden_target_evaluation_performed"])
                self.assertFalse(payload["authorizes_hidden_target_evaluation"])
            self.assertFalse(tuple(root.glob("*.tmp")))
            with self.assertRaises(FrozenInstanceError):
                loaded.reused = False  # type: ignore[misc]
            with self.assertRaises(TypeError):
                loaded.protocol["new"] = "forbidden"  # type: ignore[index]

            factory = mock.Mock(side_effect=AssertionError("must reuse"))
            reused = load_or_create_residual_teacher_cache(
                root,
                expected_binding=original.binding,
                expected_protocol=original.protocol,
                action_dim=_ACTION_DIM,
                observation_dim=_OBSERVATION_DIM,
                create=factory,
            )
            factory.assert_not_called()
            self.assertTrue(reused.reused)
            self.assertEqual(reused.examples_sha256, loaded.examples_sha256)

    def test_absent_cache_is_created_once_then_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            original = _cache()
            factory = mock.Mock(return_value=original)

            created = load_or_create_residual_teacher_cache(
                root,
                expected_binding=original.binding,
                expected_protocol=original.protocol,
                action_dim=_ACTION_DIM,
                observation_dim=_OBSERVATION_DIM,
                create=factory,
            )
            reused = load_or_create_residual_teacher_cache(
                root,
                expected_binding=original.binding,
                expected_protocol=original.protocol,
                action_dim=_ACTION_DIM,
                observation_dim=_OBSERVATION_DIM,
                create=factory,
            )

            factory.assert_called_once_with()
            self.assertFalse(created.reused)
            self.assertTrue(reused.reused)


class ResidualTeacherCacheIdentityTests(unittest.TestCase):
    def test_loader_rejects_every_binding_and_protocol_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = _cache()
            write_residual_teacher_cache(
                root,
                original,
                action_dim=_ACTION_DIM,
                observation_dim=_OBSERVATION_DIM,
            )

            for field in (
                "source_manifest_sha256",
                "dataset_content_sha256",
                "victim_cache_digest",
                "request_sha256",
            ):
                with self.subTest(field=field):
                    mismatched = replace(
                        original.binding,
                        **{field: "f" * 64},
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "binding|identity|mismatch",
                    ):
                        load_residual_teacher_cache(
                            root,
                            expected_binding=mismatched,
                            expected_protocol=original.protocol,
                            action_dim=_ACTION_DIM,
                            observation_dim=_OBSERVATION_DIM,
                        )

            protocol = _protocol()
            protocol["code_digest"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "protocol|mismatch"):
                load_residual_teacher_cache(
                    root,
                    expected_binding=original.binding,
                    expected_protocol=protocol,
                    action_dim=_ACTION_DIM,
                    observation_dim=_OBSERVATION_DIM,
                )

    def test_writer_rejects_invalid_records_counts_and_unknown_metadata(self) -> None:
        original = _cache()
        invalid_action = BehaviorCloneStep(
            (0.25, 0.75),
            action=3,
            accepted=True,
            trajectory_id=("d1-train-block-0:bc-gradient-source:classical_cnn:0"),
            step_index=0,
            action_distribution=(0.05, 0.05, 0.1, 0.8),
        )
        contaminated = _step("train", family="modern_cnn")
        duplicate = replace(
            original,
            train_steps=(original.train_steps[0], original.train_steps[0]),
        )
        wrong_count_metrics = {
            **{role: dict(metrics) for role, metrics in original.role_metrics.items()},
            "train": {
                **dict(original.role_metrics["train"]),
                "steps": 2,
            },
        }
        cases = (
            replace(original, train_steps=(invalid_action,)),
            replace(original, train_steps=(contaminated,)),
            duplicate,
            replace(original, role_metrics=wrong_count_metrics),
        )

        for index, cache in enumerate(cases):
            with (
                self.subTest(index=index),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaisesRegex(
                    ValueError,
                    "action|source|duplicate|count|steps|cache",
                ),
            ):
                write_residual_teacher_cache(
                    Path(directory),
                    cache,
                    action_dim=_ACTION_DIM,
                    observation_dim=_OBSERVATION_DIM,
                )


class ResidualTeacherCacheTamperTests(unittest.TestCase):
    def _write(self, root: Path) -> ResidualTeacherCache:
        cache = _cache()
        write_residual_teacher_cache(
            root,
            cache,
            action_dim=_ACTION_DIM,
            observation_dim=_OBSERVATION_DIM,
        )
        return cache

    def test_content_or_metadata_checksum_tampering_is_rejected(self) -> None:
        for artifact in ("examples", "metadata"):
            with (
                self.subTest(artifact=artifact),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                cache = self._write(root)
                paths = ResidualTeacherCachePaths(root)
                path = paths.examples if artifact == "examples" else paths.metadata
                path.write_bytes(path.read_bytes() + b" ")

                with self.assertRaisesRegex(ValueError, "checksum|canonical"):
                    _load(root, cache)

    def test_semantic_tampering_fails_even_with_recomputed_checksums(self) -> None:
        mutations = (
            "unknown_metadata",
            "heldout_source",
            "source_family_reassignment",
            "target_calls",
        )
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                cache = self._write(root)
                paths = ResidualTeacherCachePaths(root)
                metadata = json.loads(paths.metadata.read_text())

                if mutation == "unknown_metadata":
                    metadata["unexpected"] = True
                elif mutation == "target_calls":
                    metadata["target_calls"] = 1
                else:
                    records = [
                        json.loads(line)
                        for line in paths.examples.read_text().splitlines()
                    ]
                    family = (
                        "transformer"
                        if mutation == "source_family_reassignment"
                        else "modern_cnn"
                    )
                    records[0]["source_family"] = family
                    records[0]["trajectory_id"] = records[0]["trajectory_id"].replace(
                        "classical_cnn", family
                    )
                    paths.examples.write_text(
                        "".join(
                            json.dumps(
                                record,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                            for record in records
                        )
                    )
                    metadata["examples_sha256"] = _rewrite_checksum(paths.examples)
                paths.metadata.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True)
                )
                _rewrite_checksum(paths.metadata)

                with self.assertRaisesRegex(
                    ValueError,
                    "schema|source|target|held.?out|cache",
                ):
                    _load(root, cache)

    def test_noncanonical_jsonl_is_rejected_after_checksum_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = self._write(root)
            paths = ResidualTeacherCachePaths(root)
            records = [
                json.loads(line) for line in paths.examples.read_text().splitlines()
            ]
            paths.examples.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
            )
            examples_sha256 = _rewrite_checksum(paths.examples)
            metadata = json.loads(paths.metadata.read_text())
            metadata["examples_sha256"] = examples_sha256
            paths.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True))
            _rewrite_checksum(paths.metadata)

            with self.assertRaisesRegex(ValueError, "canonical"):
                _load(root, cache)

    def test_partial_cache_pairs_are_rejected_without_regeneration(self) -> None:
        partial_sets = (
            ("teacher_ranker_examples.jsonl",),
            (
                "teacher_ranker_examples.jsonl",
                "teacher_ranker_examples.jsonl.sha256",
            ),
            (
                "teacher_ranker_manifest.json",
                "teacher_ranker_manifest.json.sha256",
            ),
        )
        original = _cache()
        for names in partial_sets:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in names:
                    (root / name).write_text("partial\n")
                factory = mock.Mock(return_value=original)

                with self.assertRaisesRegex(ValueError, "partial|incomplete"):
                    load_or_create_residual_teacher_cache(
                        root,
                        expected_binding=original.binding,
                        expected_protocol=original.protocol,
                        action_dim=_ACTION_DIM,
                        observation_dim=_OBSERVATION_DIM,
                        create=factory,
                    )
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
