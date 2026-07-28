import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from rl_transfer.cifar_config import MacPilotConfig
from rl_transfer.cifar_victim_cache import (
    victim_cache_contract,
    victim_cache_digest,
    victim_code_digest,
)
from rl_transfer.phase2_cache import (
    EXPECTED_VICTIM_FAMILIES,
    MAX_VICTIM_CHECKPOINT_BYTES,
    mirror_verified_victim_cache,
    pinned_manifest_victim_cache_binding,
)


class VerifiedVictimCacheTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        instances_per_family: int = 3,
    ) -> tuple[Path, str, Path, str, dict[str, Path]]:
        source = root / "source"
        contract = {
            "schema_version": 1,
            "dataset": "CIFAR-10",
            "fixture": "authenticated-bank",
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        cache_dir = source / fingerprint[:12]
        cache_dir.mkdir(parents=True)
        victim_instances: dict[str, list[dict[str, object]]] = {}
        checkpoints: dict[str, Path] = {}
        for family in EXPECTED_VICTIM_FAMILIES:
            family_records: list[dict[str, object]] = []
            for index in range(instances_per_family):
                victim_id = f"{family}_victim_{index}"
                path = cache_dir / f"{victim_id}.pt"
                torch.save(
                    {
                        "schema_version": 1,
                        "model": {"weight": torch.ones(1)},
                        "metadata": {
                            "fingerprint": fingerprint,
                            "cache_contract": contract,
                            "training_seed": index,
                        },
                    },
                    path,
                )
                checkpoint_sha = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                path.with_suffix(".pt.sha256").write_text(
                    checkpoint_sha + "\n"
                )
                checkpoints[victim_id] = path
                family_records.append(
                    {
                        "victim_id": victim_id,
                        "checkpoint": f"/remote/cache/{victim_id}.pt",
                        "checkpoint_sha256": checkpoint_sha,
                    }
                )
            victim_instances[family] = family_records
        manifest = root / "study_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_runs": [
                        {
                            "victim_cache_digest": fingerprint,
                            "victim_instances": victim_instances,
                        }
                    ],
                },
                sort_keys=True,
            )
        )
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        manifest.with_suffix(".json.sha256").write_text(
            manifest_sha + "\n"
        )
        return source, fingerprint, manifest, manifest_sha, checkpoints

    def _mirror(
        self,
        source: Path,
        target: Path,
        manifest: Path,
        manifest_sha: str,
        fingerprint: str,
    ) -> dict[str, object]:
        return mirror_verified_victim_cache(
            source,
            target,
            study_manifest_path=manifest,
            expected_study_manifest_sha256=manifest_sha,
            expected_cache_fingerprint=fingerprint,
        )

    def _write_manifest_with_contract(
        self,
        manifest: Path,
        fingerprint: str,
        contract: dict[str, object],
        *,
        freeze_sha256: str,
    ) -> str:
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "runtime_environment": {
                        "pip_freeze_sha256": freeze_sha256,
                    },
                    "source_runs": [
                        {
                            "victim_cache_contract": contract,
                            "victim_cache_digest": fingerprint,
                        },
                        {
                            "victim_cache_contract": contract,
                            "victim_cache_digest": fingerprint,
                        },
                    ],
                },
                sort_keys=True,
            )
        )
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        manifest.with_suffix(".json.sha256").write_text(
            manifest_sha + "\n"
        )
        return manifest_sha

    def _binding_fixture(
        self,
        root: Path,
    ) -> tuple[
        Path,
        str,
        Path,
        str,
        MacPilotConfig,
        str,
        str,
    ]:
        phase1_freeze = (
            "numpy==2.4.1\n"
            "torch==2.13.0\n"
            "torchvision==0.28.0\n"
            "-e git+https://github.com/example/research.git@"
            + ("1" * 40)
            + "#egg=rl_transfer_research\n"
        )
        current_freeze = phase1_freeze.replace("1" * 40, "2" * 40)
        phase1_freeze_path = root / "pip_freeze.txt"
        current_freeze_path = root / "phase2-pip-freeze.txt"
        phase1_freeze_path.write_text(phase1_freeze)
        current_freeze_path.write_text(current_freeze)
        phase1_freeze_sha = hashlib.sha256(
            phase1_freeze.encode()
        ).hexdigest()
        current_freeze_sha = hashlib.sha256(
            current_freeze.encode()
        ).hexdigest()
        content_sha = "c" * 64
        phase1_dataset_version = (
            "torchvision-0.28.0;"
            f"content-sha256={content_sha};"
            f"environment-sha256={phase1_freeze_sha}"
        )
        current_dataset_version = (
            "torchvision-0.28.0;"
            f"content-sha256={content_sha};"
            f"environment-sha256={current_freeze_sha}"
        )
        base = MacPilotConfig.from_json(
            Path("configs/rl_transfer/cifar10_rtx_phase2_base.json")
        )
        fingerprint_config = replace(
            base,
            seed=17,
            split_seed=20260727,
            victim_seed=1000000,
        )
        split_digest = "d" * 64
        contract = victim_cache_contract(
            fingerprint_config,
            split_digest,
            phase1_dataset_version,
            victim_code_digest(),
            "cuda",
        )
        fingerprint = victim_cache_digest(
            fingerprint_config,
            split_digest,
            phase1_dataset_version,
            victim_code_digest(),
            "cuda",
        )
        manifest = root / "study_manifest.json"
        manifest_sha = self._write_manifest_with_contract(
            manifest,
            fingerprint,
            contract,
            freeze_sha256=phase1_freeze_sha,
        )
        return (
            manifest,
            manifest_sha,
            current_freeze_path,
            current_dataset_version,
            fingerprint_config,
            split_digest,
            fingerprint,
        )

    def test_pinned_binding_allows_only_editable_code_commit_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                manifest,
                manifest_sha,
                current_freeze,
                current_dataset_version,
                fingerprint_config,
                split_digest,
                fingerprint,
            ) = self._binding_fixture(root)

            binding = pinned_manifest_victim_cache_binding(
                manifest,
                expected_manifest_sha256=manifest_sha,
                current_dataset_version=current_dataset_version,
                current_freeze_path=current_freeze,
                fingerprint_config=fingerprint_config,
                expected_split_digest=split_digest,
            )

            self.assertEqual(binding.fingerprint, fingerprint)
            self.assertNotEqual(
                binding.dataset_version,
                current_dataset_version,
            )
            self.assertEqual(
                binding.dependency_compatibility,
                "identical_non_project_pins",
            )

    def test_pinned_binding_rejects_dependency_or_split_drift(
        self,
    ) -> None:
        for mutation in ("dependency", "split"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (
                    manifest,
                    manifest_sha,
                    current_freeze,
                    current_dataset_version,
                    fingerprint_config,
                    split_digest,
                    _,
                ) = self._binding_fixture(root)
                if mutation == "dependency":
                    changed = current_freeze.read_text().replace(
                        "numpy==2.4.1",
                        "numpy==9.9.9",
                    )
                    current_freeze.write_text(changed)
                    current_dataset_version = current_dataset_version.rsplit(
                        "=", 1
                    )[0] + "=" + hashlib.sha256(changed.encode()).hexdigest()
                else:
                    split_digest = "e" * 64

                with self.assertRaisesRegex(
                    ValueError,
                    "dependency|contract",
                ):
                    pinned_manifest_victim_cache_binding(
                        manifest,
                        expected_manifest_sha256=manifest_sha,
                        current_dataset_version=current_dataset_version,
                        current_freeze_path=current_freeze,
                        fingerprint_config=fingerprint_config,
                        expected_split_digest=split_digest,
                    )

    def test_verified_cache_uses_independent_atomic_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fingerprint, manifest, manifest_sha, checkpoints = (
                self._fixture(root)
            )
            target = root / "target"

            report = self._mirror(
                source,
                target,
                manifest,
                manifest_sha,
                fingerprint,
            )

            original = checkpoints["classical_cnn_victim_0"]
            mirrored = target / fingerprint[:12] / original.name
            self.assertEqual(report["checkpoint_count"], 9)
            self.assertEqual(report["cache_fingerprints"], [fingerprint])
            self.assertEqual(mirrored.read_bytes(), original.read_bytes())
            self.assertEqual(report["materialization"], ["atomic_copy"])
            self.assertNotEqual(mirrored.stat().st_ino, original.stat().st_ino)
            self.assertTrue(report["all_verified"])
            self.assertNotIn("source", report)
            self.assertNotIn("destination", report)

    def test_replaced_sidecar_or_checkpoint_fails_pinned_authenticity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fingerprint, manifest, manifest_sha, checkpoints = (
                self._fixture(root)
            )
            checkpoint = checkpoints["modern_cnn_victim_0"]
            checkpoint.write_bytes(checkpoint.read_bytes() + b"replacement")
            replacement_sha = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            checkpoint.with_suffix(".pt.sha256").write_text(
                replacement_sha + "\n"
            )

            with self.assertRaisesRegex(ValueError, "checksum"):
                self._mirror(
                    source,
                    root / "target",
                    manifest,
                    manifest_sha,
                    fingerprint,
                )

    def test_replaced_manifest_and_sidecar_cannot_override_pinned_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fingerprint, manifest, manifest_sha, _ = self._fixture(
                root
            )
            manifest.write_text(manifest.read_text() + " ")
            manifest.with_suffix(".json.sha256").write_text(
                hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n"
            )

            with self.assertRaisesRegex(ValueError, "manifest checksum"):
                self._mirror(
                    source,
                    root / "target",
                    manifest,
                    manifest_sha,
                    fingerprint,
                )

    def test_concurrent_source_replacement_during_copy_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fingerprint, manifest, manifest_sha, _ = self._fixture(
                root
            )

            def replace_during_copy(source_path, destination_path):
                Path(source_path).write_bytes(b"concurrent-replacement")
                Path(destination_path).write_bytes(
                    Path(source_path).read_bytes()
                )

            with (
                mock.patch(
                    "rl_transfer.phase2_cache.shutil.copy2",
                    side_effect=replace_during_copy,
                ),
                self.assertRaisesRegex(ValueError, "changed"),
            ):
                self._mirror(
                    source,
                    root / "target",
                    manifest,
                    manifest_sha,
                    fingerprint,
                )

    def test_missing_or_extra_checkpoint_fails_closed(self) -> None:
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, fingerprint, manifest, manifest_sha, checkpoints = (
                    self._fixture(root)
                )
                if mutation == "missing":
                    checkpoint = checkpoints["transformer_victim_2"]
                    checkpoint.unlink()
                    checkpoint.with_suffix(".pt.sha256").unlink()
                else:
                    extra = source / fingerprint[:12] / "extra.pt"
                    extra.write_bytes(b"extra")
                    extra.with_suffix(".pt.sha256").write_text(
                        hashlib.sha256(extra.read_bytes()).hexdigest() + "\n"
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "missing or extra",
                ):
                    self._mirror(
                        source,
                        root / "target",
                        manifest,
                        manifest_sha,
                        fingerprint,
                    )

    def test_oversized_count_or_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fingerprint, manifest, manifest_sha, _ = self._fixture(
                root,
                instances_per_family=4,
            )
            with self.assertRaisesRegex(ValueError, "count"):
                self._mirror(
                    source,
                    root / "target",
                    manifest,
                    manifest_sha,
                    fingerprint,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fingerprint, manifest, manifest_sha, checkpoints = (
                self._fixture(root)
            )
            checkpoint = checkpoints["classical_cnn_victim_0"]
            with checkpoint.open("r+b") as handle:
                handle.truncate(MAX_VICTIM_CHECKPOINT_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "safe size"):
                self._mirror(
                    source,
                    root / "target",
                    manifest,
                    manifest_sha,
                    fingerprint,
                )


if __name__ == "__main__":
    unittest.main()
