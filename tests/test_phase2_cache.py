import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rl_transfer.phase2_cache import (
    EXPECTED_VICTIM_FAMILIES,
    MAX_VICTIM_CHECKPOINT_BYTES,
    mirror_verified_victim_cache,
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
