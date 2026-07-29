from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from rl_transfer.verified_artifacts import load_verified_json


class VerifiedJsonSecurityTests(unittest.TestCase):
    def test_loader_rejects_symlinked_payload_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.json"
            outside.write_text('{"safe":true}')
            digest = hashlib.sha256(outside.read_bytes()).hexdigest()
            payload = root / "payload.json"
            sidecar = root / "payload.json.sha256"
            payload.symlink_to(outside)
            sidecar.write_text(digest + "\n")

            with self.assertRaisesRegex(ValueError, "symlink"):
                load_verified_json(payload)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload.json"
            payload.write_text('{"safe":true}')
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            digest_file = root / "digest.txt"
            digest_file.write_text(digest + "\n")
            payload.with_suffix(".json.sha256").symlink_to(digest_file)

            with self.assertRaisesRegex(ValueError, "symlink"):
                load_verified_json(payload)

    def test_loader_rejects_oversized_json_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            with path.open("wb") as handle:
                handle.seek(16 * 1024 * 1024)
                handle.write(b"{}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".json.sha256").write_text(digest + "\n")

            with self.assertRaisesRegex(ValueError, "size"):
                load_verified_json(path)


if __name__ == "__main__":
    unittest.main()
