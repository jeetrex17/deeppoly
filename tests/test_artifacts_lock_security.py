from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl_transfer.artifacts import exclusive_file_lock


class ExclusiveFileLockSecurityTests(unittest.TestCase):
    def test_lock_rejects_a_symlink_without_touching_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.txt"
            outside.write_bytes(b"outside-must-not-change")
            lock = root / "study.lock"
            lock.symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "lock|symlink|regular"):
                with exclusive_file_lock(lock):
                    self.fail("symlinked lock must never be acquired")

            self.assertEqual(outside.read_bytes(), b"outside-must-not-change")

    def test_lock_still_serializes_a_regular_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "study.lock"

            with exclusive_file_lock(lock):
                self.assertTrue(lock.is_file())
                self.assertFalse(lock.is_symlink())


if __name__ == "__main__":
    unittest.main()
