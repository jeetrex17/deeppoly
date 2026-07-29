from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "d1_remote.sh"
CONTROL = (
    REPOSITORY_ROOT
    / "output"
    / "rl_transfer"
    / "cifar10_rtx_d1_residual_20260729_control"
)


class D1RemoteControllerTests(unittest.TestCase):
    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("bash", str(SCRIPT), *arguments),
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_internal_attempt_path_accepts_only_exact_control_child(self) -> None:
        valid = CONTROL / "full-20260729T010203Z"
        self.assertEqual(
            self._run("_validate_attempt", str(valid)).returncode,
            0,
        )
        invalid = (
            Path("/tmp/full-20260729T010203Z"),
            CONTROL / "full-invalid",
            CONTROL.parent / "full-20260729T010203Z",
            valid / "nested",
        )
        for path in invalid:
            with self.subTest(path=path):
                self.assertNotEqual(
                    self._run("_validate_attempt", str(path)).returncode,
                    0,
                )

    def test_only_interrupt_exit_codes_are_retryable(self) -> None:
        for code in ("130", "143"):
            with self.subTest(code=code):
                self.assertEqual(
                    self._run("_retryable_exit", code).returncode,
                    0,
                )
        for code in ("0", "1", "2", "124", "137", "139"):
            with self.subTest(code=code):
                self.assertNotEqual(
                    self._run("_retryable_exit", code).returncode,
                    0,
                )

    def test_controller_does_not_rely_on_optimizable_python_assertions(self) -> None:
        self.assertNotIn("assert ", SCRIPT.read_text())

    def test_inner_deadline_has_cleanup_margin_inside_hard_eight_hour_cap(
        self,
    ) -> None:
        script = SCRIPT.read_text()

        def integer_constant(name: str) -> int:
            match = re.search(rf"^{name}=([0-9]+)$", script, flags=re.MULTILINE)
            self.assertIsNotNone(match, name)
            return int(match.group(1))  # type: ignore[union-attr]

        inner = integer_constant("INNER_DEADLINE_SECONDS")
        outer = integer_constant("OUTER_TIMEOUT_SECONDS")
        grace = integer_constant("OUTER_KILL_GRACE_SECONDS")

        self.assertGreaterEqual(outer - inner, 60)
        self.assertLessEqual(outer + grace, 8 * 60 * 60)
        self.assertIn('--deadline-seconds "$INNER_DEADLINE_SECONDS"', script)


if __name__ == "__main__":
    unittest.main()
