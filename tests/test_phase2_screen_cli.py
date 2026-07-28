import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from rl_transfer.cifar_config import MacPilotConfig
from rl_transfer.phase2_config import Phase2ScreenConfig


CONFIG_PATH = Path("configs/rl_transfer/cifar10_rtx_phase2_screen.json")


class Phase2CLITests(unittest.TestCase):
    def test_cli_dry_run_does_not_launch_training(self) -> None:
        from rl_transfer import phase2_screen_cli

        config = Phase2ScreenConfig.from_json(CONFIG_PATH)
        base = MacPilotConfig.from_json(Path(config.base_config))
        plan = {"mode": "source_only_screen"}
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "phase2-screen",
                    "--config",
                    str(CONFIG_PATH),
                    "--dry-run",
                ],
            ),
            mock.patch.object(
                phase2_screen_cli,
                "load_validated_phase2_config",
                return_value=(config, base, CONFIG_PATH),
            ),
            mock.patch.object(
                phase2_screen_cli,
                "build_phase2_dry_run",
                return_value=plan,
            ),
            mock.patch.object(
                phase2_screen_cli,
                "run_phase2_screen",
            ) as runner,
            redirect_stdout(io.StringIO()) as output,
        ):
            phase2_screen_cli.main()
        runner.assert_not_called()
        self.assertEqual(json.loads(output.getvalue()), plan)

    def test_cli_has_no_target_or_all_phase_argument(self) -> None:
        from rl_transfer import phase2_screen_cli

        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "phase2-screen",
                    "--config",
                    str(CONFIG_PATH),
                    "--phase",
                    "target",
                ],
            ),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            phase2_screen_cli.main()


if __name__ == "__main__":
    unittest.main()
