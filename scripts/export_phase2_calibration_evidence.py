#!/usr/bin/env python3
"""Export the verified Phase 2 calibration run into a compact Git bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rl_transfer.phase2_calibration_export import (
    export_phase2_calibration_evidence,
)


DEFAULT_SOURCE = Path("output/rl_transfer/cifar10_rtx_phase2_calibration")
DEFAULT_OUTPUT = Path("docs/research/cifar10_rtx_phase2_calibration")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the source-only Phase 2 calibration archive "
            "and export portable evidence."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Phase 2 calibration artifact root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="compact evidence output directory",
    )
    parser.add_argument(
        "--attempt-log",
        action="append",
        default=[],
        type=Path,
        help="execution log to hash without publishing its contents",
    )
    arguments = parser.parse_args(argv)
    summary = export_phase2_calibration_evidence(
        arguments.source,
        arguments.output,
        attempt_logs=tuple(arguments.attempt_log),
    )
    print(
        json.dumps(
            {
                "output": arguments.output.as_posix(),
                "status": summary["status"],
                "tested_global_temperature_qualified": summary["decision"][
                    "tested_global_temperature_qualified"
                ],
                "stop_tested_global_temperature_protocol": summary["decision"][
                    "stop_tested_global_temperature_protocol"
                ],
                "target_calls": summary["target_evaluation"]["target_calls"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
