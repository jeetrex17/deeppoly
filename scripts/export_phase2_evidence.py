#!/usr/bin/env python3
"""Export the verified RTX Phase 2 screen into a compact Git bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rl_transfer.phase2_export import export_phase2_evidence


DEFAULT_SOURCE = Path(
    "output/rl_transfer/cifar10_rtx_phase2_screen"
)
DEFAULT_OUTPUT = Path(
    "docs/research/cifar10_rtx_phase2_stage_b"
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the RTX Phase 2 Stage B archive and export "
            "portable source-only evidence."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Phase 2 screen artifact root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="compact evidence output directory",
    )
    arguments = parser.parse_args(argv)
    summary = export_phase2_evidence(
        arguments.source,
        arguments.output,
    )
    print(
        json.dumps(
            {
                "output": arguments.output.as_posix(),
                "status": summary["status"],
                "promotion_passed": summary["promotion"]["passed"],
                "verified_runs": summary["integrity"][
                    "verified_runs"
                ],
                "target_calls": summary["target_evaluation"][
                    "target_calls"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
