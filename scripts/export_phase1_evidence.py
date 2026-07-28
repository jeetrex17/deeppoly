#!/usr/bin/env python3
"""Export the verified RTX Phase 1 study into a compact Git bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_transfer.phase1_export import export_phase1_evidence


DEFAULT_SOURCE = Path(
    "output/rl_transfer/cifar10_rtx_publication/cifar10-rtx-publication"
)
DEFAULT_OUTPUT = Path("docs/research/cifar10_rtx_phase1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Verify the RTX Phase 1 archive and export portable evidence.")
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Phase 1 study artifact root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="compact evidence output directory",
    )
    arguments = parser.parse_args()
    summary = export_phase1_evidence(
        arguments.source,
        arguments.output,
    )
    print(
        json.dumps(
            {
                "output": arguments.output.as_posix(),
                "status": summary["status"],
                "verified_runs": summary["integrity"]["verified_runs"],
                "target_calls": summary["target_evaluation"]["target_calls"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
