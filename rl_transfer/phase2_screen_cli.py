"""Command-line entry point for source-only Phase 2 screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase2_screen import (
    build_phase2_dry_run,
    load_validated_phase2_config,
    run_phase2_screen,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the time-bounded source-only Phase 2 screen. "
            "This command has no target-evaluation mode."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the source-only runtime plan",
    )
    args = parser.parse_args()
    if args.dry_run:
        config, base, _ = load_validated_phase2_config(args.config)
        result = build_phase2_dry_run(config, base)
    else:
        result = run_phase2_screen(
            args.config,
            progress=lambda message: print(
                f"[phase2-screen] {message}",
                flush=True,
            ),
        )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
