"""Command-line entry point for the source-only Phase 2 Stage A screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .phase2_temperature_screen import (
    FOLDS,
    STAGE_A_TEMPERATURES,
    StageARequest,
    build_stage_a_dry_run,
    load_phase1_source_selection,
    run_temperature_screen_from_datasets,
)
from .reproducibility import tree_digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase2-temperature-screen",
        description=(
            "Diagnose frozen Phase 1 policies on verified exact-source "
            "victims; results do not select the new Phase 2 architecture"
        ),
    )
    parser.add_argument("--phase1-manifest", type=Path, required=True)
    parser.add_argument("--phase1-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--fold", choices=FOLDS, action="append")
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=600.0,
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _request(arguments: argparse.Namespace) -> StageARequest:
    manifest = arguments.phase1_manifest.resolve()
    root = (
        arguments.phase1_root.resolve()
        if arguments.phase1_root is not None
        else manifest.parent
    )
    return StageARequest(
        phase1_manifest=manifest,
        phase1_root=root,
        output_dir=arguments.output_dir,
        data_root=arguments.data_root,
        seeds=tuple(arguments.seed or (17,)),
        folds=tuple(arguments.fold or FOLDS),
        temperatures=STAGE_A_TEMPERATURES,
        deadline_seconds=arguments.deadline_seconds,
        device="cuda",
        download=arguments.download,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    request = _request(arguments)
    selection = load_phase1_source_selection(request)
    if arguments.dry_run:
        print(
            json.dumps(
                build_stage_a_dry_run(request, selection),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Stage A requires an available CUDA device")
    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError(
            "install the vision extra before running Stage A"
        ) from error
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(request.data_root),
        train=True,
        transform=ToTensor(),
        download=request.download,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(request.data_root),
        train=False,
        transform=ToTensor(),
        download=request.download,
    )
    result = run_temperature_screen_from_datasets(
        request,
        train_dataset,
        test_dataset,
        dataset_version=f"torchvision-{torchvision.__version__}",
        dataset_content_sha256=tree_digest(request.data_root),
        progress=print,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
