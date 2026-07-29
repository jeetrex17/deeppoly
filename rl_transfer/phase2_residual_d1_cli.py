"""CLI for the bounded source-only D1 residual-ranker diagnostic."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .phase2_calibration_cli import _runtime_environment
from .phase2_residual_d1 import D1_MAX_SECONDS, ResidualD1Request
from .phase2_residual_d1_runner import (
    load_residual_d1_source,
)
from .phase2_residual_d1_smoke import run_residual_d1_gpu_smoke
from .phase2_residual_d1_study import (
    build_residual_d1_study_dry_run,
    run_residual_d1_study_from_datasets,
)
from .reproducibility import seed_everything, tree_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase2-residual-d1",
        description=(
            "Run the bounded D1a and conditional D1b residual-ranker study "
            "on verified source victims only"
        ),
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10"))
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=D1_MAX_SECONDS,
    )
    parser.add_argument("--download", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    return parser


def _request(
    arguments: argparse.Namespace,
    *,
    smoke_test: bool,
) -> ResidualD1Request:
    source_manifest = arguments.source_manifest.resolve()
    source_root = (
        arguments.source_root.resolve()
        if arguments.source_root is not None
        else source_manifest.parent
    )
    return ResidualD1Request(
        source_manifest=source_manifest,
        source_root=source_root,
        output_dir=(
            arguments.output_dir.resolve()
            if smoke_test
            else arguments.output_dir.resolve() / "d1a"
        ),
        data_root=arguments.data_root,
        deadline_seconds=arguments.deadline_seconds,
        download=arguments.download,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    request = _request(arguments, smoke_test=arguments.smoke_test)
    source = load_residual_d1_source(request)
    if arguments.dry_run:
        print(
            json.dumps(
                build_residual_d1_study_dry_run(request, source),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("D1 execution requires the designated CUDA workstation")
    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError("install the vision extra before running D1") from error

    seed_everything(request.seed)
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(request.data_root),
        train=True,
        transform=ToTensor(),
        download=arguments.download,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(request.data_root),
        train=False,
        transform=ToTensor(),
        download=arguments.download,
    )
    dataset_content_sha256 = tree_digest(request.data_root)
    runtime_environment = _runtime_environment(torchvision.__version__)
    result = (
        run_residual_d1_gpu_smoke(
            request,
            train_dataset,
            test_dataset,
            dataset_content_sha256=dataset_content_sha256,
            runtime_environment=runtime_environment,
            progress=print,
        )
        if arguments.smoke_test
        else run_residual_d1_study_from_datasets(
            request,
            train_dataset,
            test_dataset,
            dataset_version=f"torchvision-{torchvision.__version__}",
            dataset_content_sha256=dataset_content_sha256,
            runtime_environment=runtime_environment,
            progress=print,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
