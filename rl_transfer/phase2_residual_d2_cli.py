"""CUDA-only command line entrypoint for the D2 source-only study."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .phase2_calibration_cli import _runtime_environment
from .phase2_residual_d2 import ResidualD2Request
from .phase2_residual_d2_runner import run_residual_d2_from_datasets
from .reproducibility import seed_everything, tree_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase2-residual-d2",
        description="Run the preregistered CUDA-only D2 source-only study",
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--deadline-seconds", type=float, default=8 * 60 * 60.0)
    parser.add_argument("--download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    source_manifest = arguments.source_manifest.resolve()
    request = ResidualD2Request(
        source_manifest=source_manifest,
        source_root=(arguments.source_root or source_manifest.parent).resolve(),
        output_dir=arguments.output_dir.resolve(),
        data_root=arguments.data_root.resolve(),
        download=arguments.download,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("D2 may only run on the designated RTX CUDA host")
    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError("D2 requires the vision extra on the RTX host") from error
    seed_everything(17)
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(request.data_root), train=True, transform=ToTensor(), download=request.download
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(request.data_root), train=False, transform=ToTensor(), download=request.download
    )
    result = run_residual_d2_from_datasets(
        request, train_dataset, test_dataset,
        dataset_version=f"torchvision-{torchvision.__version__}",
        dataset_content_sha256=tree_digest(request.data_root),
        runtime_environment=_runtime_environment(torchvision.__version__),
        deadline_seconds=arguments.deadline_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
