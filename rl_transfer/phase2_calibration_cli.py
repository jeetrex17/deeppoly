"""CLI for the bounded frozen Phase 2 calibration diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Sequence

import torch

from .phase2_calibration_screen import (
    CALIBRATION_MAX_SECONDS,
    Phase2CalibrationRequest,
    build_calibration_dry_run,
    load_phase2_calibration_source,
    run_calibration_from_datasets,
)
from .reproducibility import seed_everything, tree_digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase2-calibration-diagnostic",
        description=(
            "Replay completed frozen Phase 2 policies at five temperatures "
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
        default=CALIBRATION_MAX_SECONDS,
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _request(arguments: argparse.Namespace) -> Phase2CalibrationRequest:
    manifest = arguments.source_manifest.resolve()
    source_root = (
        arguments.source_root.resolve()
        if arguments.source_root is not None
        else manifest.parent
    )
    return Phase2CalibrationRequest(
        source_manifest=manifest,
        source_root=source_root,
        output_dir=arguments.output_dir,
        data_root=arguments.data_root,
        deadline_seconds=arguments.deadline_seconds,
        device="cuda",
        download=arguments.download,
    )


def _runtime_environment(torchvision_version: str) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    payload: dict[str, object] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {**payload, "environment_sha256": digest}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    request = _request(arguments)
    source = load_phase2_calibration_source(request)
    if arguments.dry_run:
        print(
            json.dumps(
                build_calibration_dry_run(request, source),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("the Phase 2 calibration diagnostic requires CUDA")
    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError("install the vision extra before calibration") from error
    seed_everything(17)
    runtime_environment = _runtime_environment(torchvision.__version__)
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
    result = run_calibration_from_datasets(
        request,
        train_dataset,
        test_dataset,
        dataset_version=f"torchvision-{torchvision.__version__}",
        dataset_content_sha256=tree_digest(request.data_root),
        runtime_environment=runtime_environment,
        progress=print,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
