import argparse
import json
from pathlib import Path

from .cifar_study import run_cifar_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-fold CIFAR-10 transfer study")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_cifar_study(
        args.config,
        resume=not args.no_resume,
        device=args.device,
        progress=lambda message: print(f"[cifar-study] {message}", flush=True),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
