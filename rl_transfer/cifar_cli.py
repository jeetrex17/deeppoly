import argparse
import json
from pathlib import Path

from .cifar_pilot import run_cifar_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded CIFAR-10 Mac pilot")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl_transfer/cifar10_m4_pilot.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_cifar_pilot(
        args.config,
        resume=not args.no_resume,
        device=args.device,
        output_dir=args.output_dir,
        progress=lambda message: print(f"[cifar-pilot] {message}", flush=True),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
