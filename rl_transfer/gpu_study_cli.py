import argparse
import json
from pathlib import Path

from .gpu_study import load_validated_study_config, run_gpu_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the source-gated RTX publication-candidate study"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("source", "all"), default="all")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        config, base, _ = load_validated_study_config(args.config)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "name": config.name,
                    "device": config.device,
                    "seeds": list(config.seeds),
                    "target_families": list(config.target_families),
                    "victim_seed": config.victim_seed,
                    "target_instances_per_family": (
                        config.target_instances_per_family
                    ),
                    "component_ablations": base.train_ablation_policies,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run_gpu_study(
        args.config,
        phase=args.phase,
        progress=lambda message: print(f"[rtx-study] {message}", flush=True),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
