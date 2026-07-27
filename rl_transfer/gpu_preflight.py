"""CUDA preflight and conservative call-volume estimates for the RTX study."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import time

import torch

from .cifar_models import build_cifar_victims
from .cifar_pilot import MacPilotConfig
from .gpu_config import RTXPublicationConfig


EVALUATED_METHODS = 6
PPO_TRAINING_VARIANTS = 2


def cuda_preflight(minimum_memory_gib: float = 10.0) -> dict[str, object]:
    if not math.isfinite(minimum_memory_gib) or minimum_memory_gib <= 0:
        raise ValueError("minimum CUDA memory must be positive and finite")
    available = torch.cuda.is_available()
    if not available:
        return {
            "passed": False,
            "cuda_available": False,
            "error": "CUDA is not available in this Python environment",
            "torch_version": torch.__version__,
        }
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    total_memory_gib = properties.total_memory / (1024**3)
    return {
        "passed": total_memory_gib >= minimum_memory_gib,
        "cuda_available": True,
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_gib": total_memory_gib,
        "minimum_memory_gib": minimum_memory_gib,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_version": torch.__version__,
    }


def benchmark_single_image_calls(
    *,
    device: str = "cuda",
    calls: int = 500,
    warmup: int = 50,
) -> dict[str, object]:
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("the publication benchmark requires an available CUDA device")
    if calls < 10 or warmup < 1:
        raise ValueError("benchmark calls and warmup must be positive")
    victims = build_cifar_victims(seed=1, profile="research")
    family_metrics: dict[str, dict[str, float | int]] = {}
    for family, (_, victim) in victims.items():
        victim = victim.to(device).eval()
        images = torch.rand((1, 3, 32, 32), device=device)
        with torch.inference_mode():
            for _ in range(warmup):
                victim(images)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(calls):
                victim(images)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        family_metrics[family] = {
            "calls": calls,
            "elapsed_seconds": elapsed,
            "calls_per_second": calls / elapsed,
            "milliseconds_per_call": 1000 * elapsed / calls,
        }
        victim.to("cpu")
        del victim
        torch.cuda.empty_cache()
    conservative_rate = min(
        float(metrics["calls_per_second"])
        for metrics in family_metrics.values()
    )
    return {
        "device": device,
        "calls": calls,
        "families": family_metrics,
        "calls_per_second": conservative_rate,
        "milliseconds_per_call": 1000 / conservative_rate,
    }


def estimate_study_calls(config_path: Path) -> dict[str, object]:
    study = RTXPublicationConfig.from_json(config_path)
    base_path = Path(study.base_config)
    if not base_path.is_file():
        for candidate in (config_path.resolve().parent, *config_path.resolve().parents):
            repository_candidate = candidate / study.base_config
            if repository_candidate.is_file():
                base_path = repository_candidate
                break
    base = MacPilotConfig.from_json(base_path)
    run_count = len(study.seeds) * len(study.target_families)
    teacher_calls_per_episode = (
        1 + base.behavior_cloning_steps
        if base.behavior_cloning_teacher == "gradient"
        else 1 + base.behavior_cloning_candidates * base.behavior_cloning_steps
    )
    teacher_calls_per_run = (
        base.behavior_cloning_episodes
        + base.behavior_cloning_validation_episodes
    ) * teacher_calls_per_episode
    ppo_calls_per_run = (
        PPO_TRAINING_VARIANTS
        * base.policy_episodes
        * base.query_budget
    )
    source_victims_per_run = (
        len(study.target_families) - 1
    ) * (
        base.source_instances_per_family
        + base.source_holdout_instances_per_family
    )
    source_evaluation_calls_per_run = (
        source_victims_per_run
        * base.source_evaluation_images
        * EVALUATED_METHODS
        * base.query_budget
    )
    target_evaluation_calls_per_run = (
        base.target_instances_per_family
        * base.outer_test_images
        * EVALUATED_METHODS
        * base.query_budget
    )
    per_run = {
        "teacher_upper_bound": teacher_calls_per_run,
        "ppo_upper_bound": ppo_calls_per_run,
        "source_evaluation_upper_bound": source_evaluation_calls_per_run,
        "target_evaluation_upper_bound": target_evaluation_calls_per_run,
    }
    source_phase = run_count * (
        teacher_calls_per_run
        + ppo_calls_per_run
        + source_evaluation_calls_per_run
    )
    target_phase = run_count * target_evaluation_calls_per_run
    return {
        "study": asdict(study),
        "run_count": run_count,
        "evaluated_methods": EVALUATED_METHODS,
        "ppo_training_variants": PPO_TRAINING_VARIANTS,
        "unique_victim_fits": (
            len(study.target_families)
            * max(
                base.target_instances_per_family,
                base.source_instances_per_family
                + base.source_holdout_instances_per_family,
            )
        ),
        "per_run": per_run,
        "source_phase_upper_bound": source_phase,
        "target_phase_upper_bound": target_phase,
        "total_upper_bound": source_phase + target_phase,
        "notes": [
            "bounds count attack-model calls, not batched victim fitting",
            "target phase revalidates cached raw source evidence without new source queries",
            "victim checkpoints are shared across policy seeds",
            "early attack success can reduce the realized call count",
            "the target phase is skipped when the source gate fails",
        ],
    }


def estimate_wall_time_hours(
    total_calls: int,
    calls_per_second: float,
    *,
    overhead_multiplier: float = 1.5,
) -> float:
    if (
        not isinstance(total_calls, int)
        or isinstance(total_calls, bool)
        or total_calls < 0
    ):
        raise ValueError("total_calls must be a non-negative integer")
    if not math.isfinite(calls_per_second) or calls_per_second <= 0:
        raise ValueError("calls_per_second must be positive and finite")
    if not math.isfinite(overhead_multiplier) or overhead_multiplier < 1:
        raise ValueError("overhead_multiplier must be at least one")
    return total_calls / calls_per_second * overhead_multiplier / 3600
