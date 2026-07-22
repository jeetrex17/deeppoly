"""Multi-seed, multi-fold orchestration for the CIFAR-10 Mac study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Callable, Mapping, Sequence

from torch.utils.data import Dataset

from .cifar_pilot import MacPilotConfig, _code_digest, run_cifar_pilot_from_datasets
from .research_metrics import asr_query_auc


FAMILIES = ("classical_cnn", "modern_cnn", "transformer")
LEARNED_METHOD = "groupdro_recurrent_ppo_stochastic"
CONTROL_METHODS = ("random_action", "bandit_action", "score_greedy")
REQUIRED_METHODS = (LEARNED_METHOD, *CONTROL_METHODS)

_T_CRITICAL_95 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)


@dataclass(frozen=True)
class CIFARStudyConfig:
    schema_version: int
    name: str
    research_valid: bool
    base_config: str
    output_dir: str
    seeds: tuple[int, ...]
    target_families: tuple[str, ...]
    minimum_seeds: int = 3
    minimum_asr_gain: float = 0.01
    minimum_auc_gain: float = 0.005
    entropy_min: float = 0.10
    entropy_max: float = 0.95

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "target_families", tuple(self.target_families))
        if self.schema_version != 1 or self.research_valid is not False:
            raise ValueError("the bounded CIFAR study must use schema 1 and remain exploratory")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.name) is None:
            raise ValueError("study name must be a safe filename component")
        if len(self.seeds) != len(set(self.seeds)) or not self.seeds:
            raise ValueError("study seeds must be non-empty and unique")
        if not self.target_families or any(family not in FAMILIES for family in self.target_families):
            raise ValueError("study target families must come from the CIFAR registry")
        if len(self.target_families) != len(set(self.target_families)):
            raise ValueError("study target families must be unique")
        if not isinstance(self.minimum_seeds, int) or isinstance(self.minimum_seeds, bool):
            raise ValueError("minimum_seeds must be an integer")
        if self.minimum_seeds < 3:
            raise ValueError("minimum_seeds must be at least three")
        gains = (self.minimum_asr_gain, self.minimum_auc_gain)
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in gains):
            raise ValueError("minimum practical gains must be finite and in (0, 1]")
        if not 0.0 <= self.entropy_min < self.entropy_max <= 1.0:
            raise ValueError("entropy bounds must satisfy 0 <= min < max <= 1")

    @classmethod
    def from_json(cls, path: Path) -> "CIFARStudyConfig":
        return cls(**json.loads(path.read_text()))


def _mean_interval(
    values: Sequence[float],
    bounds: tuple[float, float] = (0.0, 1.0),
) -> dict[str, object]:
    if not values:
        raise ValueError("cannot summarize an empty metric sequence")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("study metrics must be finite")
    if any(not bounds[0] <= value <= bounds[1] for value in values):
        raise ValueError("study metric falls outside its valid range")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"mean": mean, "ci95": None, "values": list(values)}
    degrees_of_freedom = len(values) - 1
    critical = (
        _T_CRITICAL_95[degrees_of_freedom - 1]
        if degrees_of_freedom <= len(_T_CRITICAL_95)
        else 1.96
    )
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "ci95": [
            max(bounds[0], mean - half_width),
            min(bounds[1], mean + half_width),
        ],
        "values": list(values),
    }


def _paired_interval(
    learned: Mapping[int, float],
    control: Mapping[int, float],
) -> dict[str, object]:
    if set(learned) != set(control):
        raise ValueError("paired comparisons require exactly aligned seeds")
    differences = [learned[seed] - control[seed] for seed in sorted(learned)]
    return _mean_interval(differences, bounds=(-1.0, 1.0))


def _validate_expected_axis(values: Sequence[object], label: str) -> tuple[object, ...]:
    normalized = tuple(values)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"expected {label} must be non-empty and unique")
    return normalized


def summarize_study(
    runs: Sequence[dict[str, object]],
    expected_families: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
    minimum_seeds: int = 3,
    minimum_asr_gain: float = 0.01,
    minimum_auc_gain: float = 0.005,
    entropy_bounds: tuple[float, float] = (0.10, 0.95),
) -> dict[str, object]:
    """Aggregate folds/seeds and apply a fail-closed RL promotion gate."""

    if not runs:
        raise ValueError("at least one completed run is required")
    if minimum_seeds < 3:
        raise ValueError("minimum_seeds must be at least three")
    if any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in (minimum_asr_gain, minimum_auc_gain)
    ):
        raise ValueError("minimum practical gains must be finite and in (0, 1]")
    if not 0.0 <= entropy_bounds[0] < entropy_bounds[1] <= 1.0:
        raise ValueError("entropy bounds must satisfy 0 <= min < max <= 1")
    observed_families = tuple(dict.fromkeys(str(run["target_family"]) for run in runs))
    observed_seeds = tuple(dict.fromkeys(int(run["seed"]) for run in runs))
    families = tuple(
        str(value)
        for value in _validate_expected_axis(
            observed_families if expected_families is None else expected_families,
            "families",
        )
    )
    seeds = tuple(
        int(value)
        for value in _validate_expected_axis(
            observed_seeds if expected_seeds is None else expected_seeds,
            "seeds",
        )
    )
    if any(family not in FAMILIES for family in families):
        raise ValueError("expected families must come from the CIFAR registry")

    expected_pairs = {(family, seed) for family in families for seed in seeds}
    observed_pairs: set[tuple[str, int]] = set()
    method_names: set[str] | None = None
    grouped: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    victim_gates: dict[tuple[str, int], bool] = {}
    for run in runs:
        if run.get("status") != "complete":
            raise ValueError("all study runs must be complete")
        family = str(run["target_family"])
        seed = int(run["seed"])
        pair = (family, seed)
        if pair not in expected_pairs:
            raise ValueError(f"unexpected study run: {family}/seed-{seed}")
        if pair in observed_pairs:
            raise ValueError(f"duplicate study run: {family}/seed-{seed}")
        observed_pairs.add(pair)
        victim_gate = run.get("victim_accuracy_gate")
        if not isinstance(victim_gate, Mapping) or not isinstance(victim_gate.get("passed"), bool):
            raise ValueError("every run requires an explicit boolean victim-quality gate")
        victim_gates[pair] = bool(victim_gate["passed"])
        evaluation = run.get("evaluation")
        if not isinstance(evaluation, Mapping) or not evaluation:
            raise ValueError("every run requires non-empty evaluation metrics")
        run_methods = {str(method) for method in evaluation}
        if not set(REQUIRED_METHODS).issubset(run_methods):
            raise ValueError("every run must include RL, random, bandit, and score-greedy methods")
        if method_names is None:
            method_names = run_methods
        elif run_methods != method_names:
            raise ValueError("evaluation methods must align exactly across every run")
        family_metrics = grouped.setdefault(family, {})
        alignment: tuple[int, str, int, tuple[int, ...]] | None = None
        for method, metrics in evaluation.items():
            if not isinstance(metrics, Mapping):
                raise ValueError("method metrics must be mappings")
            curve = metrics.get("asr_at_budgets")
            if not isinstance(curve, Mapping) or not curve:
                raise ValueError("ASR curves must be non-empty")
            integer_curve = {int(key): float(value) for key, value in curve.items()}
            if len(integer_curve) != len(curve) or any(budget < 0 for budget in integer_curve):
                raise ValueError("ASR curve budgets must be unique non-negative integers")
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in integer_curve.values()
            ):
                raise ValueError("ASR curves must contain finite probabilities")
            ordered_curve = dict(sorted(integer_curve.items()))
            if any(
                later < earlier
                for earlier, later in zip(
                    ordered_curve.values(),
                    tuple(ordered_curve.values())[1:],
                )
            ):
                raise ValueError("ASR curves must be non-decreasing")
            final_budget = max(ordered_curve)
            raw_values = (
                ordered_curve[final_budget],
                metrics.get("asr_query_auc"),
                metrics.get("normalized_action_entropy"),
            )
            if any(value is None or isinstance(value, bool) for value in raw_values):
                raise ValueError("ASR, AUC, and entropy metrics must be numeric")
            values = tuple(float(value) for value in raw_values)
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("ASR, AUC, and entropy metrics must be finite probabilities")
            eligible = metrics.get("eligible")
            successes = metrics.get("successes")
            query_budget = metrics.get("query_budget")
            max_calls = metrics.get("max_total_target_calls")
            eligible_digest = metrics.get("eligible_sample_ids_sha256")
            digest_before = metrics.get("policy_digest_before")
            digest_after = metrics.get("policy_digest_after")
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (eligible, successes, query_budget, max_calls)
            ):
                raise ValueError("counts and query budgets must be integers")
            if (
                eligible <= 0
                or not 0 <= successes <= eligible
                or query_budget < 1
                or not 1 <= max_calls <= query_budget
                or final_budget != query_budget
            ):
                raise ValueError("invalid eligibility, success, or query-budget accounting")
            if not math.isclose(values[0], successes / eligible, abs_tol=1e-12):
                raise ValueError("final ASR must equal successes divided by eligible samples")
            if not math.isclose(values[1], asr_query_auc(ordered_curve), abs_tol=1e-12):
                raise ValueError("ASR/query AUC must match the recorded curve")
            if metrics.get("frozen") is not True or digest_before != digest_after:
                raise ValueError("every evaluated method must remain frozen")
            if not isinstance(digest_before, str) or not digest_before:
                raise ValueError("frozen policy digests must be non-empty strings")
            if metrics.get("initialization_included") is not True:
                raise ValueError("the total query budget must include initialization")
            if not isinstance(eligible_digest, str) or not eligible_digest:
                raise ValueError("eligible sample identity digest is required")
            method_alignment = (
                eligible,
                eligible_digest,
                query_budget,
                tuple(ordered_curve),
            )
            if alignment is None:
                alignment = method_alignment
            elif method_alignment != alignment:
                raise ValueError("methods must share eligible samples and query-budget endpoints")
            bucket = family_metrics.setdefault(
                str(method),
                {"asr": {}, "auc": {}, "entropy": {}},
            )
            bucket["asr"][seed], bucket["auc"][seed], bucket["entropy"][seed] = values

    grid_complete = observed_pairs == expected_pairs
    missing_pairs = sorted(expected_pairs - observed_pairs)
    aggregate: dict[str, object] = {}
    fold_promotions: dict[str, bool] = {}
    fold_details: dict[str, object] = {}
    for family in families:
        methods = grouped.get(family)
        if methods is None:
            fold_promotions[family] = False
            fold_details[family] = {
                "passed": False,
                "reason": "no completed runs",
                "comparisons": {},
            }
            continue
        aggregate[family] = {
            method: {
                "final_asr": _mean_interval(tuple(values["asr"].values())),
                "asr_query_auc": _mean_interval(tuple(values["auc"].values())),
                "action_entropy": _mean_interval(tuple(values["entropy"].values())),
            }
            for method, values in methods.items()
        }
        learned = methods[LEARNED_METHOD]
        comparisons = {
            control_name: {
                "final_asr_delta": _paired_interval(
                    learned["asr"], methods[control_name]["asr"]
                ),
                "asr_query_auc_delta": _paired_interval(
                    learned["auc"], methods[control_name]["auc"]
                ),
            }
            for control_name in CONTROL_METHODS
        }
        family_grid_complete = set(learned["asr"]) == set(seeds)
        comparison_passed = all(
            comparison[metric]["ci95"] is not None
            and comparison[metric]["ci95"][0] > 0.0
            and comparison[metric]["mean"]
            >= (
                minimum_asr_gain
                if metric == "final_asr_delta"
                else minimum_auc_gain
            )
            for comparison in comparisons.values()
            for metric in ("final_asr_delta", "asr_query_auc_delta")
        )
        entropy_values = tuple(learned["entropy"].values())
        entropy_mean = statistics.fmean(entropy_values)
        entropy_passed = all(
            entropy_bounds[0] <= value <= entropy_bounds[1]
            for value in entropy_values
        )
        fold_passed = bool(
            family_grid_complete
            and len(seeds) >= minimum_seeds
            and comparison_passed
            and entropy_passed
        )
        fold_promotions[family] = fold_passed
        fold_details[family] = {
            "passed": fold_passed,
            "grid_complete": family_grid_complete,
            "seeds": list(sorted(learned["asr"])),
            "rl_action_entropy_mean": entropy_mean,
            "rl_action_entropy_by_seed": {
                str(seed): learned["entropy"][seed]
                for seed in sorted(learned["entropy"])
            },
            "entropy_passed": entropy_passed,
            "comparison_passed": comparison_passed,
            "comparisons": comparisons,
        }
    victim_gates_passed = grid_complete and all(victim_gates.values())
    return {
        "aggregate": aggregate,
        "promotion_gate": {
            "passed": bool(
                grid_complete
                and victim_gates_passed
                and fold_promotions
                and all(fold_promotions.values())
            ),
            "grid_complete": grid_complete,
            "missing_runs": [f"{family}/seed-{seed}" for family, seed in missing_pairs],
            "victim_gates_passed": victim_gates_passed,
            "minimum_seeds": minimum_seeds,
            "minimum_asr_gain": minimum_asr_gain,
            "minimum_auc_gain": minimum_auc_gain,
            "entropy_bounds": list(entropy_bounds),
            "folds": fold_promotions,
            "fold_details": fold_details,
            "requirements": [
                f"the exact configured family-by-seed grid is complete with at least {minimum_seeds} unique seeds per fold",
                f"paired Student-t 95% lower bounds for stochastic RL minus random, bandit, and score-greedy are positive, with mean gains of at least {minimum_asr_gain:.3f} final ASR and {minimum_auc_gain:.3f} ASR/query AUC",
                f"normalized action entropy for every seed remains between {entropy_bounds[0]:.2f} and {entropy_bounds[1]:.2f}",
                "every victim-quality gate passes",
            ],
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def run_study_from_datasets(
    config: CIFARStudyConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    dataset_version: str,
    resume: bool = True,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    report = progress or (lambda _message: None)
    started = time.monotonic()
    study_code_digest = _code_digest()
    base = MacPilotConfig.from_json(Path(config.base_config))
    output_dir = Path(config.output_dir) / config.name
    runs: list[dict[str, object]] = []
    for target_family in config.target_families:
        for seed in config.seeds:
            if _code_digest() != study_code_digest:
                raise RuntimeError("package code changed during the study; discard and rerun")
            report(f"starting target={target_family} seed={seed}")
            derived = replace(
                base,
                name=f"{config.name}-{target_family}-seed-{seed}",
                seed=seed,
                target_family=target_family,
                output_dir=str(output_dir / "runs"),
                device=device or base.device,
            )
            run = run_cifar_pilot_from_datasets(
                derived,
                train_dataset,
                test_dataset,
                resume=resume,
                dataset_version=dataset_version,
                progress=lambda message, family=target_family, run_seed=seed: report(
                    f"[{family}/seed-{run_seed}] {message}"
                ),
            )
            if _code_digest() != study_code_digest:
                raise RuntimeError("package code changed during the study; discard and rerun")
            runs.append(run)
            partial = {
                "schema_version": 1,
                "name": config.name,
                "status": "running",
                "research_valid": False,
                "study_code_digest": study_code_digest,
                "config": asdict(config),
                "runs": runs,
                **summarize_study(
                    runs,
                    expected_families=config.target_families,
                    expected_seeds=config.seeds,
                    minimum_seeds=config.minimum_seeds,
                    minimum_asr_gain=config.minimum_asr_gain,
                    minimum_auc_gain=config.minimum_auc_gain,
                    entropy_bounds=(config.entropy_min, config.entropy_max),
                ),
            }
            _write_json(output_dir / "study_manifest.json", partial)
    summary = summarize_study(
        runs,
        expected_families=config.target_families,
        expected_seeds=config.seeds,
        minimum_seeds=config.minimum_seeds,
        minimum_asr_gain=config.minimum_asr_gain,
        minimum_auc_gain=config.minimum_auc_gain,
        entropy_bounds=(config.entropy_min, config.entropy_max),
    )
    if not summary["promotion_gate"]["grid_complete"]:
        raise RuntimeError("completed study is missing configured family/seed runs")
    manifest = {
        "schema_version": 1,
        "name": config.name,
        "status": "complete",
        "research_valid": False,
        "study_code_digest": study_code_digest,
        "config": asdict(config),
        "elapsed_seconds": time.monotonic() - started,
        "runs": runs,
        **summary,
    }
    _write_json(output_dir / "study_manifest.json", manifest)
    return manifest


def run_cifar_study(
    config_path: Path,
    resume: bool = True,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError("install the vision extra before running the CIFAR study") from error
    config = CIFARStudyConfig.from_json(config_path)
    base = MacPilotConfig.from_json(Path(config.base_config))
    train_dataset = torchvision.datasets.CIFAR10(
        root=base.data_root,
        train=True,
        transform=ToTensor(),
        download=base.download,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=base.data_root,
        train=False,
        transform=ToTensor(),
        download=base.download,
    )
    return run_study_from_datasets(
        config,
        train_dataset,
        test_dataset,
        dataset_version=f"torchvision-{torchvision.__version__}",
        resume=resume,
        device=device,
        progress=progress,
    )
