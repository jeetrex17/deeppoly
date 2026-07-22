"""Compact and visualize a completed multi-fold CIFAR study manifest."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
from typing import Mapping

from .cifar_study import summarize_study


METHODS = (
    ("groupdro_recurrent_ppo_stochastic", "RL stochastic", "#54d6a0"),
    ("random_action", "Random", "#ffb85c"),
    ("bandit_action", "Score bandit", "#63d5ff"),
    ("score_greedy", "Score greedy", "#c790ff"),
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def compact_study_manifest(study: Mapping[str, object]) -> dict[str, object]:
    if study.get("status") != "complete" or not study.get("runs"):
        raise ValueError("a completed study with runs is required")
    config = study.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("the study manifest requires a configuration mapping")
    summary = summarize_study(
        study["runs"],
        expected_families=config.get("target_families"),
        expected_seeds=config.get("seeds"),
        minimum_seeds=int(config.get("minimum_seeds", 3)),
        minimum_asr_gain=float(config.get("minimum_asr_gain", 0.01)),
        minimum_auc_gain=float(config.get("minimum_auc_gain", 0.005)),
        entropy_bounds=(
            float(config.get("entropy_min", 0.10)),
            float(config.get("entropy_max", 0.95)),
        ),
    )
    source_summary_verified = (
        study.get("aggregate") == summary["aggregate"]
        and study.get("promotion_gate") == summary["promotion_gate"]
    )
    compact_runs = []
    for run in study["runs"]:
        evaluations = {
            method: {
                key: metrics[key]
                for key in (
                    "eligible",
                    "successes",
                    "asr_at_budgets",
                    "asr_query_auc",
                    "normalized_action_entropy",
                    "query_budget",
                    "max_total_target_calls",
                    "initialization_included",
                    "eligible_sample_ids_sha256",
                    "policy_digest_before",
                    "policy_digest_after",
                    "frozen",
                )
            }
            for method, metrics in run["evaluation"].items()
        }
        compact_runs.append(
            {
                "seed": run["seed"],
                "target_family": run["target_family"],
                "source_families": run["source_families"],
                "elapsed_seconds": run["elapsed_seconds"],
                "target_test_accuracy": run["target_test_accuracy"],
                "victim_accuracy_gate": run["victim_accuracy_gate"],
                "victim_instances": {
                    family: [
                        {
                            key: instance.get(key)
                            for key in (
                                "victim_id",
                                "instance_index",
                                "training_seed",
                                "checkpoint_sha256",
                                "source_validation_accuracy",
                                "resumed",
                            )
                        }
                        for instance in instances
                    ]
                    for family, instances in run["victim_instances"].items()
                },
                "provenance": {
                    key: run.get(key)
                    for key in (
                        "fingerprint",
                        "config_digest",
                        "split_digest",
                        "victim_cache_digest",
                        "victim_code_digest",
                    )
                },
                "runtime": {
                    key: run.get("runtime", {}).get(key)
                    for key in ("git_revision", "code_digest", "torch", "platform", "determinism")
                },
                "policy_training": {
                    key: run["policy"]["training"].get(key)
                    for key in (
                        "episodes",
                        "trained_episodes",
                        "policy_sample_pool_size",
                        "unique_policy_samples_visited",
                        "source_calls",
                        "source_calls_by_family",
                        "source_calls_by_victim",
                        "final_family_weights",
                    )
                },
                "policy_training_blocks": [
                    {
                        key: block.get(key)
                        for key in (
                            "episode_offset",
                            "episodes",
                            "trained_episodes",
                            "unique_sample_count",
                            "family_weights",
                            "family_diagnostics",
                            "instance_diagnostics",
                            "ppo",
                        )
                    }
                    for block in run["policy"]["training"].get("blocks", [])
                ],
                "policy_checkpoint_sha256": run["policy"].get("checkpoint_sha256"),
                "evaluation": evaluations,
            }
        )
    return {
        "schema_version": 1,
        "name": study["name"],
        "status": study["status"],
        "research_valid": study["research_valid"],
        "elapsed_seconds": study["elapsed_seconds"],
        "study_code_digest": study.get("study_code_digest"),
        "config": config,
        "source_summary_verified": source_summary_verified,
        "promotion_gate": summary["promotion_gate"],
        "aggregate": summary["aggregate"],
        "runs": compact_runs,
    }


def _safe_stem(name: object) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._-")
    if not stem:
        raise ValueError("study name must contain a safe filename character")
    return stem


def render_asr_svg(compact: Mapping[str, object]) -> str:
    families = list(compact["aggregate"])
    seed_count = len({int(run["seed"]) for run in compact["runs"]})
    run_count = len(compact["runs"])
    width, height = 1000, 590
    left, right, top, bottom = 95, 55, 125, 120
    chart_width, chart_height = width - left - right, height - top - bottom
    max_asr = max(
        float(compact["aggregate"][family][method]["final_asr"]["mean"])
        for family in families
        for method, _, _ in METHODS
    )
    ceiling = max(0.05, (int(max_asr * 100 / 5) + 1) * 0.05)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.title{fill:#eef4ff;font-size:27px;font-weight:700}.subtitle{fill:#aebbd3;font-size:15px}.label{fill:#eef4ff;font-size:14px}.small{fill:#aebbd3;font-size:12px}</style>',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<text class="title" x="55" y="55">CIFAR-10 frozen transfer ASR by target family</text>',
        f'<text class="subtitle" x="55" y="83">{seed_count} seed(s); {run_count} family/seed runs; mean held-out-target ASR.</text>',
    ]
    for tick in range(6):
        value = ceiling * tick / 5
        y = top + chart_height * (1 - tick / 5)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#31415f"/>')
        parts.append(f'<text class="small" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{100*value:.0f}%</text>')
    group_width = chart_width / len(families)
    bar_width = min(48, group_width / (len(METHODS) + 1))
    for family_index, family in enumerate(families):
        center = left + group_width * (family_index + 0.5)
        for method_index, (method, _, color) in enumerate(METHODS):
            value = float(compact["aggregate"][family][method]["final_asr"]["mean"])
            bar_height = chart_height * value / ceiling
            offset = method_index - (len(METHODS) - 1) / 2
            x = center + offset * bar_width - bar_width * 0.42
            y = top + chart_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width*0.84:.1f}" height="{bar_height:.1f}" rx="5" fill="{color}"/>')
            parts.append(f'<text class="small" x="{x+bar_width*0.42:.1f}" y="{y-8:.1f}" text-anchor="middle">{100*value:.1f}%</text>')
        parts.append(f'<text class="label" x="{center:.1f}" y="{top+chart_height+35}" text-anchor="middle">{html.escape(family.replace("_", " ").title())}</text>')
    legend_x = 75
    for _, label, color in METHODS:
        parts.append(f'<rect x="{legend_x}" y="535" width="14" height="14" rx="3" fill="{color}"/>')
        parts.append(f'<text class="small" x="{legend_x+22}" y="547">{label}</text>')
        legend_x += 220
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def render_markdown(compact: Mapping[str, object], figure_name: str) -> str:
    seed_count = len({int(run["seed"]) for run in compact["runs"]})
    family_count = len({str(run["target_family"]) for run in compact["runs"]})
    promotion = compact["promotion_gate"]
    lines = [
        "# CIFAR-10 cross-victim RL study",
        "",
        f"Status: exploratory (`research_valid: {str(compact['research_valid']).lower()}`). "
        f"Runtime: **{compact['elapsed_seconds'] / 60:.1f} minutes**.",
        "",
        f"![Frozen ASR by held-out family](figures/{figure_name})",
        "",
        "| Held-out target | Seed | Time | Target clean acc. | Gate threshold | Eligible | RL stochastic | Random | Score bandit | Score greedy | Victim gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in compact["runs"]:
        evaluation = run["evaluation"]
        learned = evaluation["groupdro_recurrent_ppo_stochastic"]
        random = evaluation["random_action"]
        bandit = evaluation["bandit_action"]
        greedy = evaluation["score_greedy"]
        threshold = run["victim_accuracy_gate"].get("thresholds", {}).get(
            run["target_family"]
        )
        threshold_cell = f"{100*threshold:.1f}%" if threshold is not None else "n/a"

        def result_cell(metrics: Mapping[str, object]) -> str:
            curve = metrics["asr_at_budgets"]
            final_budget = max(curve, key=lambda value: int(value))
            return (
                f"{metrics['successes']}/{metrics['eligible']} "
                f"({100*curve[final_budget]:.1f}%)"
            )

        lines.append(
            f"| {run['target_family']} | {run['seed']} | "
            f"{run['elapsed_seconds'] / 60:.1f}m | "
            f"{100*run['target_test_accuracy']:.1f}% | {threshold_cell} | "
            f"{learned['eligible']} | "
            f"{result_cell(learned)} | {result_cell(random)} | "
            f"{result_cell(bandit)} | {result_cell(greedy)} | "
            f"{'pass' if run['victim_accuracy_gate']['passed'] else 'fail'} |"
        )
    interpretation = [
        f"The study completed {len(compact['runs'])} run(s) across {family_count} held-out "
        f"target family/families and {seed_count} unique seed(s).",
    ]
    if promotion.get("passed"):
        interpretation.append(
            "The configured exploratory promotion gate passed: the paired 95% lower "
            "bounds favor stochastic RL over random, score-bandit, and score-greedy controls, victim gates "
            "passed, and policy entropy stayed in range."
        )
    else:
        reasons = []
        if not promotion.get("grid_complete", True):
            reasons.append("the configured family/seed grid is incomplete")
        if seed_count < 3:
            reasons.append("fewer than three aligned seeds are available")
        if not promotion.get("victim_gates_passed", True):
            reasons.append("one or more victim-quality gates failed")
        if any(not passed for passed in promotion.get("folds", {}).values()):
            reasons.append("at least one fold failed the paired RL-versus-control criterion")
        detail = "; ".join(reasons) if reasons else "one or more prespecified checks failed"
        interpretation.append(f"The promotion gate did not pass because {detail}.")
    interpretation.append(
        "All reported target attacks use frozen deployment and a shared total query budget; "
        "the results remain exploratory and should not be treated as a publication claim."
    )
    lines.extend(("", "## Interpretation", "", " ".join(interpretation), ""))
    return "\n".join(lines)


def write_study_report(input_path: Path, output_dir: Path) -> dict[str, Path]:
    study = json.loads(input_path.read_text())
    compact = compact_study_manifest(study)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(compact["name"]).replace("-", "_")
    json_path = output_dir / f"{stem}_results.json"
    markdown_path = output_dir / f"{stem}_summary.md"
    figure_path = figures / f"{stem}_asr.svg"
    _atomic_json(json_path, compact)
    figure_path.write_text(render_asr_svg(compact))
    markdown_path.write_text(render_markdown(compact, figure_path.name))
    return {"json": json_path, "markdown": markdown_path, "figure": figure_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact CIFAR study report")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research"))
    args = parser.parse_args()
    for label, path in write_study_report(args.input, args.output_dir).items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
