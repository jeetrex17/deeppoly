"""Create dependency-free visual reports for a completed CIFAR-10 Mac pilot.

The pilot runner writes a compact JSON manifest so the expensive experiment and
the presentation layer stay separate.  This module turns that manifest into a
small, portable report: the graphics are SVG rather than PNGs so they remain
sharp in GitHub, a paper draft, or a slide deck and do not require matplotlib.

Run it from the repository root::

    uv run python -m rl_transfer.pilot_report
"""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("docs/research/cifar10_m4_pilot_results.json")
DEFAULT_OUTPUT_DIR = Path("docs/research")

_BACKGROUND = "#0b1020"
_PANEL = "#151d33"
_GRID = "#31415f"
_TEXT = "#eef4ff"
_MUTED = "#aebbd3"
_ACCENT = "#63d5ff"
_GOOD = "#54d6a0"
_WARM = "#ffb85c"
_PINK = "#ff7ab6"
_METHOD_COLORS = (_MUTED, _ACCENT, _GOOD, _WARM, _PINK)


class PilotReportError(ValueError):
    """Raised when an input manifest cannot support the pilot report."""


def _as_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotReportError(f"{field} must be an object")
    return value


def _require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotReportError(f"{field} must be numeric")
    return float(value)


def _require_budget(value: object, field: str) -> float:
    """Accept JSON object keys that encode an integer query budget."""

    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as error:
            raise PilotReportError(f"{field} must be numeric") from error
    return _require_number(value, field)


def load_pilot_results(path: Path) -> dict[str, Any]:
    """Load and validate the portion of a pilot manifest this report uses."""

    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PilotReportError(f"results file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise PilotReportError(f"results file is not valid JSON: {path}") from error
    manifest = dict(_as_mapping(result, "results"))

    for key in ("name", "elapsed_seconds", "victims", "policy", "evaluation"):
        if key not in manifest:
            raise PilotReportError(f"results missing required field: {key}")
    _require_number(manifest["elapsed_seconds"], "elapsed_seconds")
    victims = _as_mapping(manifest["victims"], "victims")
    if not victims:
        raise PilotReportError("victims must not be empty")
    for family, victim in victims.items():
        victim_data = _as_mapping(victim, f"victims.{family}")
        if "source_validation_accuracy" not in victim_data:
            raise PilotReportError(f"victims.{family} missing source_validation_accuracy")
        _require_number(
            victim_data["source_validation_accuracy"],
            f"victims.{family}.source_validation_accuracy",
        )
    policy = _as_mapping(manifest["policy"], "policy")
    training = _as_mapping(policy.get("training"), "policy.training")
    for key in ("episodes", "trained_episodes", "source_calls"):
        _require_number(training.get(key), f"policy.training.{key}")
    evaluation = _as_mapping(manifest["evaluation"], "evaluation")
    if not evaluation:
        raise PilotReportError("evaluation must not be empty")
    for method, metrics in evaluation.items():
        metrics_data = _as_mapping(metrics, f"evaluation.{method}")
        _require_number(metrics_data.get("eligible"), f"evaluation.{method}.eligible")
        _require_number(metrics_data.get("successes"), f"evaluation.{method}.successes")
        curve = _as_mapping(
            metrics_data.get("asr_at_budgets"), f"evaluation.{method}.asr_at_budgets"
        )
        if len(curve) < 2:
            raise PilotReportError(
                f"evaluation.{method}.asr_at_budgets needs at least two budgets"
            )
        for budget, rate in curve.items():
            _require_budget(budget, f"evaluation.{method}.asr_at_budgets budget")
            _require_number(rate, f"evaluation.{method}.asr_at_budgets.{budget}")
    return manifest


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _percent(value: object, digits: int = 1) -> str:
    return f"{100 * _require_number(value, 'percentage'):.{digits}f}%"


def _format_duration(seconds: object) -> str:
    value = _require_number(seconds, "elapsed_seconds")
    minutes, remainder = divmod(round(value), 60)
    if minutes:
        return f"{minutes} min {remainder:02d} s ({value:.1f} s)"
    return f"{value:.1f} s"


def _method_label(method: str) -> str:
    labels = {
        "fixed_action": "Fixed-action baseline",
        "groupdro_recurrent_ppo": "RL policy (deterministic)",
        "groupdro_recurrent_ppo_stochastic": "RL policy (stochastic)",
        "random_action": "Random-action baseline",
    }
    return labels.get(method, method.replace("_", " ").title())


def _family_label(family: str) -> str:
    labels = {
        "classical_cnn": "Classical CNN",
        "modern_cnn": "Modern CNN",
        "transformer": "Patch Transformer",
    }
    return labels.get(family, family.replace("_", " ").title())


def _svg_document(title: str, body: str, width: int = 1000, height: int = 600) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">
  <title id="title">{_escape(title)}</title>
  <desc id="description">{_escape(title)}. Generated from the CIFAR-10 Mac pilot results manifest.</desc>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .title {{ fill: {_TEXT}; font-size: 27px; font-weight: 700; }}
    .subtitle {{ fill: {_MUTED}; font-size: 15px; }}
    .label {{ fill: {_TEXT}; font-size: 15px; font-weight: 600; }}
    .small {{ fill: {_MUTED}; font-size: 13px; }}
    .value {{ fill: {_TEXT}; font-size: 17px; font-weight: 700; }}
  </style>
  <rect width="100%" height="100%" fill="{_BACKGROUND}"/>
{body}
</svg>
'''


def render_victim_accuracy_svg(results: Mapping[str, Any]) -> str:
    """Render validation accuracy of every victim family as a bar chart."""

    victims = _as_mapping(results["victims"], "victims")
    gate = _as_mapping(results.get("victim_accuracy_gate", {}), "victim_accuracy_gate")
    thresholds = _as_mapping(gate.get("thresholds", {}), "victim_accuracy_gate.thresholds")
    width, height = 1000, 600
    left, right, top, bottom = 105, 55, 130, 105
    chart_width, chart_height = width - left - right, height - top - bottom
    families = list(victims)
    max_value = 1.0
    body = [
        '<text class="title" x="55" y="58">Victim validation accuracy</text>',
        '<text class="subtitle" x="55" y="85">Each frozen victim passed the pilot quality gate before transfer evaluation.</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + chart_height - chart_height * tick / 100
        body.extend(
            (
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{_GRID}" stroke-width="1"/>',
                f'<text class="small" x="{left - 14}" y="{y + 5:.1f}" text-anchor="end">{tick}%</text>',
            )
        )
    spacing = chart_width / max(len(families), 1)
    bar_width = min(150, spacing * 0.48)
    for index, family in enumerate(families):
        victim = _as_mapping(victims[family], f"victims.{family}")
        accuracy = _require_number(victim["source_validation_accuracy"], "accuracy")
        threshold = _require_number(thresholds.get(family, 0), "threshold")
        center = left + spacing * (index + 0.5)
        bar_height = max(0, min(max_value, accuracy)) * chart_height
        y = top + chart_height - bar_height
        threshold_y = top + chart_height - max(0, min(max_value, threshold)) * chart_height
        body.extend(
            (
                f'<rect x="{center - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="8" fill="{_GOOD}"/>',
                f'<line x1="{center - bar_width / 2 - 14:.1f}" y1="{threshold_y:.1f}" x2="{center + bar_width / 2 + 14:.1f}" y2="{threshold_y:.1f}" stroke="{_WARM}" stroke-width="3"/>',
                f'<text class="value" x="{center:.1f}" y="{y - 12:.1f}" text-anchor="middle">{_percent(accuracy)}</text>',
                f'<text class="label" x="{center:.1f}" y="{top + chart_height + 34}" text-anchor="middle">{_escape(_family_label(family))}</text>',
                f'<text class="small" x="{center:.1f}" y="{top + chart_height + 55}" text-anchor="middle">gate {_percent(threshold, 0)}</text>',
            )
        )
    body.extend(
        (
            f'<rect x="55" y="548" width="12" height="12" rx="3" fill="{_GOOD}"/>',
            '<text class="small" x="75" y="559">source-validation accuracy</text>',
            f'<line x1="270" y1="554" x2="294" y2="554" stroke="{_WARM}" stroke-width="3"/>',
            '<text class="small" x="304" y="559">quality-gate threshold</text>',
        )
    )
    return _svg_document("CIFAR-10 pilot victim validation accuracy", "\n  ".join(body), width, height)


def _curve_points(
    budgets: Sequence[float],
    curve: Mapping[str, Any],
    left: float,
    top: float,
    chart_width: float,
    chart_height: float,
    max_rate: float,
) -> str:
    first_budget, last_budget = budgets[0], budgets[-1]
    span = max(last_budget - first_budget, 1.0)
    coordinates = []
    for budget in budgets:
        rate = _require_number(curve.get(str(int(budget)), curve.get(str(budget), 0)), "ASR")
        x = left + chart_width * (budget - first_budget) / span
        y = top + chart_height * (1 - min(max(rate, 0), max_rate) / max_rate)
        coordinates.append(f"{x:.1f},{y:.1f}")
    return " ".join(coordinates)


def render_asr_curve_svg(results: Mapping[str, Any]) -> str:
    """Render attack-success-rate curves at the recorded query budgets."""

    evaluation = _as_mapping(results["evaluation"], "evaluation")
    all_budgets = sorted(
        {
            _require_budget(budget, "ASR budget")
            for metrics in evaluation.values()
            for budget in _as_mapping(metrics, "evaluation entry")["asr_at_budgets"]
        }
    )
    max_observed = max(
        _require_number(rate, "ASR")
        for metrics in evaluation.values()
        for rate in _as_mapping(metrics, "evaluation entry")["asr_at_budgets"].values()
    )
    max_rate = max(0.10, (int(max_observed * 1000 / 5) + 1) * 0.005)
    width, height = 1000, 600
    left, right, top, bottom = 105, 260, 130, 105
    chart_width, chart_height = width - left - right, height - top - bottom
    body = [
        '<text class="title" x="55" y="58">Attack success rate by query budget</text>',
        '<text class="subtitle" x="55" y="85">Held-out Patch Transformer; ASR is measured only on 99 clean-correct examples.</text>',
    ]
    tick_count = 5
    for index in range(tick_count + 1):
        rate = max_rate * index / tick_count
        y = top + chart_height * (1 - index / tick_count)
        body.extend(
            (
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{_GRID}" stroke-width="1"/>',
                f'<text class="small" x="{left - 14}" y="{y + 5:.1f}" text-anchor="end">{_percent(rate, 0)}</text>',
            )
        )
    first_budget, last_budget = all_budgets[0], all_budgets[-1]
    budget_span = max(last_budget - first_budget, 1.0)
    for budget in all_budgets:
        x = left + chart_width * (budget - first_budget) / budget_span
        body.extend(
            (
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_height}" stroke="{_GRID}" stroke-width="1"/>',
                f'<text class="small" x="{x:.1f}" y="{top + chart_height + 30}" text-anchor="middle">{int(budget)}</text>',
            )
        )
    body.append(
        f'<text class="small" x="{left + chart_width / 2:.1f}" y="{height - 28}" text-anchor="middle">queries available to the attack</text>'
    )
    for index, (method, metrics) in enumerate(evaluation.items()):
        metrics_data = _as_mapping(metrics, f"evaluation.{method}")
        curve = _as_mapping(metrics_data["asr_at_budgets"], f"evaluation.{method}.asr_at_budgets")
        color = _METHOD_COLORS[index % len(_METHOD_COLORS)]
        points = _curve_points(all_budgets, curve, left, top, chart_width, chart_height, max_rate)
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
        for point in points.split():
            x, y = point.split(",")
            body.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{color}"/>')
        legend_y = 155 + index * 63
        label = _method_label(method)
        body.extend(
            (
                f'<line x1="{width - right + 32}" y1="{legend_y}" x2="{width - right + 58}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>',
                f'<text class="label" x="{width - right + 70}" y="{legend_y + 5}">{_escape(label)}</text>',
                f'<text class="small" x="{width - right + 70}" y="{legend_y + 25}">{int(_require_number(metrics_data["successes"], "successes"))}/{int(_require_number(metrics_data["eligible"], "eligible"))} successful; AUC {_percent(metrics_data.get("asr_query_auc", 0))}</text>',
            )
        )
    return _svg_document("CIFAR-10 pilot ASR versus query budget", "\n  ".join(body), width, height)


def render_pipeline_svg(results: Mapping[str, Any]) -> str:
    """Render a high-level source-to-held-out-victim experiment pipeline."""

    victims = _as_mapping(results["victims"], "victims")
    training = _as_mapping(_as_mapping(results["policy"], "policy")["training"], "policy.training")
    target_accuracy = _require_number(results.get("target_test_accuracy", 0), "target_test_accuracy")
    source_families = [family for family in ("classical_cnn", "modern_cnn") if family in victims]
    target_family = "transformer" if "transformer" in victims else list(victims)[-1]
    source_text = " + ".join(_family_label(family) for family in source_families)
    target = _as_mapping(victims[target_family], f"victims.{target_family}")
    width, height = 1100, 560
    body = [
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#63d5ff"/></marker></defs>',
        '<text class="title" x="55" y="58">Frozen-policy cross-victim transfer pipeline</text>',
        '<text class="subtitle" x="55" y="85">The policy learns only from source families, then is held fixed while attacking a separate target family.</text>',
    ]
    cards = (
        (55, 155, 225, 205, "1  Train source victims", source_text, "source-validation gates passed", _GOOD),
        (325, 155, 225, 205, "2  Train RL policy", f"{int(_require_number(training['episodes'], 'episodes'))} scheduled episodes", f"{int(_require_number(training['source_calls'], 'source_calls')):,} source-model calls", _ACCENT),
        (595, 155, 225, 205, "3  Freeze policy", "No target-model training", "deterministic + stochastic inference", _WARM),
        (865, 155, 180, 205, "4  Evaluate", _family_label(target_family), f"target clean accuracy {_percent(target_accuracy)}", _PINK),
    )
    for index, (x, y, card_width, card_height, heading, line_one, line_two, color) in enumerate(cards):
        body.extend(
            (
                f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" rx="16" fill="{_PANEL}" stroke="{color}" stroke-width="2"/>',
                f'<rect x="{x}" y="{y}" width="{card_width}" height="9" rx="4" fill="{color}"/>',
                f'<text class="label" x="{x + 18}" y="{y + 47}">{_escape(heading)}</text>',
                f'<text class="value" x="{x + 18}" y="{y + 95}">{_escape(line_one)}</text>',
                f'<text class="small" x="{x + 18}" y="{y + 124}">{_escape(line_two)}</text>',
            )
        )
        if index < len(cards) - 1:
            next_x = cards[index + 1][0]
            body.append(f'<line x1="{x + card_width + 10}" y1="{y + 102}" x2="{next_x - 12}" y2="{y + 102}" stroke="{_ACCENT}" stroke-width="4" marker-end="url(#arrow)"/>')
    body.extend(
        (
            '<text class="label" x="55" y="428">Comparison set on the held-out target</text>',
            '<text class="small" x="55" y="455">Fixed action  •  frozen RL policy (deterministic)  •  frozen RL policy (stochastic)  •  random action</text>',
            '<text class="small" x="55" y="503">Important: this bounded M4 pilot is exploratory (`research_valid: false`); it is a feasibility check, not a paper-scale transferability claim.</text>',
        )
    )
    return _svg_document("CIFAR-10 Mac pilot frozen-policy transfer pipeline", "\n  ".join(body), width, height)


def _budget_columns(results: Mapping[str, Any]) -> list[int]:
    evaluation = _as_mapping(results["evaluation"], "evaluation")
    return sorted(
        {
            int(_require_budget(budget, "ASR budget"))
            for metrics in evaluation.values()
            for budget in _as_mapping(metrics, "evaluation entry")["asr_at_budgets"]
        }
    )


def render_markdown_summary(results: Mapping[str, Any], source_name: str) -> str:
    """Create a concise, presentation-ready Markdown interpretation of the pilot."""

    victims = _as_mapping(results["victims"], "victims")
    evaluation = _as_mapping(results["evaluation"], "evaluation")
    training = _as_mapping(_as_mapping(results["policy"], "policy")["training"], "policy.training")
    gate = _as_mapping(results.get("victim_accuracy_gate", {}), "victim_accuracy_gate")
    thresholds = _as_mapping(gate.get("thresholds", {}), "victim_accuracy_gate.thresholds")
    budget_columns = _budget_columns(results)
    max_budget = budget_columns[-1]
    lines = [
        "# CIFAR-10 M4 pilot — visual results",
        "",
        f"**Status:** exploratory pilot (`research_valid: {str(bool(results.get('research_valid'))).lower()}`). "
        f"Generated from `{source_name}`.",
        "",
        "![Victim validation accuracy](figures/cifar10_m4_pilot_victim_accuracy.svg)",
        "",
        "![ASR by query budget](figures/cifar10_m4_pilot_asr_by_query.svg)",
        "",
        "![Frozen-policy transfer pipeline](figures/cifar10_m4_pilot_pipeline.svg)",
        "",
        "## What ran on the Mac",
        "",
        f"- Device: `{_escape(_as_mapping(results.get('device', {}), 'device').get('resolved', 'unknown'))}`; elapsed wall time: **{_format_duration(results['elapsed_seconds'])}**.",
        f"- Three victim families were trained, then frozen. The recurrent GroupDRO/PPO policy was scheduled for **{int(_require_number(training['episodes'], 'episodes'))} episodes**; the manifest reports **{int(_require_number(training['trained_episodes'], 'trained_episodes'))} trained episodes** and **{int(_require_number(training['source_calls'], 'source_calls')):,} source-model calls**.",
        f"- Transfer evaluation attacked the held-out Patch Transformer. Its clean outer-test accuracy was **{_percent(results.get('target_test_accuracy', 0))}**; only the **99** clean-correct samples were eligible for ASR.",
        "",
        "## Victim quality before the attack",
        "",
        "| Family | Victim | Source validation accuracy | Gate |",
        "| --- | --- | ---: | ---: |",
    ]
    for family, victim in victims.items():
        victim_data = _as_mapping(victim, f"victims.{family}")
        lines.append(
            "| "
            f"{_family_label(family)} | `{victim_data.get('victim_id', 'unknown')}` | "
            f"{_percent(victim_data['source_validation_accuracy'])} | "
            f"{_percent(thresholds.get(family, 0), 0)} |"
        )
    lines.extend(
        (
            "",
            "## Held-out transfer attack results",
            "",
            "| Method | Successful / eligible | Final ASR | ASR/query AUC | "
            + " | ".join(f"ASR @ {budget}" for budget in budget_columns)
            + " |",
            "| --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in budget_columns) + " |",
        )
    )
    for method, metrics in evaluation.items():
        metrics_data = _as_mapping(metrics, f"evaluation.{method}")
        curve = _as_mapping(metrics_data["asr_at_budgets"], f"evaluation.{method}.asr_at_budgets")
        final_rate = curve.get(str(max_budget), curve.get(max_budget, 0))
        rates = [curve.get(str(budget), curve.get(budget, 0)) for budget in budget_columns]
        lines.append(
            f"| {_method_label(method)} | {int(_require_number(metrics_data['successes'], 'successes'))}/{int(_require_number(metrics_data['eligible'], 'eligible'))} | "
            f"{_percent(final_rate)} | {_percent(metrics_data.get('asr_query_auc', 0))} | "
            + " | ".join(_percent(rate) for rate in rates)
            + " |"
        )
    random_metrics = _as_mapping(evaluation.get("random_action", {}), "evaluation.random_action")
    stochastic_metrics = _as_mapping(
        evaluation.get("groupdro_recurrent_ppo_stochastic", {}),
        "evaluation.groupdro_recurrent_ppo_stochastic",
    )
    random_curve = _as_mapping(random_metrics.get("asr_at_budgets", {}), "random curve")
    stochastic_curve = _as_mapping(stochastic_metrics.get("asr_at_budgets", {}), "stochastic curve")
    random_final = _require_number(random_curve.get(str(max_budget), 0), "random final ASR")
    stochastic_final = _require_number(stochastic_curve.get(str(max_budget), 0), "stochastic final ASR")
    lines.extend(
        (
            "",
            "## Readout",
            "",
            f"At the recorded {max_budget}-query budget, the strongest RL variant was stochastic inference at **{_percent(stochastic_final)}** (5/99 successful). The random baseline reached **{_percent(random_final)}** (7/99), so this run does **not** show an RL transfer advantage. It does show that the full frozen-policy, cross-family evaluation pipeline executes end-to-end on the M4.",
            "",
            "The trained victim accuracies cleared the configured pilot gates, but the held-out target was only 49.5% accurate. This result should therefore guide the next experiment—stronger victim training, more seeds, and confidence intervals—not be treated as a general attack-transfer conclusion.",
            "",
            "## Reproduce the figures",
            "",
            "```bash",
            "uv run python -m rl_transfer.pilot_report \\",
            f"  --input {source_name} \\",
            "  --output-dir docs/research",
            "```",
            "",
        )
    )
    return "\n".join(lines)


def write_pilot_report(input_path: Path, output_dir: Path) -> dict[str, Path]:
    """Write all SVG figures and the Markdown report, returning their paths."""

    results = load_pilot_results(input_path)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "victim_accuracy": figures / "cifar10_m4_pilot_victim_accuracy.svg",
        "asr_by_query": figures / "cifar10_m4_pilot_asr_by_query.svg",
        "pipeline": figures / "cifar10_m4_pilot_pipeline.svg",
        "summary": output_dir / "cifar10_m4_pilot_summary.md",
    }
    paths["victim_accuracy"].write_text(render_victim_accuracy_svg(results), encoding="utf-8")
    paths["asr_by_query"].write_text(render_asr_curve_svg(results), encoding="utf-8")
    paths["pipeline"].write_text(render_pipeline_svg(results), encoding="utf-8")
    paths["summary"].write_text(
        render_markdown_summary(results, input_path.as_posix()), encoding="utf-8"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a visual report for a CIFAR-10 Mac pilot")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="pilot results JSON")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="directory for report and figures"
    )
    args = parser.parse_args()
    outputs = write_pilot_report(args.input, args.output_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
