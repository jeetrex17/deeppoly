"""Validated metric extraction for the confirmatory transfer gate."""

from __future__ import annotations


def final_asr(metrics: object, *, primary_budget: int = 50) -> float:
    """Return the ASR at the preregistered primary query budget."""

    if not isinstance(metrics, dict):
        raise ValueError("method metrics must be a mapping")
    curve = metrics.get("asr_at_budgets")
    if not isinstance(curve, dict) or not curve:
        raise ValueError("ASR curve is required")
    normalized = {
        int(budget): float(value)
        for budget, value in curve.items()
    }
    if primary_budget not in normalized:
        raise ValueError(
            f"the primary endpoint requires ASR at {primary_budget} calls"
        )
    return normalized[primary_budget]


def victim_macro_metrics(
    metrics: dict[str, object],
    *,
    primary_budget: int = 50,
    expected_victim_count: int | None = None,
) -> tuple[float, float, tuple[tuple[str, int, str], ...]]:
    """Average victims equally and retain cohort alignment evidence."""

    by_victim = metrics.get("by_victim")
    if not isinstance(by_victim, dict) or not by_victim:
        raise ValueError("per-victim metrics are required")
    if (
        expected_victim_count is not None
        and len(by_victim) != expected_victim_count
    ):
        raise ValueError(
            "per-victim metrics do not match the locked victim count"
        )
    asr_values: list[float] = []
    auc_values: list[float] = []
    alignment: list[tuple[str, int, str]] = []
    for victim_id, raw in sorted(by_victim.items()):
        if not isinstance(raw, dict):
            raise ValueError("per-victim metrics must be mappings")
        asr_values.append(
            final_asr(raw, primary_budget=primary_budget)
        )
        auc_values.append(float(raw["asr_query_auc"]))
        eligible = raw.get("eligible")
        digest = raw.get("eligible_sample_ids_sha256")
        if (
            not isinstance(eligible, int)
            or isinstance(eligible, bool)
            or eligible <= 0
            or not isinstance(digest, str)
            or not digest
        ):
            raise ValueError("per-victim eligible cohorts are invalid")
        alignment.append((str(victim_id), eligible, digest))
    return (
        sum(asr_values) / len(asr_values),
        sum(auc_values) / len(auc_values),
        tuple(alignment),
    )
