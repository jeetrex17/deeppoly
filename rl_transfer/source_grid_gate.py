"""Global source-grid gate for the locked RTX study."""

from __future__ import annotations

from .gpu_config import RTXPublicationConfig


def source_grid_gate(
    runs: list[dict[str, object]],
    study: RTXPublicationConfig,
) -> dict[str, object]:
    expected = {
        (family, seed)
        for family in study.target_families
        for seed in study.seeds
    }
    observed = {
        (str(run.get("target_family")), int(run.get("seed", -1)))
        for run in runs
    }
    failures: list[str] = []
    bank_digests: set[str] = set()
    checkpoints: dict[str, str] = {}
    checkpoints_by_family: dict[str, set[str]] = {
        family: set() for family in study.target_families
    }
    persistent_by_family: dict[str, set[str]] = {
        family: set() for family in study.target_families
    }
    for run in runs:
        family = str(run.get("target_family"))
        seed = int(run.get("seed", -1))
        run_id = f"{family}/seed-{seed}"
        if run.get("status") != "source_complete":
            failures.append(f"{run_id}: source phase did not complete")
        if run.get("target_evaluation_performed") is not False:
            failures.append(
                f"{run_id}: target access occurred before the source gate"
            )
        if run.get("target_calls") != 0:
            failures.append(f"{run_id}: target call count is not zero")
        if run.get("victim_seed") != study.victim_seed:
            failures.append(f"{run_id}: fixed victim seed mismatch")
        bank_digest = run.get("victim_bank_digest")
        if not isinstance(bank_digest, str) or len(bank_digest) != 64:
            failures.append(f"{run_id}: victim bank digest is invalid")
        else:
            bank_digests.add(bank_digest)
        if run.get("validation_roles_disjoint") is not True:
            failures.append(f"{run_id}: validation roles overlap")
        victim_gate = run.get("victim_accuracy_gate")
        if (
            not isinstance(victim_gate, dict)
            or victim_gate.get("passed") is not True
        ):
            failures.append(f"{run_id}: victim accuracy gate failed")
        source_gate = run.get("source_competence_gate")
        if (
            not isinstance(source_gate, dict)
            or source_gate.get("passed") is not True
        ):
            failures.append(f"{run_id}: source competence gate failed")
        policy = run.get("policy")
        checkpoint = (
            policy.get("checkpoint_sha256")
            if isinstance(policy, dict)
            else None
        )
        if not isinstance(checkpoint, str) or len(checkpoint) != 64:
            failures.append(f"{run_id}: policy checkpoint digest is invalid")
        else:
            checkpoints[run_id] = checkpoint
            if family in checkpoints_by_family:
                checkpoints_by_family[family].add(checkpoint)
        persistent = (
            policy.get("persistent_digest")
            if isinstance(policy, dict)
            else None
        )
        if not isinstance(persistent, str) or not persistent:
            failures.append(f"{run_id}: policy persistent digest is invalid")
        elif family in persistent_by_family:
            persistent_by_family[family].add(persistent)
        training = (
            policy.get("training")
            if isinstance(policy, dict)
            else None
        )
        cloning = (
            training.get("behavior_cloning")
            if isinstance(training, dict)
            else None
        )
        if (
            not isinstance(cloning, dict)
            or cloning.get("enabled") is not True
            or not isinstance(cloning.get("gate"), dict)
            or cloning["gate"].get("passed") is not True
        ):
            failures.append(
                f"{run_id}: behavior-cloning representation gate failed"
            )
    failures.extend(
        f"{family}/seed-{seed}: missing source run"
        for family, seed in sorted(expected - observed)
    )
    if len(bank_digests) != 1:
        failures.append("source runs do not share one fixed victim bank")
    for family, digests in checkpoints_by_family.items():
        if len(digests) != len(study.seeds):
            failures.append(
                f"{family}: policy seed checkpoints are not distinct"
            )
    for family, digests in persistent_by_family.items():
        if len(digests) != len(study.seeds):
            failures.append(
                f"{family}: policy seeds did not produce distinct policies"
            )
    return {
        "passed": not failures and observed == expected,
        "grid_complete": observed == expected,
        "expected_runs": len(expected),
        "completed_runs": len(observed & expected),
        "victim_bank_digest": (
            next(iter(bank_digests))
            if len(bank_digests) == 1
            else None
        ),
        "policy_checkpoints": checkpoints,
        "failures": failures,
        "requirements": [
            "the exact family-by-seed source grid is complete",
            "every victim accuracy gate passes",
            "behavior cloning beats its preregistered representation baselines",
            "the learned policy beats matched controls on exact sources",
            "the learned policy beats matched controls on unseen source instances",
            "no target-family attack query occurs before this gate passes",
        ],
    }
