"""Shared fixtures for the Stage A temperature diagnostic tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rl_transfer.artifacts import sha256_file
from rl_transfer.cifar_config import MacPilotConfig
from rl_transfer.phase2_temperature_screen import (
    FOLDS,
    STAGE_A_TEMPERATURES,
    Phase1Selection,
    Phase1SourceFold,
    StageARequest,
)
from rl_transfer.results import ResearchResultRow
from rl_transfer.verified_artifacts import write_verified_json


def write_bytes_with_sidecar(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def phase1_fixture(root: Path, folds: tuple[str, ...]) -> Path:
    base_payload = json.loads(
        Path("configs/rl_transfer/cifar10_rtx_bc_ppo.json").read_text()
    )
    source_runs: list[dict[str, object]] = []
    for heldout_family in folds:
        fingerprint = hashlib.sha256(
            f"fold:{heldout_family}".encode()
        ).hexdigest()
        run_dir = root / "runs" / fingerprint[:12]
        run_dir.mkdir(parents=True, exist_ok=True)
        policy_digest = write_bytes_with_sidecar(
            run_dir / "policy.pt",
            f"policy:{heldout_family}".encode(),
        )
        raw_rows = run_dir / "source_results.jsonl"
        raw_rows.write_text("")
        source_cache = {
            "results_sha256": sha256_file(raw_rows),
            "query_traces_sha256": "0" * 64,
        }
        write_verified_json(run_dir / "source_evaluation.json", source_cache)

        config_payload = {
            **base_payload,
            "name": f"phase1-{heldout_family}-17",
            "seed": 17,
            "target_family": heldout_family,
            "split_seed": 20260727,
            "victim_seed": 1000000,
        }
        MacPilotConfig(**config_payload)
        config_digest = hashlib.sha256(
            json.dumps(
                config_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        source_families = tuple(
            family for family in FOLDS if family != heldout_family
        )
        victim_instances: dict[str, list[dict[str, object]]] = {}
        for family in FOLDS:
            entries: list[dict[str, object]] = []
            for instance in range(3):
                victim_id = f"fixture_{family}__instance_{instance}"
                checkpoint = (
                    root
                    / "runs"
                    / "victim_cache"
                    / ("a" * 12)
                    / f"{victim_id}.pt"
                )
                checkpoint_digest = write_bytes_with_sidecar(
                    checkpoint,
                    f"{family}:{instance}".encode(),
                )
                entries.append(
                    {
                        "victim_id": victim_id,
                        "family": family,
                        "instance_index": instance,
                        "training_seed": 100 + instance,
                        "checkpoint": f"/remote/{checkpoint.name}",
                        "checkpoint_sha256": checkpoint_digest,
                        "source_validation_accuracy": 0.9,
                    }
                )
            victim_instances[family] = entries
        run = {
            "schema_version": 1,
            "status": "source_complete",
            "research_valid": False,
            "seed": 17,
            "target_family": heldout_family,
            "target_calls": 0,
            "target_evaluation_performed": False,
            "split_seed": 20260727,
            "victim_seed": 1000000,
            "split_digest": "b" * 64,
            "data_role_digests": {"source_gate": "c" * 64},
            "dataset": {"name": "CIFAR-10", "version": "in-memory"},
            "fingerprint": fingerprint,
            "config": config_payload,
            "config_digest": config_digest,
            "source_families": list(source_families),
            "victim_cache_digest": "a" * 64,
            "victim_cache_contract": {"schema_version": 1},
            "victim_accuracy_gate": {"passed": True},
            "victim_instances": victim_instances,
            "policy": {
                "checkpoint": "/remote/policy.pt",
                "checkpoint_sha256": policy_digest,
                "persistent_digest": "d" * 64,
            },
        }
        write_verified_json(run_dir / "manifest.json", run)
        source_runs.append(run)
    manifest = {
        "schema_version": 1,
        "name": "phase1-fixture",
        "status": "source_learning_failed",
        "research_valid": False,
        "target_calls": 0,
        "target_evaluation_performed": False,
        "config": {
            "seeds": [17],
            "target_families": list(FOLDS),
            "split_seed": 20260727,
            "victim_seed": 1000000,
        },
        "source_runs": source_runs,
    }
    path = root / "study_manifest.json"
    write_verified_json(path, manifest)
    return path


def stage_a_request(
    manifest_path: Path,
    output_dir: Path,
    **overrides: object,
) -> StageARequest:
    values: dict[str, object] = {
        "phase1_manifest": manifest_path,
        "phase1_root": manifest_path.parent,
        "output_dir": output_dir,
        "data_root": Path("data/cifar10"),
        "seeds": (17,),
        "folds": FOLDS,
        "temperatures": STAGE_A_TEMPERATURES,
        "eligible_images_per_family": 64,
        "deadline_seconds": 600.0,
        "device": "cuda",
        "download": False,
    }
    values.update(overrides)
    return StageARequest(**values)


def result_row(
    victim_id: str,
    family: str,
    index: int,
    *,
    clean_correct: bool,
    method: str = "score_greedy",
) -> ResearchResultRow:
    return ResearchResultRow(
        sample_id=f"cifar10:{family}:{victim_id}:{index}",
        victim_id=victim_id,
        victim_family=family,
        method=method,
        threat_model="T1",
        seed=17,
        query_budget=50,
        clean_correct=clean_correct,
        success=False,
        query_to_success=None,
        total_target_calls=50,
        linf=0.0,
        l2=0.0,
        policy_digest="digest",
        action_trace=tuple(0 for _ in range(49)),
    )


def fold_summary(
    fold: str,
    values: dict[float, tuple[float, float, float]],
) -> dict[str, object]:
    score_asr, score_auc = 0.08, 0.03
    return {
        "seed": 17,
        "heldout_family": fold,
        "score_greedy": {"asr": score_asr, "auc": score_auc},
        "temperatures": {
            str(temperature): {
                "asr": asr,
                "auc": auc,
                "normalized_action_entropy": entropy,
                "asr_gain_vs_score": asr - score_asr,
                "auc_gain_vs_score": auc - score_auc,
                "frozen": True,
            }
            for temperature, (asr, auc, entropy) in values.items()
        },
    }


def phase1_selection(root: Path) -> Phase1Selection:
    folds = tuple(
        Phase1SourceFold(
            seed=17,
            heldout_family=family,
            source_families=tuple(item for item in FOLDS if item != family),
            fingerprint=f"{index + 1:064x}",
            run_dir=root / f"run-{index}",
            policy_path=root / f"policy-{index}.pt",
            source_results_path=root / f"rows-{index}.jsonl",
            run_manifest={},
            source_victims={},
        )
        for index, family in enumerate(FOLDS)
    )
    return Phase1Selection(
        manifest_path=root / "manifest.json",
        manifest_sha256="a" * 64,
        split_digest="b" * 64,
        source_gate_digest="c" * 64,
        dataset_version="in-memory",
        dataset_content_sha256=None,
        target_calls=0,
        folds=folds,
    )
