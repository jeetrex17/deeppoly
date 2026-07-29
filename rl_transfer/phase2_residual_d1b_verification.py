"""Independent child-artifact verification for completed D1b studies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

from .artifacts import sha256_file
from .phase2_residual_d1 import validate_source_only_payload as _source_only
from .phase2_residual_d1_evidence import (
    load_verified_jsonl_records,
    verify_d1_raw_evidence,
    verify_d1_recorded_summaries,
)
from .phase2_residual_d1_runner import _verified_child
from .phase2_residual_d1b import D1B_BLOCK_ENDPOINTS, D1B_METHODS
from .phase2_residual_d1b_artifacts import canonical_json_digest
from .verified_artifacts import load_verified_json


_DIGEST = re.compile(r"[0-9a-f]{64}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _verified_receipt(
    output_dir: Path,
    endpoint: int,
    previous: Mapping[str, object] | None,
) -> dict[str, object]:
    receipt_path = output_dir / f"ppo_block_{endpoint:03d}.receipt.json"
    receipt = load_verified_json(receipt_path)
    _source_only(receipt, "D1b block receipt")
    checkpoint = _mapping(receipt.get("checkpoint"), "D1b block checkpoint")
    metadata = _mapping(receipt.get("core_metadata"), "D1b core metadata")
    expected_checkpoint = f"ppo_block_{endpoint:03d}.pt"
    metadata_digest = canonical_json_digest(metadata)
    policy_digest = metadata.get("ppo_policy_digest")
    opaque_reference = f"d1b-{policy_digest}-{metadata_digest}"
    if (
        receipt.get("schema_version") != 1
        or receipt.get("name") != "phase2-d1b-residual-ranker-ppo-receipt"
        or set(checkpoint) != {"name", "sha256"}
        or checkpoint.get("name") != expected_checkpoint
        or checkpoint.get("sha256") != sha256_file(output_dir / expected_checkpoint)
        or metadata.get("block_index") != endpoint // 50
        or metadata.get("episodes_completed") != endpoint
        or metadata.get("episode_offset") != endpoint - 50
        or metadata.get("episodes") != 50
        or not isinstance(policy_digest, str)
        or _DIGEST.fullmatch(policy_digest) is None
        or receipt.get("policy_digest") != policy_digest
        or receipt.get("metadata_digest") != metadata_digest
        or receipt.get("opaque_reference") != opaque_reference
    ):
        raise ValueError("D1b receipt identity, checkpoint, or digest is invalid")
    expected_parent = (
        None
        if previous is None
        else {
            "reference": previous["opaque_reference"],
            "policy_digest": previous["policy_digest"],
            "metadata_digest": previous["metadata_digest"],
        }
    )
    if metadata.get("parent_checkpoint") != expected_parent:
        raise ValueError("D1b receipt parent chain is invalid")
    return receipt


def d1b_block_records(output_dir: Path) -> list[dict[str, object]]:
    """Return fixed-name block records from verified receipt artifacts."""

    records: list[dict[str, object]] = []
    previous: dict[str, object] | None = None
    for endpoint in D1B_BLOCK_ENDPOINTS:
        receipt_path = output_dir / f"ppo_block_{endpoint:03d}.receipt.json"
        receipt = _verified_receipt(output_dir, endpoint, previous)
        checkpoint = _mapping(receipt.get("checkpoint"), "D1b block checkpoint")
        expected_checkpoint = f"ppo_block_{endpoint:03d}.pt"
        records.append(
            {
                "endpoint": endpoint,
                "checkpoint": expected_checkpoint,
                "checkpoint_sha256": checkpoint.get("sha256"),
                "receipt": receipt_path.name,
                "receipt_sha256": sha256_file(receipt_path),
                "opaque_reference": receipt.get("opaque_reference"),
                "policy_digest": receipt.get("policy_digest"),
                "metadata_digest": receipt.get("metadata_digest"),
            }
        )
        previous = receipt
    return records


def verify_complete_d1b_children(
    output_dir: Path,
    manifest: Mapping[str, object],
) -> None:
    """Verify fixed child names, digests, raw evidence, and the block chain."""

    _source_only(manifest, "D1b complete manifest")
    if (
        manifest.get("status") != "complete"
        or manifest.get("evaluation_role") != "d1b_evaluation"
    ):
        raise ValueError("D1b complete manifest has an invalid status or role")
    checkpoint = _mapping(
        manifest.get("checkpoint"),
        "D1b calibrated checkpoint",
    )
    if checkpoint.get("name") != "residual_ranker_ppo.pt":
        raise ValueError("D1b calibrated checkpoint filename is invalid")
    _verified_child(
        output_dir / "residual_ranker_ppo.pt",
        checkpoint.get("sha256"),
    )
    _verified_child(
        output_dir / "source_results.jsonl",
        manifest.get("results_sha256"),
    )
    _verified_child(
        output_dir / "source_query_traces.jsonl",
        manifest.get("query_traces_sha256"),
    )
    figures = _mapping(manifest.get("figures"), "D1b figures")
    if set(figures) != {"asr_by_query.svg", "final_asr.svg"}:
        raise ValueError("D1b complete figure set is invalid")
    for name, digest in figures.items():
        _verified_child(output_dir / name, digest)
    rows = load_verified_jsonl_records(output_dir / "source_results.jsonl")
    traces = load_verified_jsonl_records(output_dir / "source_query_traces.jsonl")
    recomputed = verify_d1_raw_evidence(
        rows,
        traces,
        expected_methods=D1B_METHODS,
    )
    verify_d1_recorded_summaries(
        recomputed,
        _mapping(manifest.get("source_evaluation"), "D1b evaluation"),
    )
    if recomputed != manifest.get("raw_evidence_verification"):
        raise ValueError("D1b raw evidence recomputation changed")
    blocks = manifest.get("ppo_blocks")
    if not isinstance(blocks, list) or len(blocks) != len(D1B_BLOCK_ENDPOINTS):
        raise ValueError("D1b complete block artifact list is invalid")
    previous: dict[str, object] | None = None
    for expected_endpoint, record in zip(D1B_BLOCK_ENDPOINTS, blocks):
        block = _mapping(record, "D1b block record")
        if block.get("endpoint") != expected_endpoint:
            raise ValueError("D1b block endpoints are not contiguous")
        expected_checkpoint = f"ppo_block_{expected_endpoint:03d}.pt"
        expected_receipt = f"ppo_block_{expected_endpoint:03d}.receipt.json"
        if (
            block.get("checkpoint") != expected_checkpoint
            or block.get("receipt") != expected_receipt
        ):
            raise ValueError("D1b block artifact filename is invalid")
        _verified_child(
            output_dir / expected_checkpoint,
            block.get("checkpoint_sha256"),
        )
        _verified_child(
            output_dir / expected_receipt,
            block.get("receipt_sha256"),
        )
        receipt = _verified_receipt(output_dir, expected_endpoint, previous)
        if any(
            block.get(field) != receipt.get(receipt_field)
            for field, receipt_field in (
                ("opaque_reference", "opaque_reference"),
                ("policy_digest", "policy_digest"),
                ("metadata_digest", "metadata_digest"),
            )
        ):
            raise ValueError("D1b block record differs from its receipt")
        previous = receipt


__all__ = ("d1b_block_records", "verify_complete_d1b_children")
