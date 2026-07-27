"""Canonical provenance digests for fixed victim banks."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence


def victim_bank_digest(
    victim_instances: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
) -> str:
    """Hash sorted victim identities and checkpoint content digests."""

    rows: list[dict[str, str]] = []
    for family, instances in sorted(victim_instances.items()):
        if not instances:
            raise ValueError("every victim family must contain an instance")
        for metrics in instances:
            victim_id = metrics.get("victim_id")
            checkpoint = metrics.get("checkpoint_sha256")
            if (
                not isinstance(victim_id, str)
                or not victim_id
                or not isinstance(checkpoint, str)
                or len(checkpoint) != 64
            ):
                raise ValueError(
                    "victim provenance requires IDs and SHA-256 digests"
                )
            rows.append(
                {
                    "family": str(family),
                    "victim_id": victim_id,
                    "checkpoint_sha256": checkpoint,
                }
            )
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
