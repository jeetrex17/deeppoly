"""Input validation and redaction for the Phase 1 evidence exporter."""

from __future__ import annotations

import ipaddress
import math
from pathlib import Path
import re
from typing import Mapping, Sequence
from urllib.parse import urlsplit

from .artifacts import sha256_file
from .paths import resolve_descendant
from .verified_artifacts import load_verified_json


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_IPV4_PATTERN = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
_HOME_PATH_PATTERN = re.compile(
    r"(?i)(?:^|[\s\"'=:(])"
    r"(?:~[/\\]|/(?:home|Users|root)/|"
    r"[A-Z]:[\\/](?:Users|Documents and Settings)[\\/])"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{4,}")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?:password|passwd|api[_ -]?key|access[_ -]?token|"
    r"refresh[_ -]?token|client[_ -]?secret|authorization)"
    r"\b\s*[:=]\s*\S+"
)
_SECRET_FIELD_PATTERN = re.compile(
    r"(?:^|_)(?:password|passwd|access_token|accesstoken|"
    r"refresh_token|refreshtoken|api_key|apikey|authorization|"
    r"bearer_token|client_secret|private_key|secret|secret_key|"
    r"credential|credentials)(?:$|_)"
)


def require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def require_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _normalized_field_name(value: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")


def _is_secret_field(value: object) -> bool:
    return (
        isinstance(value, str)
        and _SECRET_FIELD_PATTERN.search(_normalized_field_name(value)) is not None
    )


def _is_internal_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.strip("[]()"))
    except ValueError:
        return False
    return not address.is_global


def _contains_internal_address(value: str) -> bool:
    if any(_is_internal_address(match) for match in _IPV4_PATTERN.findall(value)):
        return True
    stripped = value.strip()
    if _is_internal_address(stripped):
        return True
    if "://" not in value:
        return False
    try:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            return True
        return parsed.hostname is not None and _is_internal_address(parsed.hostname)
    except ValueError:
        return True


def _is_sensitive_string(value: str) -> bool:
    return bool(
        value.startswith("/")
        or _ABSOLUTE_WINDOWS_PATH.match(value)
        or _HOME_PATH_PATTERN.search(value)
        or ".pt" in value.lower()
        or _BEARER_PATTERN.search(value)
        or _SECRET_ASSIGNMENT_PATTERN.search(value)
        or _contains_internal_address(value)
    )


def validate_portable_value(value: object, label: str = "export") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _is_secret_field(key):
                raise ValueError(f"{label} contains non-portable or sensitive text")
            validate_portable_value(key, label)
            validate_portable_value(item, label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            validate_portable_value(item, label)
        return
    if isinstance(value, str):
        if _is_sensitive_string(value):
            raise ValueError(f"{label} contains non-portable or sensitive text")


def validated_output_directory(
    source_root: Path,
    output_dir: Path,
) -> Path:
    source = source_root.resolve(strict=True)
    if output_dir.is_symlink():
        raise ValueError("evidence output directory cannot be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir.resolve(strict=True)
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError(
            "evidence output directory must not overlap the source archive"
        )
    return output


def verified_source_runs(
    study_root: Path,
    study: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    runs_value = require_sequence(study.get("source_runs"), "source_runs")
    if not runs_value:
        raise ValueError("source_runs cannot be empty")
    verified: list[dict[str, object]] = []
    checksums: list[dict[str, str]] = []
    seen: set[str] = set()
    runs_root = resolve_descendant(study_root, "runs", label="run root")
    for index, embedded_value in enumerate(runs_value):
        embedded = dict(require_mapping(embedded_value, f"source_runs[{index}]"))
        fingerprint = embedded.get("fingerprint")
        if (
            not isinstance(fingerprint, str)
            or SHA256_PATTERN.fullmatch(fingerprint) is None
        ):
            raise ValueError(f"source_runs[{index}] fingerprint is invalid")
        if fingerprint in seen:
            raise ValueError("source run fingerprints must be unique")
        seen.add(fingerprint)
        run_dir = resolve_descendant(
            runs_root,
            fingerprint[:12],
            label=f"source run {fingerprint[:12]}",
        )
        manifest_path = resolve_descendant(
            run_dir,
            "manifest.json",
            label="run manifest",
        )
        run = load_verified_json(manifest_path)
        if run != embedded:
            raise ValueError("run manifest disagrees with study manifest")
        if run.get("status") != "source_complete":
            raise ValueError("every exported run must be source_complete")
        if run.get("target_evaluation_performed") is not False:
            raise ValueError("Phase 1 evidence must be target-free")
        if (
            nonnegative_integer(
                run.get("target_calls"),
                "run target_calls",
            )
            != 0
        ):
            raise ValueError("Phase 1 evidence must be target-free")
        target_family = run.get("target_family")
        source_families_value = require_sequence(
            run.get("source_families"),
            "source families",
        )
        if (
            not isinstance(target_family, str)
            or not target_family
            or any(
                not isinstance(family, str) or not family
                for family in source_families_value
            )
        ):
            raise ValueError("run victim-family contract is malformed")
        expected_victim_families = {
            target_family,
            *source_families_value,
        }
        victim_gate = require_mapping(
            run.get("victim_accuracy_gate"),
            "victim clean-accuracy gate",
        )
        victim_instances = require_mapping(
            run.get("victim_instances"),
            "victim instances",
        )
        if (
            victim_gate.get("passed") is not True
            or set(victim_instances) != expected_victim_families
            or any(
                not require_sequence(
                    records,
                    f"{family} victim instances",
                )
                for family, records in victim_instances.items()
            )
        ):
            raise ValueError("all victim families must pass clean-accuracy validation")
        cache_path = resolve_descendant(
            run_dir,
            "source_evaluation.json",
            label="source evaluation cache",
        )
        cache = load_verified_json(cache_path)
        cached_evaluation = require_mapping(
            cache.get("source_evaluation"),
            "cached source_evaluation",
        )
        if cached_evaluation != run.get("source_evaluation"):
            raise ValueError("source evaluation cache disagrees with run manifest")
        results_path = resolve_descendant(
            run_dir,
            "source_results.jsonl",
            label="source result rows",
        )
        traces_path = resolve_descendant(
            run_dir,
            "source_query_traces.jsonl",
            label="source query traces",
        )
        expected_results = digest(
            cache.get("results_sha256"),
            "source results checksum",
        )
        expected_traces = digest(
            cache.get("query_traces_sha256"),
            "source query trace checksum",
        )
        if not results_path.is_file() or sha256_file(results_path) != expected_results:
            raise ValueError("source result rows failed checksum verification")
        if not traces_path.is_file() or sha256_file(traces_path) != expected_traces:
            raise ValueError("source query traces failed checksum verification")
        audits = require_mapping(
            run.get("source_evaluation_audits"),
            "source evaluation audits",
        )
        audit_records = [
            record
            for slices in audits.values()
            for record in require_mapping(slices, "audit slice").values()
        ]
        if not audit_records or any(
            require_mapping(record, "audit record").get("passed") is not True
            for record in audit_records
        ):
            raise ValueError("source evaluation audit did not pass")
        verified.append(
            {
                "run": run,
                "cache": dict(cache),
                "run_dir": run_dir,
            }
        )
        checksums.append(
            {
                "fingerprint": fingerprint,
                "run_manifest_sha256": sha256_file(manifest_path),
                "source_evaluation_sha256": sha256_file(cache_path),
                "source_results_sha256": expected_results,
                "source_query_traces_sha256": expected_traces,
            }
        )
    verified.sort(
        key=lambda item: (
            str(item["run"].get("target_family")),
            int(item["run"].get("seed", -1)),
            str(item["run"].get("fingerprint")),
        )
    )
    checksums.sort(key=lambda item: item["fingerprint"])
    return verified, checksums


def _safe_binding(cache: Mapping[str, object]) -> dict[str, object]:
    binding = require_mapping(cache.get("binding"), "source binding")
    checkpoints_value = require_mapping(
        binding.get("policy_checkpoints"),
        "policy checkpoints",
    )
    checkpoints = {
        str(name): digest(
            require_mapping(record, "policy checkpoint").get("sha256"),
            "policy checkpoint checksum",
        )
        for name, record in checkpoints_value.items()
    }
    return {
        "code_digest": digest(
            binding.get("code_digest"),
            "binding code digest",
        ),
        "config_digest": digest(
            binding.get("config_digest"),
            "binding config digest",
        ),
        "data_role_digests": dict(
            require_mapping(
                binding.get("data_role_digests"),
                "data role digests",
            )
        ),
        "policy_checkpoint_sha256": checkpoints,
        "source_victim_checkpoint_sha256": dict(
            require_mapping(
                binding.get("source_victim_checkpoints"),
                "source victim checkpoints",
            )
        ),
        "split_digest": digest(
            binding.get("split_digest"),
            "binding split digest",
        ),
    }


def compact_raw_runs(
    verified_runs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for item in verified_runs:
        run = require_mapping(item.get("run"), "verified run")
        cache = require_mapping(item.get("cache"), "verified cache")
        policy = require_mapping(run.get("policy"), "policy")
        training = require_mapping(policy.get("training"), "policy training")
        compact.append(
            {
                "fingerprint": str(run["fingerprint"]),
                "name": str(run["name"]),
                "seed": int(run["seed"]),
                "omitted_target_family": str(run["target_family"]),
                "source_families": list(
                    require_sequence(
                        run.get("source_families"),
                        "source families",
                    )
                ),
                "status": str(run["status"]),
                "elapsed_seconds": float(run["elapsed_seconds"]),
                "binding": _safe_binding(cache),
                "policy_checkpoint_sha256": digest(
                    policy.get("checkpoint_sha256"),
                    "main policy checkpoint checksum",
                ),
                "policy_training": {
                    key: training[key]
                    for key in (
                        "episodes",
                        "trained_episodes",
                        "source_calls",
                        "source_calls_by_family",
                        "blocks",
                        "component_ablations",
                        "behavior_cloning",
                    )
                    if key in training
                },
                "victim_clean_accuracy_gate": run["victim_accuracy_gate"],
                "victim_family_instance_counts": {
                    str(family): len(
                        require_sequence(
                            records,
                            f"{family} victim instances",
                        )
                    )
                    for family, records in require_mapping(
                        run.get("victim_instances"),
                        "victim instances",
                    ).items()
                },
                "source_competence_gate": run["source_competence_gate"],
                "source_evaluation_audits": run["source_evaluation_audits"],
                "source_evaluation": cache["source_evaluation"],
            }
        )
    validate_portable_value(compact, "compact raw evidence")
    return compact
