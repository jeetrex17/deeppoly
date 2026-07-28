"""Preregistered Stage A temperature screen over frozen Phase 1 sources.

This module deliberately has no interface for held-out-family evaluation.
Only verified Phase 1 source checkpoints, exact-source victim instances, and
the fixed source-development split are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from .artifacts import sha256_file
from .paths import resolve_descendant
from .results import ResearchResultRow
from .verified_artifacts import load_verified_json


FOLDS = ("classical_cnn", "modern_cnn", "transformer")
STAGE_A_TEMPERATURES = (0.25, 0.5, 0.75, 1.0, 1.5)
STAGE_A_POLICY_SEEDS = (17,)
STAGE_A_ELIGIBLE_IMAGES = 64
STAGE_A_MAX_SECONDS = 600.0
STAGE_A_RANKING_TIE_BAND = 0.002
STAGE_A_RANKING_RULE = (
    "Rank by macro ASR gain versus matched score greedy. Treat candidates "
    "within 0.002 of the best macro ASR gain as tied, then rank by higher "
    "macro AUC gain versus score greedy, then lower temperature."
)
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class StageARequest:
    phase1_manifest: Path
    phase1_root: Path
    output_dir: Path
    data_root: Path
    seeds: tuple[int, ...] = STAGE_A_POLICY_SEEDS
    folds: tuple[str, ...] = FOLDS
    temperatures: tuple[float, ...] = STAGE_A_TEMPERATURES
    eligible_images_per_family: int = STAGE_A_ELIGIBLE_IMAGES
    deadline_seconds: float = STAGE_A_MAX_SECONDS
    device: str = "cuda"
    download: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase1_manifest", Path(self.phase1_manifest))
        object.__setattr__(self, "phase1_root", Path(self.phase1_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "folds", tuple(self.folds))
        object.__setattr__(
            self,
            "temperatures",
            tuple(float(value) for value in self.temperatures),
        )
        if self.seeds != STAGE_A_POLICY_SEEDS:
            raise ValueError("Stage A is preregistered for policy seed 17")
        if (
            not self.folds
            or len(self.folds) != len(set(self.folds))
            or any(fold not in FOLDS for fold in self.folds)
            or tuple(fold for fold in FOLDS if fold in self.folds)
            != self.folds
        ):
            raise ValueError("folds must be an ordered non-empty subset")
        if self.temperatures != STAGE_A_TEMPERATURES:
            raise ValueError("Stage A temperatures are preregistered")
        if self.eligible_images_per_family != STAGE_A_ELIGIBLE_IMAGES:
            raise ValueError("Stage A requires 64 eligible images per family")
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or not 0 < self.deadline_seconds <= STAGE_A_MAX_SECONDS
        ):
            raise ValueError("Stage A deadline must be in (0, 600] seconds")
        if self.device != "cuda":
            raise ValueError("Stage A requires the designated CUDA device")
        if not isinstance(self.download, bool):
            raise ValueError("download must be boolean")

        root = self.phase1_root.resolve()
        manifest = resolve_descendant(
            root,
            self.phase1_manifest,
            label="Phase 1 manifest",
        )
        object.__setattr__(self, "phase1_root", root)
        object.__setattr__(self, "phase1_manifest", manifest)
        output = self.output_dir.resolve()
        if output == root:
            raise ValueError("Stage A output cannot replace the Phase 1 root")
        try:
            output.relative_to((root / "runs").resolve())
        except ValueError:
            pass
        else:
            raise ValueError("Stage A output cannot overwrite Phase 1 runs")
        object.__setattr__(self, "output_dir", output)


@dataclass(frozen=True)
class Phase1SourceFold:
    seed: int
    heldout_family: str
    source_families: tuple[str, ...]
    fingerprint: str
    run_dir: Path
    policy_path: Path
    source_results_path: Path
    run_manifest: dict[str, object]
    source_victims: dict[str, tuple[dict[str, object], ...]]


@dataclass(frozen=True)
class Phase1Selection:
    manifest_path: Path
    manifest_sha256: str
    split_digest: str
    source_gate_digest: str
    dataset_version: str
    dataset_content_sha256: str | None
    target_calls: int
    folds: tuple[Phase1SourceFold, ...]


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _verify_file(path: Path, expected: object, *, label: str) -> str:
    digest = _require_digest(expected, label=f"{label} manifest checksum")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"{label} is incomplete")
    recorded = sidecar.read_text().strip()
    if _HEX_DIGEST.fullmatch(recorded) is None:
        raise ValueError(f"{label} checksum sidecar is malformed")
    actual = sha256_file(path)
    if actual != recorded or actual != digest:
        raise ValueError(f"{label} checksum verification failed")
    return actual


def _safe_artifact_name(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} is not a safe artifact name")
    return value


def _config_digest(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_seal(payload: Mapping[str, object], *, label: str) -> None:
    if (
        payload.get("target_calls") != 0
        or payload.get("target_evaluation_performed") is not False
        or payload.get("research_valid") is not False
    ):
        raise ValueError(f"{label} violates the zero-target-call seal")


def _critical_run_fields_match(
    embedded: Mapping[str, object],
    verified: Mapping[str, object],
) -> bool:
    fields = (
        "schema_version",
        "status",
        "seed",
        "target_family",
        "target_calls",
        "target_evaluation_performed",
        "split_seed",
        "victim_seed",
        "split_digest",
        "data_role_digests",
        "fingerprint",
        "config",
        "config_digest",
        "source_families",
        "victim_cache_digest",
        "victim_cache_contract",
        "victim_instances",
        "policy",
    )
    return all(embedded.get(field) == verified.get(field) for field in fields)


def load_phase1_source_selection(
    request: StageARequest,
) -> Phase1Selection:
    """Load and verify only the selected Phase 1 source folds."""

    study = load_verified_json(request.phase1_manifest)
    if (
        study.get("schema_version") != 1
        or study.get("status") != "source_learning_failed"
    ):
        raise ValueError("Phase 1 source study is not completed")
    _validate_source_seal(study, label="Phase 1 study")
    study_config = _require_mapping(
        study.get("config"),
        label="Phase 1 study config",
    )
    if (
        tuple(study_config.get("target_families", ())) != FOLDS
        or int(study_config.get("split_seed", -1)) != 20260727
        or int(study_config.get("victim_seed", -1)) != 1000000
        or 17 not in tuple(study_config.get("seeds", ()))
    ):
        raise ValueError("Phase 1 study does not match the locked Stage A split")
    raw_runs = study.get("source_runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Phase 1 study has no source runs")
    indexed: dict[tuple[int, str], Mapping[str, object]] = {}
    for item in raw_runs:
        run = _require_mapping(item, label="Phase 1 source run")
        key = (int(run.get("seed", -1)), str(run.get("target_family", "")))
        if key in indexed:
            raise ValueError("Phase 1 source grid contains duplicate cells")
        indexed[key] = run

    selected: list[Phase1SourceFold] = []
    split_digests: set[str] = set()
    source_gate_digests: set[str] = set()
    cache_contracts: list[object] = []
    dataset_versions: set[str] = set()
    dataset_content_digests: set[str] = set()
    for seed in request.seeds:
        for heldout_family in request.folds:
            embedded = indexed.get((seed, heldout_family))
            if embedded is None:
                raise ValueError("selected Phase 1 source cell is missing")
            if embedded.get("status") != "source_complete":
                raise ValueError("selected Phase 1 source cell is incomplete")
            _validate_source_seal(
                embedded,
                label=f"Phase 1 {heldout_family}/{seed}",
            )
            fingerprint = _require_digest(
                embedded.get("fingerprint"),
                label="run fingerprint",
            )
            run_dir = resolve_descendant(
                request.phase1_root,
                Path("runs") / fingerprint[:12],
                label="Phase 1 run directory",
            )
            verified = load_verified_json(run_dir / "manifest.json")
            if not _critical_run_fields_match(embedded, verified):
                raise ValueError("embedded and verified run manifests disagree")
            config = _require_mapping(
                verified.get("config"),
                label="Phase 1 run config",
            )
            if (
                verified.get("config_digest") != _config_digest(config)
                or config.get("seed") != seed
                or config.get("target_family") != heldout_family
                or config.get("split_seed") != 20260727
                or config.get("victim_seed") != 1000000
                or config.get("query_budget") != 50
                or config.get("source_instances_per_family") != 2
                or config.get("source_evaluation_images") != 200
                or config.get("train_ablation_policies") is not True
            ):
                raise ValueError("Phase 1 run config violates the locked contract")
            source_families = tuple(
                family for family in FOLDS if family != heldout_family
            )
            if tuple(verified.get("source_families", ())) != source_families:
                raise ValueError("Phase 1 source-family split mismatch")
            if (
                _require_mapping(
                    verified.get("victim_accuracy_gate"),
                    label="victim accuracy gate",
                ).get("passed")
                is not True
            ):
                raise ValueError("Phase 1 victim accuracy gate did not pass")
            dataset = _require_mapping(
                verified.get("dataset"),
                label="Phase 1 dataset provenance",
            )
            dataset_version = dataset.get("version")
            if not isinstance(dataset_version, str) or not dataset_version:
                raise ValueError("Phase 1 dataset version is missing")
            dataset_versions.add(dataset_version)
            content_match = re.search(
                r"(?:^|;)content-sha256=([0-9a-f]{64})(?:;|$)",
                dataset_version,
            )
            if content_match is not None:
                dataset_content_digests.add(content_match.group(1))

            policy_block = _require_mapping(
                verified.get("policy"),
                label="Phase 1 policy",
            )
            policy_path = resolve_descendant(
                run_dir,
                "policy.pt",
                label="Phase 1 policy checkpoint",
            )
            _verify_file(
                policy_path,
                policy_block.get("checkpoint_sha256"),
                label="Phase 1 policy checkpoint",
            )
            _require_digest(
                policy_block.get("persistent_digest"),
                label="Phase 1 persistent policy digest",
            )

            source_results = resolve_descendant(
                run_dir,
                "source_results.jsonl",
                label="Phase 1 source result rows",
            )
            source_cache = load_verified_json(
                run_dir / "source_evaluation.json"
            )
            if (
                not source_results.is_file()
                or sha256_file(source_results)
                != _require_digest(
                    source_cache.get("results_sha256"),
                    label="Phase 1 source-result checksum",
                )
            ):
                raise ValueError("Phase 1 source-result checksum failed")

            cache_digest = _require_digest(
                verified.get("victim_cache_digest"),
                label="Phase 1 victim-cache digest",
            )
            victim_root = resolve_descendant(
                request.phase1_root,
                Path("runs")
                / "victim_cache"
                / cache_digest[:12],
                label="Phase 1 victim cache",
            )
            victim_instances = _require_mapping(
                verified.get("victim_instances"),
                label="Phase 1 victim instances",
            )
            exact_sources: dict[
                str, tuple[dict[str, object], ...]
            ] = {}
            for family in source_families:
                raw_family = victim_instances.get(family)
                if not isinstance(raw_family, list) or len(raw_family) < 2:
                    raise ValueError("exact-source victim instances are missing")
                specs: list[dict[str, object]] = []
                for expected_instance, raw_spec in enumerate(raw_family[:2]):
                    spec = dict(
                        _require_mapping(
                            raw_spec,
                            label="exact-source victim",
                        )
                    )
                    victim_id = _safe_artifact_name(
                        spec.get("victim_id"),
                        label="victim ID",
                    )
                    if (
                        spec.get("family") != family
                        or spec.get("instance_index") != expected_instance
                    ):
                        raise ValueError("exact-source victim order mismatch")
                    checkpoint = resolve_descendant(
                        victim_root,
                        f"{victim_id}.pt",
                        label="exact-source victim checkpoint",
                    )
                    _verify_file(
                        checkpoint,
                        spec.get("checkpoint_sha256"),
                        label="exact-source victim checkpoint",
                    )
                    spec["local_checkpoint"] = str(checkpoint)
                    specs.append(spec)
                exact_sources[family] = tuple(specs)

            split_digest = _require_digest(
                verified.get("split_digest"),
                label="Phase 1 split digest",
            )
            roles = _require_mapping(
                verified.get("data_role_digests"),
                label="Phase 1 role digests",
            )
            source_gate_digest = _require_digest(
                roles.get("source_gate"),
                label="Phase 1 source-gate split digest",
            )
            split_digests.add(split_digest)
            source_gate_digests.add(source_gate_digest)
            cache_contracts.append(verified.get("victim_cache_contract"))
            selected.append(
                Phase1SourceFold(
                    seed=seed,
                    heldout_family=heldout_family,
                    source_families=source_families,
                    fingerprint=fingerprint,
                    run_dir=run_dir,
                    policy_path=policy_path,
                    source_results_path=source_results,
                    run_manifest=dict(verified),
                    source_victims=exact_sources,
                )
            )
    if (
        len(split_digests) != 1
        or len(source_gate_digests) != 1
        or len(
            {
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                for value in cache_contracts
            }
        )
        != 1
        or len(dataset_versions) != 1
        or len(dataset_content_digests) > 1
    ):
        raise ValueError("selected Phase 1 cells disagree on source provenance")
    return Phase1Selection(
        manifest_path=request.phase1_manifest,
        manifest_sha256=sha256_file(request.phase1_manifest),
        split_digest=next(iter(split_digests)),
        source_gate_digest=next(iter(source_gate_digests)),
        dataset_version=next(iter(dataset_versions)),
        dataset_content_sha256=(
            next(iter(dataset_content_digests))
            if dataset_content_digests
            else None
        ),
        target_calls=0,
        folds=tuple(selected),
    )


def select_fixed_source_development_indices(
    rows: Sequence[ResearchResultRow],
    *,
    family: str,
    exact_source_victim_ids: Sequence[str],
    candidate_indices: Sequence[int],
    count: int = STAGE_A_ELIGIBLE_IMAGES,
) -> tuple[int, ...]:
    """Select the first common clean-correct Phase 1 source examples."""

    victim_ids = tuple(exact_source_victim_ids)
    candidates = tuple(int(index) for index in candidate_indices)
    if (
        family not in FOLDS
        or not victim_ids
        or len(victim_ids) != len(set(victim_ids))
        or not candidates
        or len(candidates) != len(set(candidates))
        or count < 1
    ):
        raise ValueError("invalid fixed source-development cohort request")
    candidate_set = set(candidates)
    expected = {
        (victim_id, index)
        for victim_id in victim_ids
        for index in candidates
    }
    flags: dict[tuple[str, int], set[bool]] = {
        key: set() for key in expected
    }
    methods: dict[tuple[str, int], set[str]] = {
        key: set() for key in expected
    }
    for row in rows:
        if row.victim_family != family or row.victim_id not in victim_ids:
            continue
        prefix = f"cifar10:{family}:{row.victim_id}:"
        if not row.sample_id.startswith(prefix):
            raise ValueError("Phase 1 source row sample ID is malformed")
        try:
            index = int(row.sample_id[len(prefix):])
        except ValueError as error:
            raise ValueError("Phase 1 source index is malformed") from error
        if index not in candidate_set:
            continue
        key = (row.victim_id, index)
        flags[key].add(bool(row.clean_correct))
        methods[key].add(row.method)
    if any(not values for values in flags.values()):
        raise ValueError("Phase 1 source rows do not cover the fixed split")
    if any(len(values) != 1 for values in flags.values()):
        raise ValueError("Phase 1 methods disagree on clean correctness")
    method_sets = {frozenset(value) for value in methods.values()}
    if len(method_sets) != 1:
        raise ValueError("Phase 1 method coverage is not aligned")
    selected = tuple(
        index
        for index in candidates
        if all(flags[(victim_id, index)] == {True} for victim_id in victim_ids)
    )[:count]
    if len(selected) != count:
        raise ValueError("insufficient common clean-correct source images")
    return selected
