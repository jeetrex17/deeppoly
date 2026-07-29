"""Pure, fail-closed contracts for the preregistered Phase 2 D2 study.

D2 is a multi-seed, source-only GroupDRO behavior-cloning replication.  This
module intentionally contains no model or dataset loading so its role
allocation and promotion logic can be audited independently of training.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .phase2_residual_d1 import D1_HELDOUT_FAMILY, D1_SOURCE_FAMILIES

D2_SOURCE_FOLD_SEED = 17
D2_POLICY_SEEDS = (223, 227, 229)
D2_SOURCE_FAMILIES = D1_SOURCE_FAMILIES
D2_HELDOUT_FAMILY = D1_HELDOUT_FAMILY
D2_GROUPDRO_ETA = 0.1
D2_BC_EPOCHS = 12
D2_GROUPDRO_TRAIN_IMAGES = 600
D2_THRESHOLD_IMAGES = 100
D2_COMPETENCE_IMAGES = 100
D2_EVALUATION_IMAGES = 100
D2_VISITED_POLICY_TRAIN_IMAGES = 600
D2_VISITED_SOURCE_VALIDATION_IMAGES = 700
D2_MIN_MEAN_ASR_GAIN = 0.010
D2_MIN_MEAN_QUERY_AUC_GAIN = 0.005

_ROLE_CONTRACT = {
    "groupdro_training": ("policy_train", D2_GROUPDRO_TRAIN_IMAGES),
    "threshold_selection": ("source_validation", D2_THRESHOLD_IMAGES),
    "competence_gate": ("source_validation", D2_COMPETENCE_IMAGES),
    "evaluation": ("source_validation", D2_EVALUATION_IMAGES),
}


def _locked_integer(value: object, label: str, expected: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"D2 {label} must equal {expected}")


def _probability(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return float(value)


def _indices(values: Sequence[int], label: str) -> tuple[int, ...]:
    try:
        items = tuple(values)
    except TypeError as error:
        raise TypeError(f"{label} must be an index sequence") from error
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in items
    ):
        raise ValueError(f"{label} indices must be non-negative integers")
    if len(items) != len(set(items)):
        raise ValueError(f"{label} indices must be unique")
    return items


@dataclass(frozen=True)
class ResidualD2Request:
    """Immutable request for the preregistered source-only D2 BC rung."""

    source_manifest: Path
    source_root: Path
    output_dir: Path
    data_root: Path
    source_fold_seed: int = D2_SOURCE_FOLD_SEED
    policy_seeds: tuple[int, ...] = D2_POLICY_SEEDS
    heldout_family: str = D2_HELDOUT_FAMILY
    source_families: tuple[str, ...] = D2_SOURCE_FAMILIES
    groupdro_eta: float = D2_GROUPDRO_ETA
    bc_epochs: int = D2_BC_EPOCHS
    groupdro_train_images: int = D2_GROUPDRO_TRAIN_IMAGES
    threshold_images: int = D2_THRESHOLD_IMAGES
    competence_images: int = D2_COMPETENCE_IMAGES
    evaluation_images: int = D2_EVALUATION_IMAGES
    ppo_episodes: int = 0
    device: str = "cuda"
    source_only: bool = True
    hidden_target_evaluation: bool = False
    hidden_target_calls: int = 0
    download: bool = False

    def __post_init__(self) -> None:
        _locked_integer(
            self.source_fold_seed,
            "source fold seed",
            D2_SOURCE_FOLD_SEED,
        )
        try:
            policy_seeds = tuple(self.policy_seeds)
            source_families = tuple(self.source_families)
        except TypeError as error:
            raise TypeError("D2 seeds and source families must be sequences") from error
        if policy_seeds != D2_POLICY_SEEDS:
            raise ValueError(f"D2 policy seeds must equal {D2_POLICY_SEEDS}")
        if source_families != D2_SOURCE_FAMILIES:
            raise ValueError(f"D2 source families must equal {D2_SOURCE_FAMILIES}")
        if self.heldout_family != D2_HELDOUT_FAMILY:
            raise ValueError(f"D2 held-out family must equal {D2_HELDOUT_FAMILY}")
        if (
            isinstance(self.groupdro_eta, bool)
            or not isinstance(self.groupdro_eta, (int, float))
            or not math.isclose(
                float(self.groupdro_eta),
                D2_GROUPDRO_ETA,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"D2 GroupDRO eta must equal {D2_GROUPDRO_ETA}")
        for label, value, expected in (
            ("BC epochs", self.bc_epochs, D2_BC_EPOCHS),
            (
                "GroupDRO training images",
                self.groupdro_train_images,
                D2_GROUPDRO_TRAIN_IMAGES,
            ),
            ("threshold images", self.threshold_images, D2_THRESHOLD_IMAGES),
            ("competence images", self.competence_images, D2_COMPETENCE_IMAGES),
            ("evaluation images", self.evaluation_images, D2_EVALUATION_IMAGES),
            ("PPO episodes", self.ppo_episodes, 0),
            ("hidden-target calls", self.hidden_target_calls, 0),
        ):
            _locked_integer(value, label, expected)
        if self.device != "cuda":
            raise ValueError("D2 execution is locked to CUDA")
        if self.source_only is not True:
            raise ValueError("D2 must remain source-only")
        if self.hidden_target_evaluation is not False:
            raise ValueError("D2 cannot perform hidden-target evaluation")
        if self.download is not False:
            raise ValueError(
                "D2 requires authenticated local data with download disabled"
            )

        source_root = Path(self.source_root).resolve()
        source_manifest = Path(self.source_manifest).resolve()
        output_dir = Path(self.output_dir).resolve()
        data_root = Path(self.data_root).resolve()
        try:
            source_manifest.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                "D2 source manifest must be inside the sealed source root"
            ) from error
        for left, right, label in (
            (output_dir, source_root, "output and sealed source roots"),
            (output_dir, data_root, "output and dataset roots"),
            (data_root, source_root, "dataset and sealed source roots"),
        ):
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(f"D2 {label} must not overlap")

        object.__setattr__(self, "source_manifest", source_manifest)
        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "data_root", data_root)
        object.__setattr__(self, "policy_seeds", policy_seeds)
        object.__setattr__(self, "source_families", source_families)
        object.__setattr__(self, "groupdro_eta", float(self.groupdro_eta))


@dataclass(frozen=True)
class D2SourceRole:
    """One explicitly named source-image role."""

    name: str
    split: str
    sample_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        contract = _ROLE_CONTRACT.get(self.name)
        if contract is None:
            raise ValueError(f"unknown D2 source role: {self.name}")
        expected_split, expected_size = contract
        if self.split != expected_split:
            raise ValueError(
                f"D2 {self.name} must use the {expected_split} split"
            )
        sample_ids = _indices(self.sample_ids, f"D2 {self.name}")
        if len(sample_ids) != expected_size:
            raise ValueError(
                f"D2 {self.name} requires exactly {expected_size} indices"
            )
        object.__setattr__(self, "sample_ids", sample_ids)


@dataclass(frozen=True)
class D2SourceRoles:
    """The four preregistered, pairwise-disjoint D2 source roles."""

    groupdro_training: D2SourceRole
    threshold_selection: D2SourceRole
    competence_gate: D2SourceRole
    evaluation: D2SourceRole

    def __post_init__(self) -> None:
        roles = self.as_tuple
        if any(not isinstance(role, D2SourceRole) for role in roles):
            raise TypeError("D2 roles must use D2SourceRole")
        if tuple(role.name for role in roles) != tuple(_ROLE_CONTRACT):
            raise ValueError("D2 source roles do not match the locked protocol")
        identities = tuple(
            {(role.split, sample_id) for sample_id in role.sample_ids}
            for role in roles
        )
        if any(
            left & right
            for offset, left in enumerate(identities)
            for right in identities[offset + 1 :]
        ):
            raise ValueError("D2 source roles must be pairwise disjoint")

    @property
    def as_tuple(self) -> tuple[D2SourceRole, ...]:
        return (
            self.groupdro_training,
            self.threshold_selection,
            self.competence_gate,
            self.evaluation,
        )


def allocate_d2_roles(
    *,
    policy_train_candidates: Sequence[int],
    source_validation_candidates: Sequence[int],
    forbidden_policy_train_indices: Sequence[int],
    forbidden_source_validation_indices: Sequence[int],
) -> D2SourceRoles:
    """Allocate locked D2 roles from caller-supplied candidate sequences.

    Candidate order is preserved.  Callers therefore control the frozen source
    ordering explicitly, while this function rejects duplicates and excludes
    every supplied historical D1 index before taking the preregistered sizes.
    """

    policy_candidates = _indices(
        policy_train_candidates,
        "D2 policy_train candidates",
    )
    validation_candidates = _indices(
        source_validation_candidates,
        "D2 source_validation candidates",
    )
    forbidden_policy = frozenset(
        _indices(
            forbidden_policy_train_indices,
            "D2 forbidden policy_train",
        )
    )
    forbidden_validation = frozenset(
        _indices(
            forbidden_source_validation_indices,
            "D2 forbidden source_validation",
        )
    )
    if (
        len(forbidden_policy) != D2_VISITED_POLICY_TRAIN_IMAGES
        or forbidden_policy
        != frozenset(policy_candidates[:D2_VISITED_POLICY_TRAIN_IMAGES])
    ):
        raise ValueError(
            "D2 must exclude exactly the first 600 policy_train candidates"
        )
    if (
        len(forbidden_validation) != D2_VISITED_SOURCE_VALIDATION_IMAGES
        or forbidden_validation
        != frozenset(
            validation_candidates[:D2_VISITED_SOURCE_VALIDATION_IMAGES]
        )
    ):
        raise ValueError(
            "D2 must exclude exactly the first 700 source_validation candidates"
        )
    untouched_policy = tuple(
        index for index in policy_candidates if index not in forbidden_policy
    )
    untouched_validation = tuple(
        index for index in validation_candidates if index not in forbidden_validation
    )
    if len(untouched_policy) < D2_GROUPDRO_TRAIN_IMAGES:
        raise ValueError("D2 requires 600 untouched policy_train candidates")
    required_validation = (
        D2_THRESHOLD_IMAGES + D2_COMPETENCE_IMAGES + D2_EVALUATION_IMAGES
    )
    if len(untouched_validation) < required_validation:
        raise ValueError(
            "D2 requires 300 untouched source_validation candidates"
        )

    threshold_end = D2_THRESHOLD_IMAGES
    competence_end = threshold_end + D2_COMPETENCE_IMAGES
    evaluation_end = competence_end + D2_EVALUATION_IMAGES
    roles = D2SourceRoles(
        groupdro_training=D2SourceRole(
            "groupdro_training",
            "policy_train",
            untouched_policy[:D2_GROUPDRO_TRAIN_IMAGES],
        ),
        threshold_selection=D2SourceRole(
            "threshold_selection",
            "source_validation",
            untouched_validation[:threshold_end],
        ),
        competence_gate=D2SourceRole(
            "competence_gate",
            "source_validation",
            untouched_validation[threshold_end:competence_end],
        ),
        evaluation=D2SourceRole(
            "evaluation",
            "source_validation",
            untouched_validation[competence_end:evaluation_end],
        ),
    )
    validate_d2_role_exclusions(
        roles,
        forbidden_policy_train_indices=tuple(forbidden_policy),
        forbidden_source_validation_indices=tuple(forbidden_validation),
    )
    return roles


def validate_d2_role_exclusions(
    roles: D2SourceRoles,
    *,
    forbidden_policy_train_indices: Sequence[int],
    forbidden_source_validation_indices: Sequence[int],
) -> None:
    """Reject any role containing a historically observed D1 index."""

    if not isinstance(roles, D2SourceRoles):
        raise TypeError("D2 exclusion validation requires D2SourceRoles")
    forbidden = {
        "policy_train": frozenset(
            _indices(
                forbidden_policy_train_indices,
                "D2 forbidden policy_train",
            )
        ),
        "source_validation": frozenset(
            _indices(
                forbidden_source_validation_indices,
                "D2 forbidden source_validation",
            )
        ),
    }
    if any(
        forbidden[role.split].intersection(role.sample_ids)
        for role in roles.as_tuple
    ):
        raise ValueError("D2 source roles contain forbidden D1 role indices")


@dataclass(frozen=True)
class D2FamilyThresholdMetrics:
    """Teacher-accuracy threshold metrics for one visible source family."""

    family: str
    accuracy: float
    prior_accuracy: float

    def __post_init__(self) -> None:
        if self.family not in D2_SOURCE_FAMILIES:
            raise ValueError("D2 threshold metric contains a non-source family")
        object.__setattr__(
            self,
            "accuracy",
            _probability(self.accuracy, "D2 threshold accuracy"),
        )
        object.__setattr__(
            self,
            "prior_accuracy",
            _probability(self.prior_accuracy, "D2 threshold prior accuracy"),
        )


@dataclass(frozen=True)
class D2ThresholdCandidate:
    """One global confidence threshold evaluated on every source family."""

    threshold: float
    family_metrics: tuple[D2FamilyThresholdMetrics, ...]
    residual_use_fraction: float
    overrides_enabled: bool
    always_fallback: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(float(self.threshold))
            or float(self.threshold) < 0
        ):
            raise ValueError("D2 threshold must be finite and non-negative")
        try:
            metrics = tuple(self.family_metrics)
        except TypeError as error:
            raise TypeError("D2 family threshold metrics must be a sequence") from error
        if any(not isinstance(metric, D2FamilyThresholdMetrics) for metric in metrics):
            raise TypeError(
                "D2 threshold candidates require D2FamilyThresholdMetrics"
            )
        by_family = {metric.family: metric for metric in metrics}
        if (
            len(metrics) != len(D2_SOURCE_FAMILIES)
            or set(by_family) != set(D2_SOURCE_FAMILIES)
        ):
            raise ValueError(
                "D2 threshold candidate must cover every source family exactly once"
            )
        if not isinstance(self.overrides_enabled, bool) or not isinstance(
            self.always_fallback,
            bool,
        ):
            raise ValueError("D2 threshold flags must be boolean")
        residual_use = _probability(
            self.residual_use_fraction,
            "D2 residual-use fraction",
        )
        if self.always_fallback and (
            self.overrides_enabled or not math.isclose(residual_use, 0.0)
        ):
            raise ValueError(
                "D2 always-fallback must disable overrides and residual use"
            )
        ordered = tuple(by_family[family] for family in D2_SOURCE_FAMILIES)
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "family_metrics", ordered)
        object.__setattr__(self, "residual_use_fraction", residual_use)

    @property
    def macro_accuracy(self) -> float:
        return sum(metric.accuracy for metric in self.family_metrics) / len(
            self.family_metrics
        )

    @property
    def every_family_accuracy_non_regression(self) -> bool:
        return all(
            metric.accuracy >= metric.prior_accuracy
            for metric in self.family_metrics
        )


@dataclass(frozen=True)
class D2ThresholdSelection:
    """Teacher-accuracy selection requiring a separate attack-level gate."""

    selected: D2ThresholdCandidate
    safe_candidate_count: int
    every_family_accuracy_non_regression: bool


def select_family_safe_threshold(
    candidates: Sequence[D2ThresholdCandidate],
) -> D2ThresholdSelection:
    """Select one global threshold without sacrificing any source family."""

    try:
        locked = tuple(candidates)
    except TypeError as error:
        raise TypeError("D2 threshold candidates must be a sequence") from error
    if not locked or any(
        not isinstance(candidate, D2ThresholdCandidate) for candidate in locked
    ):
        raise TypeError("D2 threshold selection requires typed candidates")
    if len({candidate.threshold for candidate in locked}) != len(locked):
        raise ValueError("D2 threshold candidate values must be unique")
    fallbacks = tuple(candidate for candidate in locked if candidate.always_fallback)
    if len(fallbacks) != 1:
        raise ValueError("D2 requires exactly one always-fallback candidate")
    fallback = fallbacks[0]
    if not all(
        math.isclose(
            metric.accuracy,
            metric.prior_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for metric in fallback.family_metrics
    ):
        raise ValueError(
            "D2 always-fallback accuracy must exactly reproduce every prior"
        )

    safe = tuple(
        candidate
        for candidate in locked
        if candidate.every_family_accuracy_non_regression
    )
    if not safe:
        raise ValueError("D2 always-fallback candidate is not family safe")
    selected = max(
        safe,
        key=lambda candidate: (
            candidate.macro_accuracy,
            -candidate.residual_use_fraction,
            candidate.threshold,
        ),
    )
    return D2ThresholdSelection(
        selected=selected,
        safe_candidate_count=len(safe),
        every_family_accuracy_non_regression=(
            selected.every_family_accuracy_non_regression
        ),
    )


@dataclass(frozen=True)
class D2SourceMetric:
    """Paired score-greedy and learned metrics for one seed and family."""

    seed: int
    family: str
    baseline_asr: float
    learned_asr: float
    baseline_query_auc: float
    learned_query_auc: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed not in D2_POLICY_SEEDS
        ):
            raise ValueError(f"D2 metric seed must be one of {D2_POLICY_SEEDS}")
        if self.family not in D2_SOURCE_FAMILIES:
            raise ValueError("D2 metric contains a non-source family")
        for name in (
            "baseline_asr",
            "learned_asr",
            "baseline_query_auc",
            "learned_query_auc",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), f"D2 {name}"),
            )

    @property
    def asr_gain(self) -> float:
        return self.learned_asr - self.baseline_asr

    @property
    def query_auc_gain(self) -> float:
        return self.learned_query_auc - self.baseline_query_auc

    @property
    def non_regression(self) -> bool:
        return self.asr_gain >= 0.0 and self.query_auc_gain >= 0.0


@dataclass(frozen=True)
class D2PromotionDecision:
    """Fail-closed D2 outcome. Hidden-target authorization is impossible."""

    passed: bool
    all_seed_family_non_regression: bool
    source_gates_passed: bool
    artifact_audits_passed: bool
    eligible_for_source_only_ppo: bool
    mean_asr_gain: float
    mean_query_auc_gain: float
    worst_family_mean_asr_gain: float = 0.0
    worst_family_mean_query_auc_gain: float = 0.0
    mean_gain_gate_passed: bool = False
    worst_family_mean_gain_gate_passed: bool = False
    authorizes_hidden_target_evaluation: bool = False
    hidden_target_evaluation_performed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "passed",
            "all_seed_family_non_regression",
            "source_gates_passed",
            "artifact_audits_passed",
            "eligible_for_source_only_ppo",
            "authorizes_hidden_target_evaluation",
            "hidden_target_evaluation_performed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"D2 decision {name} must be boolean")
        if (
            self.authorizes_hidden_target_evaluation
            or self.hidden_target_evaluation_performed
        ):
            raise ValueError("D2 decisions cannot authorize or perform target access")
        numeric = {}
        for name in (
            "mean_asr_gain",
            "mean_query_auc_gain",
            "worst_family_mean_asr_gain",
            "worst_family_mean_query_auc_gain",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"D2 decision {name} must be finite")
            numeric[name] = float(value)
            object.__setattr__(self, name, float(value))
        expected_mean_gate = (
            numeric["mean_asr_gain"] >= D2_MIN_MEAN_ASR_GAIN - 1e-9
            and numeric["mean_query_auc_gain"]
            >= D2_MIN_MEAN_QUERY_AUC_GAIN - 1e-9
        )
        expected_worst_family_gate = (
            numeric["worst_family_mean_asr_gain"] > 0.0
            and numeric["worst_family_mean_query_auc_gain"] > 0.0
        )
        if self.mean_gain_gate_passed != expected_mean_gate:
            raise ValueError("D2 decision mean-gain gate is inconsistent")
        if (
            self.worst_family_mean_gain_gate_passed
            != expected_worst_family_gate
        ):
            raise ValueError("D2 decision worst-family gain gate is inconsistent")
        expected_pass = (
            self.all_seed_family_non_regression
            and self.source_gates_passed
            and self.artifact_audits_passed
            and self.mean_gain_gate_passed
            and self.worst_family_mean_gain_gate_passed
        )
        if self.passed != expected_pass:
            raise ValueError("D2 decision pass state is inconsistent")
        if self.eligible_for_source_only_ppo != self.passed:
            raise ValueError("D2 source-only PPO eligibility must match pass state")


def residual_d2_promotion_decision(
    metrics: Sequence[D2SourceMetric],
    *,
    source_gates_passed: bool,
    artifact_audits_passed: bool,
) -> D2PromotionDecision:
    """Aggregate the exact three-seed matrix and fail on any regression."""

    if source_gates_passed not in (True, False) or not isinstance(
        source_gates_passed,
        bool,
    ):
        raise ValueError("D2 source gate state must be boolean")
    if artifact_audits_passed not in (True, False) or not isinstance(
        artifact_audits_passed,
        bool,
    ):
        raise ValueError("D2 artifact audit state must be boolean")
    try:
        cells = tuple(metrics)
    except TypeError as error:
        raise TypeError("D2 promotion metrics must be a sequence") from error
    if any(not isinstance(cell, D2SourceMetric) for cell in cells):
        raise TypeError("D2 promotion requires typed source metrics")
    expected = {
        (seed, family)
        for seed in D2_POLICY_SEEDS
        for family in D2_SOURCE_FAMILIES
    }
    observed = {(cell.seed, cell.family) for cell in cells}
    if len(cells) != len(expected) or observed != expected:
        raise ValueError(
            "D2 promotion requires exactly one cell for every locked "
            "policy-seed and source-family pair"
        )

    non_regression = all(cell.non_regression for cell in cells)
    mean_asr_gain = sum(cell.asr_gain for cell in cells) / len(cells)
    mean_query_auc_gain = sum(cell.query_auc_gain for cell in cells) / len(
        cells
    )
    family_mean_asr_gains = tuple(
        sum(
            cell.asr_gain
            for cell in cells
            if cell.family == family
        )
        / len(D2_POLICY_SEEDS)
        for family in D2_SOURCE_FAMILIES
    )
    family_mean_query_auc_gains = tuple(
        sum(
            cell.query_auc_gain
            for cell in cells
            if cell.family == family
        )
        / len(D2_POLICY_SEEDS)
        for family in D2_SOURCE_FAMILIES
    )
    worst_family_mean_asr_gain = min(family_mean_asr_gains)
    worst_family_mean_query_auc_gain = min(family_mean_query_auc_gains)
    mean_gain_gate = (
        mean_asr_gain >= D2_MIN_MEAN_ASR_GAIN - 1e-9
        and mean_query_auc_gain >= D2_MIN_MEAN_QUERY_AUC_GAIN - 1e-9
    )
    worst_family_gate = (
        worst_family_mean_asr_gain > 0.0
        and worst_family_mean_query_auc_gain > 0.0
    )
    passed = (
        source_gates_passed
        and artifact_audits_passed
        and non_regression
        and mean_gain_gate
        and worst_family_gate
    )
    return D2PromotionDecision(
        passed=passed,
        all_seed_family_non_regression=non_regression,
        source_gates_passed=source_gates_passed,
        artifact_audits_passed=artifact_audits_passed,
        eligible_for_source_only_ppo=passed,
        mean_asr_gain=mean_asr_gain,
        mean_query_auc_gain=mean_query_auc_gain,
        worst_family_mean_asr_gain=worst_family_mean_asr_gain,
        worst_family_mean_query_auc_gain=worst_family_mean_query_auc_gain,
        mean_gain_gate_passed=mean_gain_gate,
        worst_family_mean_gain_gate_passed=worst_family_gate,
    )


__all__ = (
    "D2_BC_EPOCHS",
    "D2_COMPETENCE_IMAGES",
    "D2_EVALUATION_IMAGES",
    "D2_GROUPDRO_ETA",
    "D2_GROUPDRO_TRAIN_IMAGES",
    "D2_HELDOUT_FAMILY",
    "D2_MIN_MEAN_ASR_GAIN",
    "D2_MIN_MEAN_QUERY_AUC_GAIN",
    "D2_POLICY_SEEDS",
    "D2_SOURCE_FAMILIES",
    "D2_SOURCE_FOLD_SEED",
    "D2_THRESHOLD_IMAGES",
    "D2_VISITED_POLICY_TRAIN_IMAGES",
    "D2_VISITED_SOURCE_VALIDATION_IMAGES",
    "D2FamilyThresholdMetrics",
    "D2PromotionDecision",
    "D2SourceMetric",
    "D2SourceRole",
    "D2SourceRoles",
    "D2ThresholdCandidate",
    "D2ThresholdSelection",
    "ResidualD2Request",
    "allocate_d2_roles",
    "residual_d2_promotion_decision",
    "select_family_safe_threshold",
    "validate_d2_role_exclusions",
)
