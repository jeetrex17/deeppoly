"""Validated configuration for CIFAR victim and policy experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .config import AttackConfig


@dataclass(frozen=True)
class MacPilotConfig:
    schema_version: int
    name: str
    research_valid: bool
    dataset: str
    device: str
    download: bool
    data_root: str
    output_dir: str
    seed: int
    victim_train_images: int
    policy_train_images: int
    source_validation_images: int
    outer_test_images: int
    victim_epochs: int
    policy_episodes: int
    policy_update_block: int
    policy_learning_rate: float
    policy_entropy_weight: float
    policy_update_epochs: int
    query_budget: int
    grid_size: int
    epsilon: float
    step_size: float
    batch_size: int
    num_workers: int
    hidden_dim: int
    victim_learning_rate: float
    target_family: str = "transformer"
    source_instances_per_family: int = 1
    victim_profile: str = "pilot"
    reward_mode: str = "legacy"
    margin_reward_scale: float = 1.0
    terminal_success_bonus: float = 10.0
    query_penalty: float = 0.05
    rollback_on_non_improvement: bool = False
    action_history_features: bool = False
    image_patch_features: bool = False
    behavior_cloning_teacher: str = "best_of_k"
    behavior_cloning_episodes: int = 0
    behavior_cloning_validation_episodes: int = 0
    behavior_cloning_epochs: int = 0
    behavior_cloning_batch_size: int = 256
    behavior_cloning_candidates: int = 8
    behavior_cloning_steps: int = 12
    train_ablation_policies: bool = False
    source_holdout_instances_per_family: int = 0
    target_instances_per_family: int = 1
    victim_validation_images: int = 0
    source_evaluation_images: int = 0
    classical_cnn_min_accuracy: float = 0.60
    modern_cnn_min_accuracy: float = 0.50
    transformer_min_accuracy: float = 0.40
    minimum_source_asr_gain: float = 0.05
    minimum_source_auc_gain: float = 0.02
    source_entropy_min: float = 0.10
    source_entropy_max: float = 0.95
    query_trace_samples_per_method: int = -1
    split_seed: int | None = None
    victim_seed: int | None = None
    policy_actor_mode: str = "flat"
    image_patch_feature_mode: str = "means"
    behavior_cloning_soft_temperature: float | None = None
    policy_evaluation_temperature: float = 1.0

    def __post_init__(self) -> None:
        counts = (
            self.victim_train_images,
            self.policy_train_images,
            self.source_validation_images,
            self.outer_test_images,
        )
        if self.schema_version != 1 or self.dataset != "CIFAR-10":
            raise ValueError("CIFAR experiment requires schema 1 and CIFAR-10")
        if self.research_valid is not False:
            raise ValueError("research_valid remains false until evidence gates pass")
        if self.device not in {"auto", "cpu", "mps", "cuda"} or not isinstance(
            self.download,
            bool,
        ):
            raise ValueError("invalid device or download flag")
        if any(count <= 0 or count % 10 for count in counts):
            raise ValueError("image counts must be positive multiples of ten")
        if sum(counts[:3]) > 50_000 or self.outer_test_images > 1_000:
            raise ValueError("CIFAR split exceeds the available data budget")
        if not 1 <= self.victim_epochs <= 100 or not 1 <= self.policy_episodes <= 20_000:
            raise ValueError("CIFAR training exceeds its bounded run budget")
        if not 1 <= self.policy_update_block <= self.policy_episodes:
            raise ValueError("policy update block must fit within the episode budget")
        if (
            not 2 <= self.query_budget <= 100
            or not 1 <= self.grid_size <= 16
        ):
            raise ValueError("invalid attack budget")
        if not 0 < self.step_size <= self.epsilon <= 1:
            raise ValueError("invalid perturbation budget")
        if (
            not 1 <= self.batch_size <= 2_048
            or not 0 <= self.num_workers <= 16
            or not 1 <= self.hidden_dim <= 1_024
        ):
            raise ValueError("invalid runtime dimensions")
        if self.victim_learning_rate <= 0 or self.policy_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.policy_entropy_weight < 0 or not 1 <= self.policy_update_epochs <= 10:
            raise ValueError("invalid PPO update configuration")
        if self.target_family not in {
            "classical_cnn",
            "modern_cnn",
            "transformer",
        }:
            raise ValueError("target_family must come from the CIFAR victim registry")
        if not 1 <= self.source_instances_per_family <= 5:
            raise ValueError("source_instances_per_family must be between one and five")
        if (
            not isinstance(self.source_holdout_instances_per_family, int)
            or isinstance(self.source_holdout_instances_per_family, bool)
            or not 0 <= self.source_holdout_instances_per_family <= 3
        ):
            raise ValueError("source holdout instances must be between zero and three")
        if (
            not isinstance(self.target_instances_per_family, int)
            or isinstance(self.target_instances_per_family, bool)
            or not 1 <= self.target_instances_per_family <= 5
        ):
            raise ValueError("target instances must be between one and five")
        if (
            not isinstance(self.source_evaluation_images, int)
            or isinstance(self.source_evaluation_images, bool)
            or self.source_evaluation_images < 0
            or self.source_evaluation_images > self.source_validation_images
            or (
                self.source_evaluation_images > 0
                and self.source_evaluation_images % 10 != 0
            )
        ):
            raise ValueError(
                "source_evaluation_images must be zero or a bounded multiple of ten"
            )
        if (
            not isinstance(self.victim_validation_images, int)
            or isinstance(self.victim_validation_images, bool)
            or self.victim_validation_images < 0
            or (
                self.victim_validation_images > 0
                and self.victim_validation_images % 10
            )
            or self.victim_validation_images
            > self.source_validation_images
        ):
            raise ValueError(
                "victim_validation_images must be zero or a bounded multiple of ten"
            )
        self._validate_behavior_cloning()
        for label, value in (
            ("rollback_on_non_improvement", self.rollback_on_non_improvement),
            ("action_history_features", self.action_history_features),
            ("image_patch_features", self.image_patch_features),
            ("train_ablation_policies", self.train_ablation_policies),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be boolean")
        if self.behavior_cloning_teacher not in {"best_of_k", "gradient"}:
            raise ValueError(
                "behavior_cloning_teacher must be 'best_of_k' or 'gradient'"
            )
        self._validate_thresholds()
        if (
            not isinstance(self.query_trace_samples_per_method, int)
            or isinstance(self.query_trace_samples_per_method, bool)
            or not -1 <= self.query_trace_samples_per_method <= 100
        ):
            raise ValueError(
                "query trace sample limit must be -1 or at most 100"
            )
        for label, value in (
            ("split_seed", self.split_seed),
            ("victim_seed", self.victim_seed),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{label} must be a non-negative integer or null")
        if self.victim_profile not in {"pilot", "research"}:
            raise ValueError("victim_profile must be 'pilot' or 'research'")
        if self.policy_actor_mode not in {"flat", "action_conditioned"}:
            raise ValueError(
                "policy_actor_mode must be 'flat' or 'action_conditioned'"
            )
        if self.image_patch_feature_mode not in {"means", "statistics"}:
            raise ValueError(
                "image_patch_feature_mode must be 'means' or 'statistics'"
            )
        if self.behavior_cloning_soft_temperature is not None and (
            isinstance(self.behavior_cloning_soft_temperature, bool)
            or not isinstance(
                self.behavior_cloning_soft_temperature,
                (int, float),
            )
            or not math.isfinite(
                float(self.behavior_cloning_soft_temperature)
            )
            or self.behavior_cloning_soft_temperature <= 0
        ):
            raise ValueError(
                "behavior_cloning_soft_temperature must be positive and finite"
            )
        if (
            self.behavior_cloning_soft_temperature is not None
            and self.behavior_cloning_teacher != "gradient"
        ):
            raise ValueError(
                "soft behavior cloning requires the gradient teacher"
            )
        if (
            isinstance(self.policy_evaluation_temperature, bool)
            or not isinstance(
                self.policy_evaluation_temperature,
                (int, float),
            )
            or not math.isfinite(self.policy_evaluation_temperature)
            or not 0.05 <= self.policy_evaluation_temperature <= 5.0
        ):
            raise ValueError(
                "policy_evaluation_temperature must be in [0.05, 5.0]"
            )
        self.attack_config()

    def _validate_behavior_cloning(self) -> None:
        controls = (
            self.behavior_cloning_episodes,
            self.behavior_cloning_validation_episodes,
            self.behavior_cloning_epochs,
            self.behavior_cloning_batch_size,
            self.behavior_cloning_candidates,
            self.behavior_cloning_steps,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in controls
        ):
            raise ValueError("behavior-cloning controls must be integers")
        if not 0 <= self.behavior_cloning_episodes <= 10_000:
            raise ValueError("behavior_cloning_episodes must be between zero and 10000")
        if not 0 <= self.behavior_cloning_validation_episodes <= 2_000:
            raise ValueError(
                "behavior_cloning_validation_episodes must be between zero and 2000"
            )
        if self.behavior_cloning_episodes == 0 and (
            self.behavior_cloning_epochs != 0
            or self.behavior_cloning_validation_episodes != 0
        ):
            raise ValueError("disabled behavior cloning cannot have epochs or validation")
        if self.behavior_cloning_episodes > 0 and (
            not 1 <= self.behavior_cloning_epochs <= 100
            or self.behavior_cloning_validation_episodes < 1
        ):
            raise ValueError("enabled behavior cloning requires epochs and validation")
        if not 1 <= self.behavior_cloning_batch_size <= 4_096:
            raise ValueError(
                "behavior-cloning batch size must be between one and 4096"
            )
        if self.behavior_cloning_episodes > 0 and (
            not 2 <= self.behavior_cloning_candidates <= 32
            or not 1 <= self.behavior_cloning_steps <= self.query_budget - 1
        ):
            raise ValueError("invalid behavior-cloning candidates or steps")

    def _validate_thresholds(self) -> None:
        accuracy_thresholds = (
            self.classical_cnn_min_accuracy,
            self.modern_cnn_min_accuracy,
            self.transformer_min_accuracy,
        )
        if any(
            not math.isfinite(value) or not 0 < value <= 1
            for value in accuracy_thresholds
        ):
            raise ValueError("victim accuracy thresholds must be in (0, 1]")
        source_gains = (
            self.minimum_source_asr_gain,
            self.minimum_source_auc_gain,
        )
        if any(
            not math.isfinite(value) or not 0 < value <= 1
            for value in source_gains
        ):
            raise ValueError("source practical gains must be in (0, 1]")
        if not 0 <= self.source_entropy_min < self.source_entropy_max <= 1:
            raise ValueError("source entropy bounds are invalid")

    def attack_config(self) -> AttackConfig:
        return AttackConfig(
            epsilon=self.epsilon,
            step_size=self.step_size,
            grid_size=self.grid_size,
            max_queries=self.query_budget,
            reward_mode=self.reward_mode,
            margin_reward_scale=self.margin_reward_scale,
            terminal_success_bonus=self.terminal_success_bonus,
            query_penalty=self.query_penalty,
            rollback_on_non_improvement=self.rollback_on_non_improvement,
            action_history_features=self.action_history_features,
            image_patch_features=self.image_patch_features,
            image_patch_feature_mode=self.image_patch_feature_mode,
        )

    @classmethod
    def from_json(cls, path: Path) -> "MacPilotConfig":
        return cls(**json.loads(path.read_text()))

    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
