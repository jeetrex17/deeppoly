from dataclasses import asdict, dataclass, replace
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .artifacts import (
    load_model_checkpoint,
    load_recurrent_checkpoint,
    save_model_checkpoint,
    save_recurrent_checkpoint,
    sha256_file,
)
from .baselines import FixedActionPolicy, RandomActionPolicy
from .cifar_models import build_cifar_victims
from .config import AttackConfig
from .models import freeze_model
from .recurrent import PPOConfig, RecurrentAttackPolicy
from .reproducibility import seed_everything
from .research_metrics import AttackOutcome, asr_at_budgets, asr_query_auc
from .research_protocol import run_frozen_episode, train_population_policy
from .results import ResearchResultRow, write_jsonl
from .runtime import resolve_device


@dataclass(frozen=True)
class CIFARSplit:
    victim_fit: tuple[int, ...]
    policy_train: tuple[int, ...]
    source_validation: tuple[int, ...]
    outer_test: tuple[int, ...]
    digest: str


def _class_buckets(labels: Sequence[int]) -> dict[int, list[int]]:
    buckets = {label: [] for label in range(10)}
    for index, label in enumerate(labels):
        if label not in buckets:
            raise ValueError("CIFAR-10 labels must be integers in [0, 9]")
        buckets[label].append(index)
    if any(not indices for indices in buckets.values()):
        raise ValueError("every CIFAR-10 class must be present")
    return buckets


def build_cifar_split(
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    victim_fit_count: int,
    policy_train_count: int,
    source_validation_count: int,
    outer_test_count: int,
    seed: int,
) -> CIFARSplit:
    counts = (victim_fit_count, policy_train_count, source_validation_count, outer_test_count)
    if any(count <= 0 or count % 10 for count in counts):
        raise ValueError("CIFAR split counts must be positive multiples of ten")
    train_buckets = _class_buckets(train_labels)
    test_buckets = _class_buckets(test_labels)
    train_per_class = tuple(count // 10 for count in counts[:3])
    test_per_class = outer_test_count // 10
    if any(len(train_buckets[label]) < sum(train_per_class) for label in range(10)):
        raise ValueError("insufficient train examples for requested stratified split")
    if any(len(test_buckets[label]) < test_per_class for label in range(10)):
        raise ValueError("insufficient test examples for requested stratified split")
    roles = [[], [], []]
    outer: list[int] = []
    for label in range(10):
        train_indices = list(train_buckets[label])
        test_indices = list(test_buckets[label])
        random.Random(seed + label).shuffle(train_indices)
        random.Random(seed + 10_000 + label).shuffle(test_indices)
        start = 0
        for role, size in zip(roles, train_per_class):
            role.extend(train_indices[start:start + size])
            start += size
        outer.extend(test_indices[:test_per_class])
    for offset, role in enumerate(roles):
        random.Random(seed + 20_000 + offset).shuffle(role)
    random.Random(seed + 30_000).shuffle(outer)
    ordered_roles = tuple(tuple(role) for role in roles)
    ordered_outer = tuple(outer)
    encoded = json.dumps((*ordered_roles, ordered_outer), separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return CIFARSplit(*ordered_roles, ordered_outer, digest)


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

    def __post_init__(self) -> None:
        counts = (
            self.victim_train_images,
            self.policy_train_images,
            self.source_validation_images,
            self.outer_test_images,
        )
        if self.schema_version != 1 or self.dataset != "CIFAR-10":
            raise ValueError("Mac pilot requires schema 1 and CIFAR-10")
        if self.research_valid is not False:
            raise ValueError("bounded Mac pilot must not be marked research-valid")
        if self.device not in {"auto", "cpu", "mps"} or not isinstance(self.download, bool):
            raise ValueError("invalid device or download flag")
        if any(count <= 0 or count % 10 for count in counts):
            raise ValueError("image counts must be positive multiples of ten")
        if sum(counts[:3]) > 50_000 or self.outer_test_images > 1_000:
            raise ValueError("Mac pilot split exceeds its bounded data budget")
        if not 1 <= self.victim_epochs <= 10 or not 1 <= self.policy_episodes <= 1_000:
            raise ValueError("Mac pilot training exceeds its bounded run budget")
        if not 1 <= self.policy_update_block <= self.policy_episodes:
            raise ValueError("policy update block must fit within the episode budget")
        if not 2 <= self.query_budget <= 100 or self.grid_size < 1:
            raise ValueError("invalid attack budget")
        if not 0 < self.step_size <= self.epsilon <= 1:
            raise ValueError("invalid perturbation budget")
        if self.batch_size < 1 or not 0 <= self.num_workers <= 2 or self.hidden_dim < 1:
            raise ValueError("invalid runtime dimensions")
        if self.victim_learning_rate <= 0 or self.policy_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.policy_entropy_weight < 0 or not 1 <= self.policy_update_epochs <= 10:
            raise ValueError("invalid PPO update configuration")

    @classmethod
    def from_json(cls, path: Path) -> "MacPilotConfig":
        return cls(**json.loads(path.read_text()))

    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _code_digest() -> str:
    hasher = hashlib.sha256()
    package_root = Path(__file__).parent
    for path in sorted(package_root.glob("*.py")):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _dataset_samples(dataset: Dataset, indices: Sequence[int]) -> tuple[tuple[torch.Tensor, int], ...]:
    return tuple((dataset[index][0].float(), int(dataset[index][1])) for index in indices)


def _classifier_accuracy(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            labels = labels.to(device)
            predictions = model(images.to(device)).argmax(1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / total if total else 0.0


def _train_classifier(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    config: MacPilotConfig,
    device: torch.device,
    progress: Callable[[str], None],
) -> tuple[dict[str, float], ...]:
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.victim_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.victim_epochs)
    augmentation_generator = torch.Generator().manual_seed(config.seed + 50_000)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.victim_epochs):
        loss_sum = 0.0
        correct = total = 0
        for images, labels in loader:
            padded = nn.functional.pad(images, (4, 4, 4, 4), mode="reflect")
            offsets = torch.randint(0, 9, (images.shape[0], 2), generator=augmentation_generator)
            images = torch.stack(
                tuple(
                    padded[index, :, top:top + 32, left:left + 32]
                    for index, (top, left) in enumerate(offsets.tolist())
                )
            )
            flips = torch.rand(images.shape[0], generator=augmentation_generator) < 0.5
            images = torch.where(flips.view(-1, 1, 1, 1), images.flip(-1), images)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.argmax(1) == labels).sum())
            total += labels.numel()
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": loss_sum / max(1, total),
                "accuracy": correct / max(1, total),
            }
        )
        progress(
            f"victim epoch {epoch + 1}/{config.victim_epochs}: "
            f"loss={history[-1]['loss']:.4f} accuracy={history[-1]['accuracy']:.3f}"
        )
    return tuple(history)


def _checkpoint_matches(metadata: dict[str, object], fingerprint: str) -> None:
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("checkpoint fingerprint does not match this pilot run")


def _evaluate_methods(
    policy: RecurrentAttackPolicy,
    target: tuple[str, nn.Module],
    samples: tuple[tuple[torch.Tensor, int], ...],
    indices: Sequence[int],
    attack: AttackConfig,
    seed: int,
    progress: Callable[[str], None],
) -> tuple[list[ResearchResultRow], list[dict[str, object]], dict[str, object]]:
    victim_id, victim = target
    methods = {
        "groupdro_recurrent_ppo": (policy, True),
        "groupdro_recurrent_ppo_stochastic": (policy, False),
        "fixed_action": (FixedActionPolicy(0, attack.action_dim), True),
        "random_action": (RandomActionPolicy(attack.action_dim, seed), False),
    }
    rows: list[ResearchResultRow] = []
    traces: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    budgets = tuple(sorted({0, attack.max_queries, *(value for value in (5, 10, 25) if value < attack.max_queries)}))
    for method_offset, (method, (attack_policy, deterministic)) in enumerate(methods.items()):
        progress(f"evaluating {method} on {len(samples)} held-out images")
        torch.manual_seed(seed + 100_000 + method_offset)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed + 100_000 + method_offset)
        before = attack_policy.persistent_digest()
        outcomes: list[AttackOutcome] = []
        method_rows: list[ResearchResultRow] = []
        for image_index, ((image, label), dataset_index) in enumerate(zip(samples, indices)):
            sample_id = f"cifar10:test:{dataset_index}"
            result = run_frozen_episode(
                attack_policy,
                victim,
                image,
                label,
                sample_id,
                victim_id,
                "transformer",
                attack,
                deterministic=deterministic,
            )
            outcomes.append(AttackOutcome(result.clean_correct, result.query_to_success))
            row = ResearchResultRow(
                sample_id=sample_id,
                victim_id=victim_id,
                victim_family="transformer",
                method=method,
                threat_model="T1",
                seed=seed,
                query_budget=attack.max_queries,
                clean_correct=result.clean_correct,
                success=result.success,
                query_to_success=result.query_to_success,
                total_target_calls=result.total_target_calls,
                linf=result.linf,
                l2=result.l2,
                policy_digest=result.policy_digest_after,
                action_trace=result.actions,
            )
            rows.append(row)
            method_rows.append(row)
            traces.append({"method": method, **result.as_dict()})
        eligible = sum(outcome.clean_correct for outcome in outcomes)
        if eligible:
            curve = asr_at_budgets(outcomes, budgets)
            auc = asr_query_auc(curve)
        else:
            curve, auc = {}, None
        action_counts = Counter(action for row in method_rows for action in row.action_trace)
        action_total = sum(action_counts.values())
        action_entropy = 0.0
        if action_total and attack.action_dim > 1:
            action_entropy = -sum(
                (count / action_total) * math.log(count / action_total)
                for count in action_counts.values()
            ) / math.log(attack.action_dim)
        after = attack_policy.persistent_digest()
        summary[method] = {
            "eligible": eligible,
            "successes": sum(row.success for row in method_rows),
            "asr_at_budgets": curve,
            "asr_query_auc": auc,
            "policy_digest_before": before,
            "policy_digest_after": after,
            "frozen": before == after,
            "deterministic_actions": deterministic,
            "action_histogram": {
                str(action): count for action, count in sorted(action_counts.items())
            },
            "normalized_action_entropy": action_entropy,
        }
    return rows, traces, summary


def run_cifar_pilot_from_datasets(
    config: MacPilotConfig,
    train_dataset: Dataset,
    test_dataset: Dataset,
    resume: bool = True,
    dataset_version: str = "in-memory-fixture",
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    report = progress or (lambda _message: None)
    started = time.monotonic()
    seed_everything(config.seed)
    selection = resolve_device(config.device)
    device = selection.device
    report(f"resolved device: {device.type}")
    split = build_cifar_split(
        train_dataset.targets,
        test_dataset.targets,
        config.victim_train_images,
        config.policy_train_images,
        config.source_validation_images,
        config.outer_test_images,
        config.seed,
    )
    code_digest = _code_digest()
    fingerprint_source = f"{config.digest()}:{split.digest}:{code_digest}"
    fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()
    run_dir = Path(config.output_dir) / fingerprint[:12]
    run_dir.mkdir(parents=True, exist_ok=True)
    report(f"run directory: {run_dir}")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "name": config.name,
        "status": "running",
        "research_valid": False,
        "fingerprint": fingerprint,
        "run_dir": str(run_dir),
        "config": asdict(config),
        "config_digest": config.digest(),
        "split_digest": split.digest,
        "dataset": {"name": config.dataset, "version": dataset_version},
        "device": selection.as_dict(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "git_revision": _git_revision(),
            "code_digest": code_digest,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "determinism": (
                "warn_only; some MPS operators may not have deterministic implementations"
                if device.type == "mps"
                else "deterministic algorithms requested with warn_only"
            ),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    victims = build_cifar_victims(config.seed)
    victim_metrics: dict[str, object] = {}
    for family, (victim_id, model) in victims.items():
        model.to(device)
        checkpoint_path = run_dir / "victims" / f"{victim_id}.pt"
        if resume and checkpoint_path.is_file() and checkpoint_path.with_suffix(".pt.sha256").is_file():
            report(f"loading {family} victim checkpoint")
            metadata = load_model_checkpoint(checkpoint_path, model, device)
            _checkpoint_matches(metadata, fingerprint)
            history = metadata["history"]
            resumed = True
        else:
            report(f"training {family} victim ({victim_id})")
            history = _train_classifier(
                model,
                train_dataset,
                split.victim_fit,
                config,
                device,
                report,
            )
            metadata = {"fingerprint": fingerprint, "family": family, "history": history}
            save_model_checkpoint(checkpoint_path, model, metadata)
            resumed = False
        validation_accuracy = _classifier_accuracy(
            model,
            train_dataset,
            split.source_validation,
            config.batch_size,
            config.num_workers,
            device,
        )
        freeze_model(model)
        report(f"{family} validation accuracy: {validation_accuracy:.3f}")
        victim_metrics[family] = {
            "victim_id": victim_id,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "history": history,
            "source_validation_accuracy": validation_accuracy,
            "resumed": resumed,
        }
    attack = AttackConfig(
        epsilon=config.epsilon,
        step_size=config.step_size,
        grid_size=config.grid_size,
        max_queries=config.query_budget,
    )
    policy_path = run_dir / "policy.pt"
    source_victims = {family: victims[family] for family in ("classical_cnn", "modern_cnn")}
    policy_samples = _dataset_samples(train_dataset, split.policy_train)
    if resume and policy_path.is_file() and policy_path.with_suffix(".pt.sha256").is_file():
        report("loading recurrent policy checkpoint")
        policy, policy_metadata = load_recurrent_checkpoint(policy_path, device)
        _checkpoint_matches(policy_metadata, fingerprint)
        completed_episodes = int(policy_metadata["completed_episodes"])
        training_blocks = list(policy_metadata["training_blocks"])
        policy_resumed = True
    else:
        policy = RecurrentAttackPolicy(
            observation_dim=8,
            action_dim=attack.action_dim,
            hidden_dim=config.hidden_dim,
            seed=config.seed,
            config=PPOConfig(
                learning_rate=config.policy_learning_rate,
                entropy_weight=config.policy_entropy_weight,
                update_epochs=config.policy_update_epochs,
            ),
        ).to(device)
        completed_episodes = 0
        training_blocks: list[dict[str, object]] = []
        policy_resumed = False
    while completed_episodes < config.policy_episodes:
        block_episodes = min(
            config.policy_update_block,
            config.policy_episodes - completed_episodes,
        )
        report(
            f"training recurrent policy episodes {completed_episodes + 1}-"
            f"{completed_episodes + block_episodes}/{config.policy_episodes}"
        )
        block = train_population_policy(
            policy,
            source_victims,
            policy_samples,
            attack,
            episodes=block_episodes,
            seed=config.seed + completed_episodes,
            initial_family_weights=(
                training_blocks[-1]["family_weights"] if training_blocks else None
            ),
        )
        training_blocks.append(block)
        completed_episodes += block_episodes
        policy_metadata = {
            "fingerprint": fingerprint,
            "dataset": config.dataset,
            "split_digest": split.digest,
            "seed": config.seed,
            "completed_episodes": completed_episodes,
            "training_blocks": training_blocks,
        }
        save_recurrent_checkpoint(policy_path, policy, policy_metadata)
    training = {
        "episodes": config.policy_episodes,
        "completed_episodes": completed_episodes,
        "trained_episodes": sum(int(block["trained_episodes"]) for block in training_blocks),
        "source_calls": sum(int(block["source_calls"]) for block in training_blocks),
        "blocks": training_blocks,
        "final_family_weights": training_blocks[-1]["family_weights"],
    }
    if not policy_resumed:
        policy_metadata = {
            "fingerprint": fingerprint,
            "dataset": config.dataset,
            "split_digest": split.digest,
            "seed": config.seed,
            "completed_episodes": config.policy_episodes,
            "training_blocks": training_blocks,
        }
        save_recurrent_checkpoint(policy_path, policy, policy_metadata)
    report(f"trained policy episodes: {training['trained_episodes']}")
    target_samples = _dataset_samples(test_dataset, split.outer_test)
    rows, traces, evaluation = _evaluate_methods(
        policy,
        victims["transformer"],
        target_samples,
        split.outer_test,
        attack,
        config.seed,
        report,
    )
    write_jsonl(run_dir / "results.jsonl", rows)
    (run_dir / "query_traces.jsonl").write_text(
        "".join(json.dumps(trace, sort_keys=True) + "\n" for trace in traces)
    )
    manifest.update(
        {
            "status": "complete",
            "victims": victim_metrics,
            "policy": {
                "checkpoint": str(policy_path),
                "checkpoint_sha256": sha256_file(policy_path),
                "resumed": policy_resumed,
                "training": training,
            },
            "evaluation": evaluation,
            "victim_accuracy_gate": {
                "thresholds": {
                    "classical_cnn": 0.60,
                    "modern_cnn": 0.50,
                    "transformer": 0.40,
                },
                "passed": all(
                    float(victim_metrics[family]["source_validation_accuracy"]) >= threshold
                    for family, threshold in {
                        "classical_cnn": 0.60,
                        "modern_cnn": 0.50,
                        "transformer": 0.40,
                    }.items()
                ),
            },
            "target_test_accuracy": _classifier_accuracy(
                victims["transformer"][1],
                test_dataset,
                split.outer_test,
                config.batch_size,
                config.num_workers,
                device,
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


def run_cifar_pilot(
    config_path: Path,
    resume: bool = True,
    device: str | None = None,
    output_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    config = MacPilotConfig.from_json(config_path)
    if device is not None:
        config = replace(config, device=device)
    if output_dir is not None:
        config = replace(config, output_dir=str(output_dir))
    try:
        import torchvision
        from torchvision.transforms import ToTensor
    except ImportError as error:
        raise RuntimeError("install the vision extra before running the CIFAR pilot") from error
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=True,
        transform=ToTensor(),
        download=config.download,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.data_root,
        train=False,
        transform=ToTensor(),
        download=config.download,
    )
    return run_cifar_pilot_from_datasets(
        config,
        train_dataset,
        test_dataset,
        resume=resume,
        dataset_version=f"torchvision-{torchvision.__version__}",
        progress=progress,
    )
