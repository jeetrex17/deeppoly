"""Victim training and accuracy evaluation for CIFAR studies."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .cifar_config import MacPilotConfig


def classifier_accuracy(
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
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            labels = labels.to(
                device,
                non_blocking=device.type == "cuda",
            )
            predictions = model(
                images.to(device, non_blocking=device.type == "cuda")
            ).argmax(1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
    return correct / total if total else 0.0


def train_classifier(
    model: nn.Module,
    dataset: Dataset,
    indices: Sequence[int],
    config: MacPilotConfig,
    device: torch.device,
    training_seed: int,
    progress: Callable[[str], None],
) -> tuple[dict[str, float], ...]:
    torch.manual_seed(training_seed)
    if device.type == "mps":
        torch.mps.manual_seed(training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)
    generator = torch.Generator().manual_seed(training_seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.victim_learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.victim_epochs,
    )
    augmentation_generator = torch.Generator().manual_seed(
        training_seed + 50_000
    )
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.victim_epochs):
        loss_sum = 0.0
        correct = total = 0
        for images, labels in loader:
            padded = nn.functional.pad(
                images,
                (4, 4, 4, 4),
                mode="reflect",
            )
            offsets = torch.randint(
                0,
                9,
                (images.shape[0], 2),
                generator=augmentation_generator,
            )
            images = torch.stack(
                tuple(
                    padded[index, :, top:top + 32, left:left + 32]
                    for index, (top, left) in enumerate(offsets.tolist())
                )
            )
            flips = (
                torch.rand(
                    images.shape[0],
                    generator=augmentation_generator,
                )
                < 0.5
            )
            images = torch.where(
                flips.view(-1, 1, 1, 1),
                images.flip(-1),
                images,
            )
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            labels = labels.to(
                device,
                non_blocking=device.type == "cuda",
            )
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
            f"loss={history[-1]['loss']:.4f} "
            f"accuracy={history[-1]['accuracy']:.3f}"
        )
    return tuple(history)
