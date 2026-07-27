"""Compact image and perturbation features for patch-action policies."""

from __future__ import annotations

import numpy as np
import torch


def patch_image_features(
    original: torch.Tensor,
    current: torch.Tensor,
    *,
    grid_size: int,
) -> np.ndarray:
    if original.shape != current.shape or original.ndim != 3:
        raise ValueError("original and current images must share [C, H, W] shape")
    if (
        not isinstance(grid_size, int)
        or isinstance(grid_size, bool)
        or grid_size < 1
    ):
        raise ValueError("grid_size must be a positive integer")
    channels, height, width = original.shape
    if channels != 3 or height % grid_size or width % grid_size:
        raise ValueError("image shape must be RGB and divisible by grid_size")
    if not torch.isfinite(original).all() or not torch.isfinite(current).all():
        raise ValueError("image features require finite tensors")
    patch_height = height // grid_size
    patch_width = width // grid_size

    def means(image: torch.Tensor) -> torch.Tensor:
        patches = image.unfold(1, patch_height, patch_height).unfold(
            2,
            patch_width,
            patch_width,
        )
        return patches.mean(dim=(-1, -2)).permute(1, 2, 0).flatten()

    original_means = means(original)
    perturbation_means = means(current - original)
    return (
        torch.cat((original_means, perturbation_means))
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()
    )
