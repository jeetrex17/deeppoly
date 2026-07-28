"""Compact image and perturbation features for patch-action policies.

Feature vectors use deterministic block ordering. Within every block values are
row-major by patch, then channel-minor (R, G, B).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from .config import AttackConfig


PatchFeatureMode = Literal["means", "statistics"]
_FEATURE_BLOCKS: dict[PatchFeatureMode, int] = {
    "means": 2,
    "statistics": 7,
}


def patch_image_feature_dimension(
    *,
    grid_size: int,
    mode: PatchFeatureMode,
) -> int:
    """Return the exact RGB patch-feature dimension for ``mode``."""
    if (
        not isinstance(grid_size, int)
        or isinstance(grid_size, bool)
        or grid_size < 1
    ):
        raise ValueError("grid_size must be a positive integer")
    if not isinstance(mode, str) or mode not in _FEATURE_BLOCKS:
        raise ValueError("patch feature mode must be 'means' or 'statistics'")
    return _FEATURE_BLOCKS[mode] * 3 * grid_size * grid_size


def _validate_images(
    original: torch.Tensor,
    current: torch.Tensor,
    grid_size: int,
) -> tuple[int, int]:
    if not isinstance(original, torch.Tensor) or not isinstance(current, torch.Tensor):
        raise ValueError("original and current images must be tensors")
    if original.shape != current.shape or original.ndim != 3:
        raise ValueError("original and current images must share [C, H, W] shape")
    if not original.is_floating_point() or not current.is_floating_point():
        raise ValueError("image features require floating-point tensors")
    if (
        not isinstance(grid_size, int)
        or isinstance(grid_size, bool)
        or grid_size < 1
    ):
        raise ValueError("grid_size must be a positive integer")
    channels, height, width = original.shape
    if (
        channels != 3
        or height < grid_size
        or width < grid_size
        or height % grid_size
        or width % grid_size
    ):
        raise ValueError("image shape must be RGB and divisible by grid_size")
    if not bool(torch.isfinite(original).all()) or not bool(torch.isfinite(current).all()):
        raise ValueError("image features require finite tensors")
    return height // grid_size, width // grid_size


def _patches(
    image: torch.Tensor,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    return (
        image.unfold(1, patch_height, patch_height)
        .unfold(2, patch_width, patch_width)
        .permute(1, 2, 0, 3, 4)
    )


def _patch_means(
    image: torch.Tensor,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    return _patches(image, patch_height, patch_width).mean(
        dim=(-1, -2),
    ).flatten()


def _patch_edge_magnitude(patches: torch.Tensor) -> torch.Tensor:
    patch_height, patch_width = patches.shape[-2:]
    edge_count = (
        patch_height * max(0, patch_width - 1)
        + max(0, patch_height - 1) * patch_width
    )
    if edge_count == 0:
        return torch.zeros(
            patches.shape[:-2],
            dtype=patches.dtype,
            device=patches.device,
        )
    horizontal = (
        patches[..., :, 1:] - patches[..., :, :-1]
    ).abs().sum(dim=(-1, -2))
    vertical = (
        patches[..., 1:, :] - patches[..., :-1, :]
    ).abs().sum(dim=(-1, -2))
    return (horizontal + vertical) / edge_count


def _validated_epsilon(epsilon: float | None) -> float:
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not math.isfinite(float(epsilon))
        or not 0 < float(epsilon) <= 1
    ):
        raise ValueError(
            "statistics patch features require epsilon in (0, 1]",
        )
    return float(epsilon)


def _validate_attack_region(
    original: torch.Tensor,
    current: torch.Tensor,
    epsilon: float,
) -> None:
    tolerance = max(1e-6, epsilon * 1e-5)
    if (
        float(original.min()) < -tolerance
        or float(original.max()) > 1.0 + tolerance
        or float(current.min()) < -tolerance
        or float(current.max()) > 1.0 + tolerance
    ):
        raise ValueError("statistics patch features require images in [0, 1]")
    if float((current - original).abs().max()) > epsilon + tolerance:
        raise ValueError(
            "current image exceeds the configured L-infinity attack region",
        )


def _attack_headroom(
    original: torch.Tensor,
    current: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_bound = (original - epsilon).clamp(0.0, 1.0)
    upper_bound = (original + epsilon).clamp(0.0, 1.0)
    scale = 2.0 * epsilon
    positive = ((upper_bound - current) / scale).clamp(0.0, 1.0)
    negative = ((current - lower_bound) / scale).clamp(0.0, 1.0)
    return positive, negative


def _statistics_patch_features(
    original: torch.Tensor,
    current: torch.Tensor,
    patch_height: int,
    patch_width: int,
    epsilon: float,
) -> np.ndarray:
    original_float = original.detach().to(dtype=torch.float32)
    current_float = current.detach().to(dtype=torch.float32)
    _validate_attack_region(original_float, current_float, epsilon)
    perturbation = current_float - original_float
    original_patches = _patches(
        original_float,
        patch_height,
        patch_width,
    )
    perturbation_patches = _patches(
        perturbation,
        patch_height,
        patch_width,
    )
    positive_headroom, negative_headroom = _attack_headroom(
        original_float,
        current_float,
        epsilon,
    )
    blocks = (
        _patch_means(original, patch_height, patch_width).to(torch.float32),
        _patch_means(
            current - original,
            patch_height,
            patch_width,
        ).to(torch.float32),
        original_patches.std(dim=(-1, -2), unbiased=False).flatten(),
        _patch_edge_magnitude(original_patches).flatten(),
        (
            perturbation_patches.abs().mean(dim=(-1, -2)) / epsilon
        ).clamp(0.0, 1.0).flatten(),
        _patch_means(positive_headroom, patch_height, patch_width),
        _patch_means(negative_headroom, patch_height, patch_width),
    )
    return torch.cat(blocks).detach().cpu().to(torch.float32).numpy()


def patch_image_features(
    original: torch.Tensor,
    current: torch.Tensor,
    *,
    grid_size: int,
    mode: PatchFeatureMode = "means",
    epsilon: float | None = None,
) -> np.ndarray:
    """Return deterministic patch features without modifying either image.

    ``means`` is the backward-compatible vector:

    1. original patch/channel mean
    2. signed perturbation patch/channel mean

    ``statistics`` retains that prefix and adds:

    3. original population standard deviation
    4. original mean absolute internal-edge magnitude
    5. mean absolute perturbation, normalized by ``epsilon``
    6. positive attack-region headroom, normalized by ``2 * epsilon``
    7. negative attack-region headroom, normalized by ``2 * epsilon``

    Headroom reflects both the L-infinity region around ``original`` and the
    valid pixel interval. Statistical mode therefore requires ``epsilon``.
    """
    patch_height, patch_width = _validate_images(
        original,
        current,
        grid_size,
    )
    expected_dimension = patch_image_feature_dimension(
        grid_size=grid_size,
        mode=mode,
    )
    if mode == "means":
        original_means = _patch_means(
            original,
            patch_height,
            patch_width,
        )
        perturbation_means = _patch_means(
            current - original,
            patch_height,
            patch_width,
        )
        return (
            torch.cat((original_means, perturbation_means))
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )

    numeric_epsilon = _validated_epsilon(epsilon)
    result = _statistics_patch_features(
        original,
        current,
        patch_height,
        patch_width,
        numeric_epsilon,
    )
    if result.shape != (expected_dimension,) or not np.isfinite(result).all():
        raise RuntimeError("patch feature construction violated its output contract")
    return result


def configured_patch_image_features(
    original: torch.Tensor,
    current: torch.Tensor,
    config: AttackConfig,
) -> np.ndarray | None:
    """Build the configured image block, or ``None`` when it is disabled."""
    if not config.image_patch_features:
        return None
    return patch_image_features(
        original,
        current,
        grid_size=config.grid_size,
        mode=config.image_patch_feature_mode,
        epsilon=config.epsilon,
    )
