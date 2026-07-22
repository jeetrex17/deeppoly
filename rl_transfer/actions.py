from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class AttackAction:
    basis: str
    index: int
    channel: int
    sign: int
    step_scale: float = 1.0
    basis_size: int = 1

    def __post_init__(self) -> None:
        if (
            self.basis not in {"patch", "dct"}
            or self.index < 0
            or self.channel < 0
            or self.sign not in {-1, 1}
            or not math.isfinite(self.step_scale)
            or self.step_scale <= 0
            or self.basis_size < 1
        ):
            raise ValueError("invalid attack action")


def patch_catalog(grid_size: int, channels: int = 3, step_scales: tuple[float, ...] = (1.0,)) -> tuple[AttackAction, ...]:
    return tuple(AttackAction("patch", patch, channel, sign, scale, grid_size) for patch in range(grid_size ** 2) for channel in range(channels) for sign in (-1, 1) for scale in step_scales)


def dct_catalog(frequencies: int, channels: int = 3) -> tuple[AttackAction, ...]:
    return tuple(AttackAction("dct", frequency, channel, sign, basis_size=frequencies) for frequency in range(frequencies ** 2) for channel in range(channels) for sign in (-1, 1))


def project_linf(candidate: torch.Tensor, original: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.maximum(torch.minimum(candidate, original + epsilon), original - epsilon).clamp(0, 1)


def apply_action(image: torch.Tensor, original: torch.Tensor, action: AttackAction, epsilon: float, step_size: float, grid_size: int) -> torch.Tensor:
    if image.shape != original.shape or image.ndim != 3:
        raise ValueError("image and original must share [C,H,W] shape")
    if action.channel >= image.shape[0]:
        raise ValueError("action channel is outside the image")
    action_limit = grid_size ** 2 if action.basis == "patch" else action.basis_size ** 2
    if action.index >= action_limit:
        raise ValueError("action index is outside its basis")
    candidate = image.clone()
    magnitude = action.sign * action.step_scale * step_size
    if action.basis == "patch":
        row, col = divmod(action.index, grid_size)
        patch_h, patch_w = image.shape[1] // grid_size, image.shape[2] // grid_size
        candidate[action.channel, row * patch_h:(row + 1) * patch_h, col * patch_w:(col + 1) * patch_w] += magnitude
    else:
        fy, fx = divmod(action.index, action.basis_size)
        y = torch.arange(image.shape[1], dtype=image.dtype, device=image.device)
        x = torch.arange(image.shape[2], dtype=image.dtype, device=image.device)
        basis = torch.cos(math.pi * (2 * y + 1) * fy / (2 * image.shape[1])).unsqueeze(1) * torch.cos(math.pi * (2 * x + 1) * fx / (2 * image.shape[2])).unsqueeze(0)
        candidate[action.channel] += magnitude * basis
    return project_linf(candidate, original, epsilon)
