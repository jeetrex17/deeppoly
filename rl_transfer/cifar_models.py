from collections.abc import Callable

import torch
from torch import nn


class NormalizedVictim(nn.Module):
    """Keep attacks in raw pixel space while normalizing inside the victim."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model((images - self.mean) / self.std)


class ResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
        )
        self.skip = (
            nn.Identity()
            if inputs == outputs and stride == 1
            else nn.Sequential(
                nn.Conv2d(inputs, outputs, 1, stride=stride, bias=False),
                nn.BatchNorm2d(outputs),
            )
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class CIFARResidualCNN(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 96, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DepthwiseBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(inputs, inputs, 3, stride=stride, padding=1, groups=inputs, bias=False),
            nn.BatchNorm2d(inputs),
            nn.GELU(),
            nn.Conv2d(inputs, outputs, 1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CIFARDepthwiseCNN(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            DepthwiseBlock(32, 48, 1),
            DepthwiseBlock(48, 72, 2),
            DepthwiseBlock(72, 96, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CIFARPatchTransformer(nn.Module):
    def __init__(self, classes: int = 10, width: int = 48) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, width, kernel_size=4, stride=4)
        self.positions = nn.Parameter(torch.zeros(1, 64, width))
        layer = nn.TransformerEncoderLayer(
            width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.normalization = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens + self.positions[:, :tokens.shape[1]])
        return self.head(self.normalization(encoded.mean(dim=1)))


def build_cifar_victims(seed: int) -> dict[str, tuple[str, nn.Module]]:
    builders: tuple[tuple[str, str, Callable[[], nn.Module]], ...] = (
        ("classical_cnn", "cifar_residual_cnn", CIFARResidualCNN),
        ("modern_cnn", "cifar_depthwise_cnn", CIFARDepthwiseCNN),
        ("transformer", "cifar_patch_transformer", CIFARPatchTransformer),
    )
    victims: dict[str, tuple[str, nn.Module]] = {}
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for family, victim_id, builder in builders:
            victims[family] = (victim_id, NormalizedVictim(builder()))
    return victims
