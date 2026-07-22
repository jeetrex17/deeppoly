import torch
from torch import nn


class SmallCNN(nn.Module):
    """Small CIFAR-sized source victim used by the PDF protocol."""

    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, classes))
        nn.init.constant_(self.classifier[-1].bias, 0.0)
        with torch.no_grad():
            self.classifier[-1].bias[0] = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class TargetCNN(nn.Module):
    """A different inductive bias for target-model transfer tests."""

    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, classes),
        )
        nn.init.constant_(self.net[-1].bias, 0.0)
        with torch.no_grad():
            self.net[-1].bias[0] = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class TinyPatchTransformer(nn.Module):
    """Small patch-token transformer used only by the offline pilot."""

    def __init__(self, classes: int = 10, width: int = 32) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, width, kernel_size=4, stride=4)
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=64, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(width, classes)
        nn.init.constant_(self.head.bias, 0.0)
        with torch.no_grad():
            self.head.bias[0] = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        return self.head(self.encoder(tokens).mean(dim=1))


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
