from collections.abc import Callable, Mapping, Sequence
import hashlib

import torch
from torch import nn


VictimEntry = tuple[str, nn.Module]
VictimEnsemble = dict[str, tuple[VictimEntry, ...]]


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
    """Small CIFAR ResNet with configurable stage capacity.

    The defaults intentionally retain the original pilot architecture and state
    dictionary layout.  The research victim profile opts into wider, deeper
    stages without breaking callers that construct this class directly.
    """

    def __init__(
        self,
        classes: int = 10,
        widths: Sequence[int] = (32, 64, 96),
        blocks_per_stage: Sequence[int] = (1, 1, 1),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = tuple(widths)
        blocks_per_stage = tuple(blocks_per_stage)
        if not widths or len(widths) != len(blocks_per_stage):
            raise ValueError("widths and blocks_per_stage must have the same non-zero length")
        if classes < 1 or any(width < 1 for width in widths) or any(blocks < 1 for blocks in blocks_per_stage):
            raise ValueError("classes, widths, and stage block counts must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = [
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        ]
        inputs = widths[0]
        for stage, (outputs, block_count) in enumerate(zip(widths, blocks_per_stage)):
            for block in range(block_count):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(ResidualBlock(inputs, outputs, stride=stride))
                inputs = outputs
        layers.extend((nn.AdaptiveAvgPool2d(1), nn.Flatten()))
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(widths[-1], classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DepthwiseBlock(nn.Module):
    def __init__(
        self,
        inputs: int,
        outputs: int,
        stride: int,
        expansion: int = 1,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if inputs < 1 or outputs < 1 or stride < 1 or expansion < 1:
            raise ValueError("channel counts, stride, and expansion must be positive")
        hidden = inputs * expansion
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.extend(
                (
                    nn.Conv2d(inputs, hidden, 1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                )
            )
        layers.extend(
            (
                nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
                nn.Conv2d(hidden, outputs, 1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.GELU(),
            )
        )
        self.network = nn.Sequential(*layers)
        self.use_residual = residual and inputs == outputs and stride == 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.network(inputs)
        return inputs + outputs if self.use_residual else outputs


class CIFARDepthwiseCNN(nn.Module):
    """Depthwise CNN; configurable as a compact inverted-residual network."""

    def __init__(
        self,
        classes: int = 10,
        widths: Sequence[int] = (32, 48, 72, 96),
        blocks_per_stage: Sequence[int] = (1, 1, 1),
        expansion: int = 1,
        residual: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = tuple(widths)
        blocks_per_stage = tuple(blocks_per_stage)
        if len(widths) < 2 or len(blocks_per_stage) != len(widths) - 1:
            raise ValueError("blocks_per_stage must describe each post-stem width")
        if classes < 1 or any(width < 1 for width in widths) or any(blocks < 1 for blocks in blocks_per_stage):
            raise ValueError("classes, widths, and stage block counts must be positive")
        if expansion < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("expansion must be positive and dropout must be in [0, 1)")

        layers: list[nn.Module] = [
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.GELU(),
        ]
        inputs = widths[0]
        for stage, (outputs, block_count) in enumerate(zip(widths[1:], blocks_per_stage)):
            for block in range(block_count):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(
                    DepthwiseBlock(
                        inputs,
                        outputs,
                        stride,
                        expansion=expansion,
                        residual=residual,
                    )
                )
                inputs = outputs
        layers.extend((nn.AdaptiveAvgPool2d(1), nn.Flatten()))
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(widths[-1], classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CIFARPatchTransformer(nn.Module):
    """CIFAR patch transformer with legacy mean or stronger CLS pooling."""

    def __init__(
        self,
        classes: int = 10,
        width: int = 48,
        depth: int = 1,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        pooling: str = "mean",
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        if classes < 1 or width < 1 or depth < 1 or heads < 1 or width % heads:
            raise ValueError("classes, width, depth, and compatible attention heads are required")
        if mlp_ratio <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("mlp_ratio must be positive and dropout must be in [0, 1)")
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.pooling = pooling
        self.patch_embed = nn.Conv2d(3, width, kernel_size=4, stride=4)
        token_count = 64 + int(pooling == "cls")
        if pooling == "cls":
            self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.positions = nn.Parameter(torch.zeros(1, token_count, width))
        layer = nn.TransformerEncoderLayer(
            width,
            nhead=heads,
            dim_feedforward=max(1, int(width * mlp_ratio)),
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.normalization = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        if self.pooling == "cls":
            class_token = self.class_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((class_token, tokens), dim=1)
        encoded = self.encoder(tokens + self.positions[:, :tokens.shape[1]])
        pooled = encoded[:, 0] if self.pooling == "cls" else encoded.mean(dim=1)
        return self.head(self.normalization(pooled))


def _victim_builders(profile: str) -> tuple[tuple[str, str, Callable[[], nn.Module]], ...]:
    if profile == "pilot":
        return (
            ("classical_cnn", "cifar_residual_cnn", CIFARResidualCNN),
            ("modern_cnn", "cifar_depthwise_cnn", CIFARDepthwiseCNN),
            ("transformer", "cifar_patch_transformer", CIFARPatchTransformer),
        )
    if profile == "research":
        return (
            (
                "classical_cnn",
                "cifar_residual_cnn_research",
                lambda: CIFARResidualCNN(
                    widths=(64, 128, 256),
                    blocks_per_stage=(2, 2, 2),
                    dropout=0.1,
                ),
            ),
            (
                "modern_cnn",
                "cifar_depthwise_cnn_research",
                lambda: CIFARDepthwiseCNN(
                    widths=(48, 72, 144, 256),
                    blocks_per_stage=(2, 2, 3),
                    expansion=3,
                    residual=True,
                    dropout=0.1,
                ),
            ),
            (
                "transformer",
                "cifar_patch_transformer_research",
                lambda: CIFARPatchTransformer(
                    width=96,
                    depth=3,
                    heads=4,
                    mlp_ratio=4.0,
                    dropout=0.1,
                    pooling="cls",
                    norm_first=True,
                ),
            ),
        )
    raise ValueError("profile must be 'pilot' or 'research'")


def build_cifar_victims(seed: int, *, profile: str = "pilot") -> dict[str, VictimEntry]:
    """Build one victim per family without changing the caller's RNG state.

    ``profile='pilot'`` is the backward-compatible default.  The opt-in
    ``research`` profile has enough capacity for more meaningful CIFAR fits.
    """

    victims: dict[str, VictimEntry] = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for family, victim_id, builder in _victim_builders(profile):
            victims[family] = (victim_id, NormalizedVictim(builder()))
    return victims


def _instance_seed(seed: int, family: str, instance: int) -> int:
    """Derive a stable, process-independent PyTorch seed for one victim."""

    payload = f"cifar-victim-v1:{seed}:{family}:{instance}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def build_cifar_victim_ensemble(
    seed: int,
    instances_per_family: int | Mapping[str, int] = 2,
    *,
    families: Sequence[str] | None = None,
    profile: str = "research",
) -> VictimEnsemble:
    """Build independently seeded victim instances, grouped by family.

    Grouping is retained deliberately: callers can sample a model instance
    inside a family while keeping family-level GroupDRO weights and metrics.
    Instance IDs include the derived initialization seed for exact replay.
    """

    builders = {family: (victim_id, builder) for family, victim_id, builder in _victim_builders(profile)}
    selected = tuple(builders) if families is None else tuple(families)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("families must contain at least one unique family")
    unknown = set(selected) - set(builders)
    if unknown:
        raise ValueError(f"unknown CIFAR victim families: {sorted(unknown)}")
    if isinstance(instances_per_family, int):
        counts = {family: instances_per_family for family in selected}
    else:
        missing = set(selected) - set(instances_per_family)
        extra = set(instances_per_family) - set(selected)
        if missing or extra:
            raise ValueError("instance-count mapping must exactly match selected families")
        counts = {family: instances_per_family[family] for family in selected}
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in counts.values()):
        raise ValueError("every selected family requires at least one integer instance")

    ensembles: VictimEnsemble = {}
    with torch.random.fork_rng(devices=[]):
        for family in selected:
            base_id, builder = builders[family]
            instances: list[VictimEntry] = []
            for instance in range(counts[family]):
                instance_seed = _instance_seed(seed, family, instance)
                torch.manual_seed(instance_seed)
                victim_id = f"{base_id}__instance_{instance}__seed_{instance_seed}"
                instances.append((victim_id, NormalizedVictim(builder())))
            ensembles[family] = tuple(instances)
    return ensembles


def build_cifar_victim_population(
    seed: int,
    instances_per_family: int | Mapping[str, int],
    *,
    families: Sequence[str] | None = None,
    profile: str = "research",
) -> VictimEnsemble:
    """Compatibility name for a family-grouped, multi-instance population."""

    return build_cifar_victim_ensemble(
        seed,
        instances_per_family,
        families=families,
        profile=profile,
    )
