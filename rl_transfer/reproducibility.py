import hashlib
from pathlib import Path
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def state_digest(module: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


module_digest = state_digest


def tree_digest(root: Path) -> str:
    """Hash relative paths and contents for a reproducibility artifact tree."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError("artifact tree root must be a directory")
    hasher = hashlib.sha256()
    files = tuple(
        sorted(
            path
            for path in resolved.rglob("*")
            if path.is_file()
        )
    )
    if not files:
        raise ValueError("artifact tree cannot be empty")
    for path in files:
        hasher.update(
            path.relative_to(resolved).as_posix().encode("utf-8")
        )
        with path.open("rb") as handle:
            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):
                hasher.update(chunk)
    return hasher.hexdigest()
