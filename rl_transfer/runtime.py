from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    device: torch.device

    @property
    def accelerated(self) -> bool:
        return self.device.type in {"mps", "cuda"}

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "requested": self.requested,
            "resolved": self.device.type,
            "accelerated": self.accelerated,
        }


def resolve_device(requested: str = "auto") -> DeviceSelection:
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, mps, cuda")
    if requested == "cpu":
        return DeviceSelection(requested, torch.device("cpu"))
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return DeviceSelection(requested, torch.device("mps"))
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return DeviceSelection(requested, torch.device("cuda"))
    if torch.cuda.is_available():
        return DeviceSelection(requested, torch.device("cuda"))
    if torch.backends.mps.is_available():
        return DeviceSelection(requested, torch.device("mps"))
    return DeviceSelection(requested, torch.device("cpu"))
