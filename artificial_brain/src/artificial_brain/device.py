"""Device selection shared by the simulation components."""

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class DeviceContext:
    device: torch.device
    dtype: torch.dtype = torch.float32


def resolve_device(device: Optional[Union[str, torch.device]] = None,
                   dtype: torch.dtype = torch.float32) -> DeviceContext:
    """Resolve an explicit device or choose CUDA, MPS, then CPU."""
    if dtype != torch.float32:
        raise ValueError("Phase 01 supports torch.float32 only")
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            chosen = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            chosen = torch.device("mps")
        else:
            chosen = torch.device("cpu")
        return DeviceContext(chosen, dtype)
    chosen = torch.device(device)
    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if chosen.type not in available:
        raise ValueError("device must be one of cpu, mps, or cuda")
    if not available[chosen.type]:
        raise RuntimeError("Requested device {!r} is unavailable; choose cpu or use device='auto'.".format(chosen.type))
    return DeviceContext(chosen, dtype)
