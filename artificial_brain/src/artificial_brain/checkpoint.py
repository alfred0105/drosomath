"""Portable checkpoint helpers."""

from pathlib import Path
from typing import Optional, Union

import torch

from .brain import Brain


def save_checkpoint(brain: Brain, path: Union[str, Path]) -> None:
    torch.save(brain.state_dict(), str(path))


def load_checkpoint(brain: Brain, path: Union[str, Path],
                    map_location: Optional[Union[str, torch.device]] = "cpu") -> Brain:
    try:
        state = torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only support.
        state = torch.load(str(path), map_location=map_location)
    brain.load_state_dict(state)
    return brain
