import pytest
import torch

from artificial_brain.device import resolve_device


def test_cpu_override_and_float32():
    context = resolve_device("cpu")
    assert context.device == torch.device("cpu")
    assert context.dtype == torch.float32


def test_auto_selects_available_device():
    context = resolve_device()
    expected = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    assert context.device.type == expected


def test_unavailable_device_has_helpful_error():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            resolve_device("cuda")
