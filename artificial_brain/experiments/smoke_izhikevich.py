"""Short firing-pattern and numerical-stability smoke run."""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from artificial_brain.device import resolve_device
from artificial_brain.neurons import IzhikevichPopulation


def run(device: str) -> None:
    context = resolve_device(device)
    types = torch.tensor([0, 1, 2])
    population = IzhikevichPopulation(3, device=str(context.device), neuron_types=types)
    current = torch.full((3,), 10.0, device=context.device, dtype=context.dtype)
    counts = torch.zeros(3, dtype=torch.long, device=context.device)
    for _ in range(1000):
        counts += population.step(current).long()
    if not torch.isfinite(population.v).all() or not torch.isfinite(population.u).all():
        raise RuntimeError("non-finite neuron state")
    print("device={} dtype={} steps=1000 spike_counts={} finite=true".format(
        context.device, context.dtype, counts.cpu().tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    try:
        run(args.device)
    except (RuntimeError, ValueError) as exc:
        print("SMOKE FAILED: {}".format(exc), file=sys.stderr)
        raise
