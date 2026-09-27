"""Small-to-large sparse-network benchmark with per-scale failure reporting."""

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from artificial_brain.brain import Brain
from artificial_brain.device import resolve_device


SIZES = [(256, 5000), (1000, 50000), (10000, 500000)]


def benchmark(device: str, sizes, steps: int, warmup: int) -> None:
    context = resolve_device(device)
    print("selected_device={}".format(context.device))
    for hidden, edges in sizes:
        try:
            brain = Brain(hidden_neurons=hidden, initial_synapses=edges, device=str(context.device), seed=105)
            inputs = torch.zeros(brain.input_count, device=brain.device, dtype=brain.dtype)
            for _ in range(warmup):
                brain.step(inputs)
            if brain.device.type == "mps":
                torch.mps.synchronize()
            started = time.perf_counter()
            for _ in range(steps):
                brain.step(inputs)
            if brain.device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            memory = "unavailable"
            if brain.device.type == "cuda":
                memory = "allocated_bytes={}".format(torch.cuda.max_memory_allocated(brain.device))
            elif brain.device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
                memory = "allocated_bytes={}".format(torch.mps.current_allocated_memory())
            print("device={} neurons={} synapses={} steps={} wall_seconds={:.6f} steps_per_second={:.2f} memory={}".format(
                brain.device, brain.num_neurons, edges, steps, elapsed, steps / elapsed, memory))
        except (RuntimeError, MemoryError) as exc:
            print("SKIPPED device={} hidden={} synapses={} reason={}".format(context.device, hidden, edges, exc))
            if context.device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--quick", action="store_true", help="Run only the smallest network")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    benchmark(args.device, SIZES[:1] if args.quick else SIZES, args.steps, args.warmup)
