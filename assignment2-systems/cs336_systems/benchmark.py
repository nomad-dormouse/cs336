from __future__ import annotations

import argparse
import timeit
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


# Model size configurations: (d_model, d_ff, num_layers, num_heads)
MODEL_CONFIGS = {
    "s": (768, 3072, 12, 12),
    "m": (1024, 4096, 24, 16),
    "l": (1280, 5120, 36, 20),
    "xl": (1600, 6400, 48, 25),
    "2.7": (2560, 10240, 32, 32),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Transformer model forward and backward passes")
    
    # Model parameters
    parser.add_argument("--size", type=str, choices=["s", "m", "l", "xl", "2.7"], default="s",
                       help="Model size: s (small), m (medium), l (large), xl (extra large), or 2.7 (2.7B). Default: s")
    parser.add_argument("--context", type=int, default=256, help="Context length. Default: 256")
    
    # Benchmarking parameters
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to measure")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warm-up steps")
    parser.add_argument("--mode", type=str, choices=["f", "f_b"], default="f_b", 
                       help="Whether to benchmark forward only or forward+backward")
    
    return parser.parse_args()


def benchmark(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    warmup_steps: int,
    mode: Literal["f", "f_b"],
) -> tuple[float, float]:
    # Warm-up steps
    for _ in range(warmup_steps):
        output = model(input)
        if mode == 'f_b':
            loss = cross_entropy(output, target)
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()

    # Actual benchmarking
    benchmark_results = {}
    time_forward = []
    time_backward = []
    time_total = []

    for i in range(steps):
        start_time_forward = timeit.default_timer()
        output = model(input)
        torch.cuda.synchronize()
        end_time_forward = timeit.default_timer()
        time_forward.append(end_time_forward - start_time_forward)

        if mode == 'f_b':
            start_time_backward = timeit.default_timer()
            loss = cross_entropy(output, target)
            loss.backward()
            model.zero_grad()
            torch.cuda.synchronize()
            end_time_backward = timeit.default_timer()
            time_backward.append(end_time_backward - start_time_backward)
            time_total.append(time_forward[i] + time_backward[i])

    forward_time = np.array(time_forward)
    benchmark_results["forward"] = {
        "avg_time": forward_time.mean(),
        "std_time": forward_time.std(),
        "total_time": forward_time.sum(),
    }

    if mode == 'f_b':
        backward_time = np.array(time_backward)
        benchmark_results["backward"] = {
            "avg_time": backward_time.mean(),
            "std_time": backward_time.std(),
            "total_time": backward_time.sum(),
        }
        total_time = np.array(time_total)
        benchmark_results["total"] = {
            "avg_time": total_time.mean(),
            "std_time": total_time.std(),
            "total_time": total_time.sum(),
        }
    return benchmark_results


def print_benchmark_results(benchmark_results: dict):
    print(f"Benchmark Results:")
    print(f"  Steps: {benchmark_results['steps']}")
    print(f"  Warm-up steps: {benchmark_results['warmup_steps']}")
    print(f"  Forward:")
    print(f"    Average time per step: {benchmark_results['forward']['avg_time']*1000:.4f} ms")
    print(f"    Standard deviation: {benchmark_results['forward']['std_time']*1000:.4f} ms")
    print(f"    Total time: {benchmark_results['forward']['total_time']:.4f} s")
    if benchmark_results['mode'] == 'f_b':
        print(f"  Backward:")   
        print(f"    Average time per step: {benchmark_results['backward']['avg_time']*1000:.4f} ms")
        print(f"    Standard deviation: {benchmark_results['backward']['std_time']*1000:.4f} ms")
        print(f"    Total time: {benchmark_results['backward']['total_time']:.4f} s")
        print(f"  Total:")
        print(f"    Average time per step: {benchmark_results['total']['avg_time']*1000:.4f} ms")
        print(f"    Standard deviation: {benchmark_results['total']['std_time']*1000:.4f} ms")
        print(f"    Total time: {benchmark_results['total']['total_time']:.4f} s")


def main():
    # Set up arguments
    args = parse_args()
    d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS[args.size]
    vocab = 10000
    batch = 4

    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available")

    # Create model
    model = BasicsTransformerLM(
        vocab_size=vocab,
        context_length=args.context,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
    ).to(device)
    model.train()

    # Create a random dataset
    dataset = np.random.randint(0, vocab, size=(10000,), dtype=np.int64)
    
    # Get random batch
    input, target = get_batch(
        dataset=dataset,
        batch_size=batch,
        context_length=args.context,
        device=str(device),
    )

    # Run benchmark
    benchmark_results = benchmark(
        model=model,
        input=input,
        target=target,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        mode=args.mode,
    )

    # Print results
    print_benchmark_results(benchmark_results)


if __name__ == "__main__":
    main()
