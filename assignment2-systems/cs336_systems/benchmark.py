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
    parser.add_argument("--warmup-steps", type=int, default=3, help="Number of warm-up steps")
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
    times = []
    
    for _ in range(steps):
        start_time = timeit.default_timer()
        
        output = model(input)
        if mode == 'f_b':
            loss = cross_entropy(output, target)
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
        
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    total_time = sum(times)
    avg_time = total_time / steps
    
    return avg_time, total_time


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
    avg_time, total_time = benchmark(
        model=model,
        input=input,
        target=target,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        mode=args.mode,
    )

    # Print results
    print(f'Benchmark Results:')
    print(f'  Mode: {args.mode}')
    print(f'  Steps: {args.steps}')
    print(f'  Warm-up steps: {args.warmup_steps}')
    print(f'  Average time per step: {avg_time*1000:.4f} ms')
    print(f'  Total time: {total_time:.4f} s')


if __name__ == '__main__':
    main()

