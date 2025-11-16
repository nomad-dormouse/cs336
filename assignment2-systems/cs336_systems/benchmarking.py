import argparse
import sys
import timeit
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

# Import adamw_accounting from assignment1-basics
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "assignment1-basics"))
from cs336_basics.optimiser import adamw_accounting


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
    
    parser.add_argument("--mode", type=str, choices=["f", "b", "f_b"], default="f_b", 
                       help="Whether to benchmark forward only, backward only, or forward+backward")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to measure")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--sizes", type=str, nargs='+', choices=list(MODEL_CONFIGS.keys()), default=["s"],
                       help=f"Model sizes to run benchmarking sweep on. Choices: {list(MODEL_CONFIGS.keys())}. Default: s")
    parser.add_argument("--contexts", type=int, nargs='+', default=[256], 
                       help="Context lengths to run benchmarking sweep on. Default: 256")
    
    return parser.parse_args()


def benchmark(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    warmup_steps: int,
    mode: str = "f_b",
) -> dict:
    # Initialise variables
    benchmark_results = {}
    times_forward = []
    times_backward = []
    times_forward_and_backward = []

    # Warm-up steps
    for _ in tqdm(range(warmup_steps), desc="Warm-up steps"):
        output = model(input)
        if mode == 'f_b':
            loss = cross_entropy(output, target)
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()

    # Actual benchmarking
    for _ in tqdm(range(steps), desc="Benchmarking steps"):
        if mode == 'f' or mode == 'f_b':
            start_time_forward = timeit.default_timer()
            output = model(input)
            torch.cuda.synchronize()
            end_time_forward = timeit.default_timer()
            time_forward = end_time_forward - start_time_forward
            times_forward.append(time_forward)

        if mode == 'b' or mode == 'f_b':
            start_time_backward = timeit.default_timer()
            loss = cross_entropy(output, target)
            loss.backward()
            model.zero_grad()
            torch.cuda.synchronize()
            end_time_backward = timeit.default_timer()
            time_backward = end_time_backward - start_time_backward
            times_backward.append(time_backward)
        
        if mode == 'f_b':
            times_forward_and_backward.append(time_forward + time_backward)

    if mode == 'f' or mode == 'f_b':
        forward_times = np.array(times_forward)
        avg = forward_times.mean() * 1000
        std = forward_times.std() * 1000
        benchmark_results["forward_ms"] = f"{avg:.2f} ± {std:.2f}"

    if mode == 'b' or mode == 'f_b':
        backward_times = np.array(times_backward)
        avg = backward_times.mean() * 1000
        std = backward_times.std() * 1000
        benchmark_results["backward_ms"] = f"{avg:.2f} ± {std:.2f}"
        
    if mode == 'f_b':
        forward_and_backward_times = np.array(times_forward_and_backward)
        avg = forward_and_backward_times.mean() * 1000
        std = forward_and_backward_times.std() * 1000
        benchmark_results["forward_and_backward_ms"] = f"{avg:.2f} ± {std:.2f}"

    return benchmark_results


def run_benchmarking(
    size: str = "s",
    context: int = 256,
    steps: int = 10,
    warmup_steps: int = 5,
    vocab_size: int = 10000,
    batch_size: int = 4,
    d_model: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    d_ff: int = 3072,
    mode: str = "f_b",
) -> dict:
    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available")

    # Create model
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
    ).to(device)
    model.train()

    # Create a random dataset
    dataset = np.random.randint(0, vocab_size, size=(10000,), dtype=np.int64)
    
    # Get random batch
    input, target = get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context,
        device=str(device),
    )

    # Run benchmark
    benchmark_results = benchmark(
        model=model,
        input=input,
        target=target,
        steps=steps,
        warmup_steps=warmup_steps,
        mode=mode,
    )

    # Add metadata to results
    benchmark_results["size"] = size
    benchmark_results["d_model"] = d_model
    benchmark_results["d_ff"] = d_ff
    benchmark_results["num_layers"] = num_layers
    benchmark_results["num_heads"] = num_heads
    benchmark_results["context"] = context
    benchmark_results["vocab_size"] = vocab_size
    benchmark_results["batch_size"] = batch_size
    benchmark_results["mode"] = mode
    benchmark_results["steps"] = steps
    benchmark_results["warmup_steps"] = warmup_steps
    
    return benchmark_results


def run_benchmarking_experiment(
    steps: int = 10,
    warmup_steps: int = 5,
    mode: str = "f_b",
    sizes: list[str] = ["s"],
    contexts: list[int] = [256],
) -> None:
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = f"warmup_{warmup_steps}_steps_{steps}_mode_{mode}_sizes_{'_'.join(sizes)}_contexts_{'_'.join(str(c) for c in contexts)}"
    csv_file = results_dir / f"benchmarking_{filename}.csv"

    total_jobs = len(sizes) * len(contexts)
    print(f"Running {total_jobs} benchmarking jobs sequentially on single GPU...")
    
    # Run jobs sequentially on single GPU
    results = []
    job_num = 0
    for size in sizes:
        for context in contexts:
            torch.cuda.empty_cache()
            job_num += 1
            d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS[size]
            adamw_accounting(
                model_name=size,
                batch_size=4,
                vocab_size=10000,
                context_length=context,
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
            )
            print(f"\n[{job_num}/{total_jobs}] Running benchmarking for size={size} and context={context}...")
            benchmark_results = run_benchmarking(
                size=size,
                context=context,
                steps=steps,
                warmup_steps=warmup_steps,
                vocab_size=10000,
                batch_size=4,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                mode=mode,
            )
            results.append(benchmark_results)
    
    # Combine results into DataFrame
    cols = ["size", "d_model", "d_ff", "num_layers", "num_heads", "context", "vocab_size", "batch_size", "steps", "warmup_steps", "mode", "forward_and_backward_ms", "forward_ms", "backward_ms"]
    df = pd.DataFrame(results)[cols]
    
    # Save to CSV and print DataFrame
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to: {csv_file}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    run_benchmarking_experiment(
        steps=args.steps,
        warmup_steps=args.warmup,
        mode=args.mode,
        sizes=args.sizes,
        contexts=args.contexts,
    )
