import argparse
import timeit
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import submitit
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
    
    parser.add_argument("--mode", type=str, choices=["f", "b", "f_b"], default="f_b", 
                       help="Whether to benchmark forward only, backward only, or forward+backward")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to measure")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps")
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
        benchmark_results["avg_time_forward_ms"] = forward_times.mean() * 1000
        benchmark_results["std_time_forward_ms"] = forward_times.std() * 1000

    if mode == 'b' or mode == 'f_b':
        backward_times = np.array(times_backward)
        benchmark_results["avg_time_backward_ms"] = backward_times.mean() * 1000
        benchmark_results["std_time_backward_ms"] = backward_times.std() * 1000
        
    if mode == 'f_b':
        forward_and_backward_times = np.array(times_forward_and_backward)
        benchmark_results["avg_time_forward_and_backward_ms"] = forward_and_backward_times.mean() * 1000
        benchmark_results["std_time_forward_and_backward_ms"] = forward_and_backward_times.std() * 1000

    return benchmark_results


def run_benchmarking(
    size: str = "s",
    context: int = 256,
    steps: int = 10,
    warmup_steps: int = 5,
    vocab_size: int = 10000,
    batch_size: int = 4,
    mode: str = "f_b",
) -> dict:
    # Get model configuration
    d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS[size]

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
    benchmark_results["context"] = context
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
    logs_dir = Path("./results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"sizes_{'_'.join(sizes)}_contexts_{'_'.join(str(c) for c in contexts)}"
    csv_file = results_dir / f"benchmarking_{filename}.csv"

    executor = submitit.AutoExecutor(folder=logs_dir)
    executor.update_parameters(
        nodes=1,
        gpus_per_node=1,
        name=f"logs_{filename}",
    )

    jobs = []
    for size in sizes:
        for context in contexts:
            def job_fn(s=size, c=context):
                torch.cuda.empty_cache()
                print(f"Running benchmarking for size={s} and context={c}...")
                benchmark_results = run_benchmarking(s, c, steps, warmup_steps, mode=mode)
                return benchmark_results
            
            jobs.append(executor.submit(job_fn))

    print(f"Submitted {len(jobs)} jobs: {[j.job_id for j in jobs]}")
    print(f"Monitor with: squeue -u $USER")
    print(f"Logs in: {logs_dir}")
    print(f"Waiting for jobs to complete...")
    
    # Wait for all jobs to complete and collect results
    results = [job.result() for job in jobs]
    
    # Combine results into DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")
    
    # Print DataFrame
    print("\nBENCHMARKING RESULTS:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    run_benchmarking_experiment(
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        mode=args.mode,
        sizes=args.sizes,
        contexts=args.contexts,
    )
