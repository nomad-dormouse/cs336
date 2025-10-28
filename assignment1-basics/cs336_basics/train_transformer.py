# 5 Training loop

# Problem (data_loading): Implement data loading (2 points)
# Problem (checkpointing): Implement model checkpointing (1 point)
# Problem (training_together): Put it together (4 points)


from dotenv import load_dotenv
import argparse
from typing import Union, BinaryIO, IO
from jaxtyping import Int
import time
from pathlib import Path
import numpy as np
import math
import torch
from torch import Tensor, nn, optim
import wandb
import os
import psutil
from tqdm import tqdm

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimiser import (
    AdamW,
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
)


load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    
    # Weights & Biases project name
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1", help="Weights & Biases project name")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda, mps)")

    # Data
    parser.add_argument("--dataset", type=str, default="TS", help="Name of the dataset to train on (TS or OWT)")
    
    # Model
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=None, help="Feed-forward dimension")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--val_batch_size", type=int, default=256, help="Validation batch size")
    parser.add_argument("--max_iters", type=int, default=1000, help="Maximum number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Maximum learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000, help="Number of cosine annealing iterations")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Checkpointing and logging
    parser.add_argument("--eval_and_log_interval", type=int, default=10, help="Evaluate on validation batch and log metrics every N iterations")
    parser.add_argument("--checkpoint_interval", type=int, default=200, help="Save checkpoint every N iterations")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--test_mode", type=int, default=0, help="Test mode: overfit to a single batch (0 = off, 1 = on)")
    
    return parser.parse_args()


def get_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def calculate_training_parameters(
    args: argparse.Namespace,
    precision_in_bytes: int = 4,
) -> int:
    params = 2 * args.num_layers * (8 * args.d_model**2 + args.d_model) + args.d_model * (args.vocab_size + 1)
    print(f"\n{args.d_model} dimension / {args.num_layers} layers: {params:,} params")

    activations = args.batch_size * (args.num_layers * (2 * args.num_heads * args.context_length**2 + 20 * args.d_model * args.context_length) + args.context_length * (args.d_model + args.vocab_size + 1))
    memory = (4 * params + activations) * precision_in_bytes / (1024**3)
    print(f"{args.context_length} context / {args.batch_size} batch: {memory:.3f} GB")

    tokens = args.max_iters * args.batch_size * args.context_length
    print(f"{args.max_iters} iters: {tokens:,} tokens\n")


def get_data_paths(dataset: str) -> tuple[str, str]:
    if dataset == "TS":
        train_data = "results/tokeniser/tokenised_texts/TinyStoriesV2-GPT4-train_tokenised.npy"
        val_data = "results/tokeniser/tokenised_texts/TinyStoriesV2-GPT4-valid_tokenised.npy"
    elif dataset == "OWT":
        train_data = "results/tokeniser/tokenised_texts/owt_train_tokenised.npy"
        val_data = "results/tokeniser/tokenised_texts/owt_valid_tokenised.npy"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported datasets: TS, OWT")
    
    return train_data, val_data


def get_batch(
    device: str,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    # Each sequence needs context_length tokens,
    # so we can start at any index i, where (i + context_length) < len(data)
    max_start_idx = len(data) - context_length - 1
    
    # Randomly sample batch_size starting indices
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # Get input and target sequences from start indices
    input_tensor, target_tensor = get_input_and_target(
        device=device,
        data=data,
        start_indices=start_indices,
        context_length=context_length,
    )
    
    return input_tensor, target_tensor


def get_input_and_target(
    device: str,
    data: np.ndarray,
    start_indices: list[int],
    context_length: int,
) -> tuple[Int[Tensor, "start_indices_len context_length"], Int[Tensor, "start_indices_len context_length"]]:
    # Get input and target sequences from start indices
    input_sequences = []
    target_sequences = []
    for start_idx in start_indices:
        input_seq = data[start_idx:start_idx + context_length]
        target_seq = data[start_idx + 1:start_idx + context_length + 1]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    
    # Convert to tensors and move to device
    input_tensor = torch.tensor(np.array(input_sequences), dtype=torch.long, device=device)
    target_tensor = torch.tensor(np.array(target_sequences), dtype=torch.long, device=device)

    return input_tensor, target_tensor
    

def train_step(
    model: nn.Module,
    optimiser: AdamW,
    input_tensor: Int[Tensor, "batch_size context_length"],
    target_tensor: Int[Tensor, "batch_size context_length"],
    grad_clip: float,
) -> tuple[float, float]:
    # Forward pass
    logits = model(input_tensor)
    loss = cross_entropy(logits, target_tensor)
    
    # Backward pass
    optimiser.zero_grad()
    loss.backward()
    
    # Gradient clipping (returns gradient norm before clipping)
    grad_norm = gradient_clipping(list(model.parameters()), grad_clip)
    
    # Optimizer step
    optimiser.step()
    
    return logits, loss.item(), grad_norm


def evaluate_model(
    model: nn.Module,
    input_tensor: Int[Tensor, "batch_size context_length"],
    target_tensor: Int[Tensor, "batch_size context_length"],
) -> tuple[float, float, float, float, float]:
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        logits = model(input_tensor)
        loss = cross_entropy(logits, target_tensor)
        
        # Accuracy
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == target_tensor).float().mean()
        
        # Perplexity
        perplexity = loss.exp()

        # Weight norm (L2 norm of all trainable parameters)
        weight_norm = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad).sqrt()
        
        # Memory usage (RSS in MB)
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
    model.train()

    return loss.item(), accuracy.item(), perplexity.item(), weight_norm.item(), memory_mb


def save_checkpoint(
    model: nn.Module,
    optimiser: optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimiser: optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    
    # Return the iteration number
    return checkpoint['iteration']


def training_loop(
    device: str,
    model: nn.Module,
    optimiser: AdamW,
    train_data: np.ndarray,
    val_input_tensor: Int[Tensor, "args.val_batch_size args.context_length"],
    val_target_tensor: Int[Tensor, "args.val_batch_size args.context_length"],
    run_name: str,
    args: argparse.Namespace,
    start_iteration: int = 0,
) -> None:
    model.train()
    start_time = time.time()

    if args.test_mode == 1:
        print("TEST MODE: Model will be trained on a single batch\n")
        train_input_tensor, train_target_tensor = get_batch(
            device=device,
            data=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
        )
    
    for iteration in tqdm(range(start_iteration, args.max_iters)):
        # Get learning rate
        lr = get_lr_cosine_schedule(
            iteration,
            args.learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        
        # Update learning rate
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr
        
        if args.test_mode == 0:
            # Get batch
            train_input_tensor, train_target_tensor = get_batch(
                device=device,
                data=train_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
            )

        # Training step
        train_logits, train_loss, grad_norm = train_step(
            model,
            optimiser,
            train_input_tensor,
            train_target_tensor,
            args.grad_clip,
        )
        
        # Evaluation and logging
        if (iteration + 1) % args.eval_and_log_interval == 0 and iteration > 0:
            elapsed_time = time.time() - start_time

            # Training metrics
            train_preds = train_logits.argmax(dim=-1)
            train_accuracy = (train_preds == train_target_tensor).float().mean().item()

            train_perplexity = math.exp(train_loss)
            
            # Validation metrics
            val_loss, val_accuracy, val_perplexity, weight_norm, memory_mb = evaluate_model(
                model,
                val_input_tensor,
                val_target_tensor,
            )
            
            wandb.log({
                "iteration": iteration,
                "elapsed_time": elapsed_time,
                "learning_rate": lr,
                "gradient_norm": grad_norm,
                "memory_mb": memory_mb,
                "weight_norm": weight_norm,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_perplexity": train_perplexity,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_perplexity": val_perplexity,
            })
            print(f"Iter {iteration:6d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}\n")
        
        # Checkpointing
        checkpoint_dir = Path(f"results/models/checkpoints/{run_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if (iteration + 1) % args.checkpoint_interval == 0 and iteration > 0 and iteration < args.max_iters - 1:
            checkpoint_path = checkpoint_dir / f"iter_{iteration}.pt"
            save_checkpoint(model, optimiser, iteration, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}\n")

    # Final checkpoint - save to trained directory
    trained_dir = Path("results/models/trained")
    trained_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = trained_dir / f"{run_name}.pt"
    save_checkpoint(model, optimiser, iteration, final_checkpoint_path)
    print(f"\nTraining completed! Final checkpoint saved: {final_checkpoint_path}\n")

    wandb.finish()


def train_transformer(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    print("\nCreating model...")
    if args.d_ff is None:
        args.d_ff = 4 * args.d_model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created model has {num_params:,} parameters")
    
    print("Compiling model with torch.compile...")
    if str(device) == "mps":
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model, mode="max-autotune")
    
    print("\nCreating optimiser...")
    optimiser = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    
    print("\nLoading data...")
    train_data_path, val_data_path = get_data_paths(args.dataset)
    train_data = np.load(train_data_path, mmap_mode='r')
    print(f"Loaded {train_data.shape[0]:,} training tokens from {train_data_path}")
    val_data = np.load(val_data_path)
    val_input_tensor, val_target_tensor = get_batch(
        device=device,
        data=val_data,
        batch_size=args.val_batch_size,
        context_length=args.context_length,
    )
    print(f"Loaded {val_input_tensor.shape[0]:,} validation input and target sequences of {args.context_length} tokens")

    start_iteration = 0
    if args.resume_from:
        start_iteration = load_checkpoint(args.resume_from, model, optimiser)
        print(f"\nResuming from checkpoint at iteration {start_iteration}: {args.resume_from}")

    print("\nInitialising Weights and Biases for training logging...")
    training_run_name = (
        f"v{args.vocab_size}"
        f"-c{args.context_length}"
        f"-d{args.d_model}"
        f"-f{args.d_ff}"
        f"-l{args.num_layers}"
        f"-h{args.num_heads}"
        f"-b{args.batch_size}"
        f"-r{args.learning_rate}"
        f"-i{args.max_iters}"
        f"-{args.dataset}-{str(device)}"
    )
    if args.test_mode == 1:
        training_run_name += "-test"
    config = vars(args).copy()
    config.update({
        "device": device,
        "torch_version": torch.__version__,
        "model_parameters": num_params,
    })
    wandb.init(
        project=args.wandb_project,
        name=training_run_name,
        config=config,
        dir=os.getenv("WANDB_DIR"),
    )
    
    print("\nStarting training loop...\n")
    training_loop(
        device=device,
        model=model,
        optimiser=optimiser,
        train_data=train_data,
        val_input_tensor=val_input_tensor,
        val_target_tensor=val_target_tensor,
        run_name=training_run_name,
        start_iteration=start_iteration,
        args=args,
    )


if __name__ == "__main__":
    args = parse_args()
    calculate_training_parameters(args)
    train_transformer(args)
