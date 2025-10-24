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
import torch
from torch import Tensor, nn, optim
import wandb
import os

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimiser import (
    AdamW,
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
)


load_dotenv()


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    # Each sequence needs context_length tokens, so we can start at any index i, where (i + context_length) < len(data)
    max_start_idx = len(data) - context_length - 1
    
    # Randomly sample batch_size starting indices
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda, mps)")
    
    # Model configuration
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=None, help="Feed-forward dimension (default: 4 * d_model)")

    # Data configuration
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
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
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Save checkpoint every N iterations")
    parser.add_argument("--log_interval", type=int, default=10, help="Log metrics every N iterations")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate on validation set every N iterations")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Weights and Biases
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights and Biases logging")
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


def load_data_memmap(data_path: str) -> np.ndarray:
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, mmap_mode='r')  # Memory-mapped read-only
    print(f"Data shape: {data.shape}, dtype: {data.dtype}")
    return data


def create_model(args) -> nn.Module:
    if args.d_ff is None:
        args.d_ff = 4 * args.d_model
    
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )
    
    return model


def create_optimiser(model: nn.Module, args) -> AdamW:
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    return optimizer


def evaluate_model(
    model: nn.Module,
    val_data: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_eval_batches = 10  # Evaluate on 10 batches
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            # Get a batch of validation data
            input_seq, target_seq = get_batch(val_data, args.batch_size, args.context_length, str(device))
            
            # Forward pass
            logits = model(input_seq)
            loss = cross_entropy(logits, target_seq)
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_eval_batches


def train_step(
    model: nn.Module,
    optimizer: AdamW,
    train_data: np.ndarray,
    args,
    device: torch.device,
) -> float:
    # Get batch
    input_seq, target_seq = get_batch(train_data, args.batch_size, args.context_length, str(device))
    
    # Forward pass
    logits = model(input_seq)
    loss = cross_entropy(logits, target_seq)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    gradient_clipping(list(model.parameters()), args.grad_clip)
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()


def training_loop(
    model: nn.Module,
    optimiser: AdamW,
    train_data: np.ndarray,
    val_data: np.ndarray,
    args,
    device: torch.device,
    start_iteration: int = 0,
) -> None:
    print("Starting training loop...")
    model.train()
    start_time = time.time()
    
    for iteration in range(start_iteration, args.max_iters):
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
        
        # Training step
        loss = train_step(model, optimiser, train_data, args, device)
        
        # Logging
        if iteration % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {iteration:6d} | Loss: {loss:.4f} | LR: {lr:.2e} | Time: {elapsed_time:.1f}s")
            
            if not args.no_wandb:
                wandb.log({
                    "iteration": iteration,
                    "train_loss": loss,
                    "learning_rate": lr,
                    "elapsed_time": elapsed_time,
                })
        
        # Evaluation
        if iteration % args.eval_interval == 0 and iteration > 0:
            val_loss = evaluate_model(model, val_data, args, device)
            print(f"Validation loss: {val_loss:.4f}\n")
            
            if not args.no_wandb:
                wandb.log({
                    "iteration": iteration,
                    "val_loss": val_loss,
                })
        
        # Checkpointing
        checkpoint_dir = Path("results/checkpoints")
        if iteration % args.checkpoint_interval == 0 and iteration > 0:
            checkpoint_path = checkpoint_dir / f"cp_{iteration}.pt"
            save_checkpoint(model, optimiser, iteration, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}\n")
    
    # Final checkpoint
    final_checkpoint_path = checkpoint_dir / f"cp_{iteration}_final.pt"
    save_checkpoint(model, optimiser, args.max_iters, final_checkpoint_path)
    print(f"\nTraining completed! Final checkpoint saved: {final_checkpoint_path}\n")


def train_transformer(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if not args.no_wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            name=args.wandb_run_name,
            config=vars(args),
        )
    
    print("Creating model...")
    model = create_model(args)
    model = model.to(device)
    
    # # Compile model for faster inference
    # model = torch.compile(model, mode="max-autotune")
    # print("Model compiled with torch.compile")
    
    print("Creating optimiser...")
    optimiser = create_optimiser(model, args)
    
    print("Loading data...")
    train_data = load_data_memmap(args.train_data)
    val_data = load_data_memmap(args.val_data)
    
    start_iteration = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iteration = load_checkpoint(args.resume_from, model, optimiser)
        print(f"Resumed from iteration {start_iteration}")
    
    training_loop(
        model=model,
        optimiser=optimiser,
        train_data=train_data,
        val_data=val_data,
        args=args,
        device=device,
        start_iteration=start_iteration,
    )
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train_transformer(args)
