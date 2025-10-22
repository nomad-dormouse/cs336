# 3 Training a Transformer LM

# Problem (cross_entropy): Implement Cross entropy
# Problem (adamw): Implement AdamW (2 points)
# Problem (adamwAccounting): Resource accounting for training with AdamW


from dotenv import load_dotenv
import torch
from torch import Tensor, nn, optim
from jaxtyping import Float, Int
from typing import Iterable, Optional, Callable


# Load environment variables
load_dotenv()


def toy_example(
    num_iterations: int,
    learning_rate: float,
):
    torch.manual_seed(42)
    weights = nn.Parameter(5 * torch.randn((10, 10)))
    opt = torch.optim.SGD([weights], lr=learning_rate)

    for _ in range(num_iterations):
        opt.zero_grad() # Reset the gradients for all learnable parameters
        loss = (weights**2).mean() # Compute a scalar loss value
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients
        opt.step() # Run optimizer step


def adamw_accounting(
    model_name: str,
    batch_size: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int | None = None,
    precision_in_bits: int = 32,
) -> dict:
    """
    Calculate memory and compute requirements for training with AdamW optimizer.
    
    Args:
        batch_size: Number of sequences in a batch
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        num_layers: Number of transformer layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension (defaults to 4 * d_model)
        precision_in_bits: Bits per parameter (32 for float32)
        
    Returns:
        Dictionary with memory breakdown for parameters, activations, gradients, and optimizer state
    """
    if d_ff is None:
        d_ff = 4 * d_model
    
    d_k = d_model // num_heads
    bytes_per_param = precision_in_bits // 8
    
    # 1. PARAMETERS MEMORY
    # Token embeddings: vocab_size × d_model
    embedding_params = vocab_size * d_model
    
    # Per transformer block:
    # - 2 RMSNorm: 2 × d_model
    # - Attention: 4 × d_model² (Q, K, V, output projections)
    # - FFN: 3 × d_model × d_ff (w1, w3, w2)
    block_params = 2 * d_model + 4 * d_model * d_model + 3 * d_model * d_ff
    
    # All transformer blocks
    transformer_params = num_layers * block_params
    
    # Final layer norm + LM head
    final_params = d_model + d_model * vocab_size
    
    total_params = embedding_params + transformer_params + final_params
    parameters_memory = total_params * bytes_per_param
    
    # 2. ACTIVATIONS MEMORY (forward pass)
    # Embeddings: batch_size × context_length × d_model
    embedding_activations = batch_size * context_length * d_model
    
    # Per transformer block activations:
    # - Input to block: batch_size × context_length × d_model
    # - RMSNorm output: batch_size × context_length × d_model
    # - QKV projections: 3 × batch_size × context_length × d_model
    # - QK^T attention: batch_size × num_heads × context_length × context_length
    # - Attention output: batch_size × context_length × d_model
    # - FFN input: batch_size × context_length × d_model
    # - FFN w1 output: batch_size × context_length × d_ff
    # - FFN w2 output: batch_size × context_length × d_model
    # - Block output: batch_size × context_length × d_model
    
    # Attention matrices (largest component)
    attention_activations = batch_size * num_heads * context_length * context_length
    
    # FFN activations
    ffn_activations = batch_size * context_length * d_ff
    
    # Other activations (approximate)
    other_activations = 10 * batch_size * context_length * d_model  # Conservative estimate
    
    per_block_activations = attention_activations + ffn_activations + other_activations
    all_blocks_activations = num_layers * per_block_activations
    
    # Final activations
    final_activations = batch_size * context_length * d_model  # Final RMSNorm
    lm_head_activations = batch_size * context_length * vocab_size  # Logits
    
    total_activations = embedding_activations + all_blocks_activations + final_activations + lm_head_activations
    activations_memory = total_activations * bytes_per_param
    
    # 3. GRADIENTS MEMORY (same size as parameters)
    gradients_memory = parameters_memory
    
    # 4. OPTIMIZER STATE MEMORY (AdamW)
    # AdamW stores: first moment, second moment for each parameter
    # Each moment has same size as the parameter
    optimizer_state_memory = 2 * parameters_memory  # 2 moments per parameter
    
    # 5. TOTAL PEAK MEMORY
    # Peak memory = max(forward_pass, backward_pass)
    # Forward pass: parameters + activations
    # Backward pass: parameters + activations + gradients + optimizer_state
    forward_pass_memory = parameters_memory + activations_memory
    backward_pass_memory = parameters_memory + activations_memory + gradients_memory + optimizer_state_memory
    peak_memory = max(forward_pass_memory, backward_pass_memory)
    
    # 6. COMPUTE REQUIREMENTS (FLOPs)
    # Forward pass FLOPs (from transformer accounting)
    d_k = d_model // num_heads
    
    # Per block FLOPs
    qkv_flops = 3 * 2 * context_length * d_model * d_model
    qk_flops = num_heads * 2 * context_length * context_length * d_k
    av_flops = num_heads * 2 * context_length * context_length * d_k
    output_proj_flops = 2 * context_length * d_model * d_model
    attention_flops = qkv_flops + qk_flops + av_flops + output_proj_flops
    
    ffn_flops = 3 * 2 * context_length * d_model * d_ff
    block_flops = attention_flops + ffn_flops
    
    # Total forward FLOPs
    forward_flops = num_layers * block_flops + 2 * context_length * d_model * vocab_size
    
    # Backward pass typically requires ~2x forward pass FLOPs
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops
    
    results = {
        "model": {
            "model_name": model_name,
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "context_length": context_length,
            "num_layers": num_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "precision_in_bits": precision_in_bits,
        },
        "memory_breakdown": {
            "parameters": {
                "count": total_params,
                "memory_bytes": parameters_memory,
                "memory_gb": parameters_memory / (1024**3),
            },
            "activations": {
                "count": total_activations,
                "memory_bytes": activations_memory,
                "memory_gb": activations_memory / (1024**3),
            },
            "gradients": {
                "count": total_params,
                "memory_bytes": gradients_memory,
                "memory_gb": gradients_memory / (1024**3),
            },
            "optimizer_state": {
                "count": 2 * total_params,  # 2 moments per parameter
                "memory_bytes": optimizer_state_memory,
                "memory_gb": optimizer_state_memory / (1024**3),
            },
            "total": {
                "forward_pass_gb": forward_pass_memory / (1024**3),
                "backward_pass_gb": backward_pass_memory / (1024**3),
                "peak_memory_gb": peak_memory / (1024**3),
            }
        },
        "compute": {
            "forward_flops": forward_flops,
            "backward_flops": backward_flops,
            "total_flops": total_flops,
        },
        "algebraic_expressions": {
            "parameters": f"{total_params} = {embedding_params} + {num_layers} × {block_params} + {final_params}",
            "activations": f"{total_activations} ≈ {batch_size} × {context_length} × ({d_model} + {num_layers} × ({num_heads} × {context_length} + {d_ff})) + {batch_size} × {context_length} × {vocab_size}",
            "gradients": f"{total_params} (same as parameters)",
            "optimizer_state": f"{2 * total_params} (2 moments per parameter)",
            "peak_memory": f"max(parameters + activations, parameters + activations + gradients + optimizer_state)"
        }
    }

    print_accounting(results)


def print_accounting(
    accounting_data: dict
) -> None:
    model = accounting_data["model"]
    memory = accounting_data["memory_breakdown"]
    compute = accounting_data["compute"]
    expressions = accounting_data["algebraic_expressions"]
    
    print(f"\nADAMW TRAINING ACCOUNTING")
    print(f"=" * 50)
    print(f"Model Configuration:")
    print(f"  Batch size: {model['batch_size']:,}")
    print(f"  Vocabulary size: {model['vocab_size']:,}")
    print(f"  Context length: {model['context_length']:,}")
    print(f"  Number of layers: {model['num_layers']}")
    print(f"  Model dimension: {model['d_model']:,}")
    print(f"  Number of heads: {model['num_heads']}")
    print(f"  Feed-forward dimension: {model['d_ff']:,}")
    print(f"  Precision: {model['precision_in_bits']} bits")
    
    print(f"\nMemory Breakdown:")
    print(f"  Parameters: {memory['parameters']['memory_gb']:.3f} GB ({memory['parameters']['count']:,} elements)")
    print(f"  Activations: {memory['activations']['memory_gb']:.3f} GB ({memory['activations']['count']:,} elements)")
    print(f"  Gradients: {memory['gradients']['memory_gb']:.3f} GB ({memory['gradients']['count']:,} elements)")
    print(f"  Optimizer state: {memory['optimizer_state']['memory_gb']:.3f} GB ({memory['optimizer_state']['count']:,} elements)")
    
    print(f"\nPeak Memory Usage:")
    print(f"  Forward pass: {memory['total']['forward_pass_gb']:.3f} GB")
    print(f"  Backward pass: {memory['total']['backward_pass_gb']:.3f} GB")
    print(f"  Peak memory: {memory['total']['peak_memory_gb']:.3f} GB")
    
    print(f"\nCompute Requirements:")
    print(f"  Forward FLOPs: {compute['forward_flops']:.2e}")
    print(f"  Backward FLOPs: {compute['backward_flops']:.2e}")
    print(f"  Total FLOPs: {compute['total_flops']:.2e}")
    
    print(f"\nAlgebraic Expressions:")
    print(f"  Parameters: {expressions['parameters']}")
    print(f"  Activations: {expressions['activations']}")
    print(f"  Gradients: {expressions['gradients']}")
    print(f"  Optimizer state: {expressions['optimizer_state']}")
    print(f"  Peak memory: {expressions['peak_memory']}")


def cross_entropy(
    logits: Float[Tensor, " ... vocab_size"],
    targets: Int[Tensor, " ..."],
) -> Float[Tensor, ""]:
    # Lets simplify the cross-entropy loss formula
    # cross_entropy_loss = -log(p[target_class])
    # = -log(softmax(logits)[target_class])
    # = -log(exp(logits[target_class]) / sum(exp(logits)))
    # = -logits[target_class] + log(sum(exp(logits)))

    # x = x - max(x) + max(x)
    # = x_substr_max + max(x)

    # log(sum(exp(x))) = log(sum(exp(x - max(x) + max(x))))
    # = log(sum(exp(x - max(x)) * exp(max(x))))
    # = log(sum(exp(x - max(x)))) + log(exp(max(x)))
    # = log(sum(exp(x - max(x)))) + max(x)
    # = log(sum(exp(x_substr_max))) + max(x)

    # - x + log(exp(max(x))) = - (x_substr_max + max(x)) + log(sum(exp(x_substr_max))) + max(x)
    # = - x_substr_max + log(sum(exp(x_substr_max)))

    # Now, let's use the trick above to modify our cross-entropy loss formula
    # cross_entropy_loss = -logits[target_class] + log(sum(exp(logits))) =
    # = -logits_substr_max[target_class] + log(sum(exp(logits_substr_max)))

    # Therefore, we can substract max logit value from logits for numerical stability of exp(logits)
    logits_stable = logits - logits.max(dim=-1, keepdim=True)[0] # shape (..., vocab_size)

    target_logits_stable = logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # shape (...)
    log_sum_exp_stable = torch.logsumexp(logits_stable, dim=-1) # shape (...)
    loss = - target_logits_stable + log_sum_exp_stable # shape (...)
    
    # Return average loss across all predictions
    return loss.mean() # shape (...) -> ()


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ):  
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)


    def step(
        self,
        closure: Optional[Callable] = None
    ):
        # Compute the loss if a closure is provided, otherwise set loss to None
        loss = closure() if closure is not None else None
        
        # Loop over parameter groups
        for group in self.param_groups:
            learning_rate = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            # Loop over parameters in the group
            for p in group["params"]:
                # Skip if the gradient is None
                if p.grad is None:
                    continue
                
                # Get the state of the parameter
                state = self.state[p]
                t = state.get("t", 1)
                first_moment = state.get("first_moment", torch.zeros_like(p.data))
                second_moment = state.get("second_moment", torch.zeros_like(p.data))

                # Update the first and second moments estimates
                first_moment = beta1 * first_moment + (1 - beta1) * p.grad.data
                second_moment = beta2 * second_moment + (1 - beta2) * p.grad.data**2
                state["first_moment"], state["second_moment"] = first_moment, second_moment
                                
                # Update the parameter value
                learning_rate_t = learning_rate * (1 - beta2**t)**0.5 / (1 - beta1**t) # Adjusted learning rate for iteration t
                p.data = p.data - learning_rate_t * first_moment / (second_moment**0.5 + eps) # Update before weight decay
                p.data = p.data - learning_rate * weight_decay * p.data # Update after weight decay
                
                # Update the iteration number
                state["t"] = t + 1

        return loss


if __name__ == "__main__":
    # num_iterations = 10
    # learning_rates = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    # for learning_rate in learning_rates:
    #     print(f"\nLearning rate: {learning_rate}")
    #     toy_example(num_iterations, learning_rate)
    
    model_name = "GPT-2_small"
    batch_size = 4,
    vocab_size = 50257
    context_length = 1024
    num_layers = 12
    d_model = 768
    num_heads = 12
    
    args = (batch_size, vocab_size, context_length, num_layers, d_model, num_heads)
    adamw_accounting(*args)
    
    