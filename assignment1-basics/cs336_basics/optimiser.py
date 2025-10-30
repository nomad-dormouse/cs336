# 4 Training a Transformer LM

# Problem (cross_entropy): Implement Cross entropy
# Problem (learning_rate_tuning): Tuning the learning rate (1 point)
# Problem (adamw): Implement AdamW (2 points)
# Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)
# Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup
# Problem (gradient_clipping): Implement gradient clipping (1 point)


from dotenv import load_dotenv, find_dotenv
import torch
from torch import Tensor, nn, optim
from jaxtyping import Float, Int
from typing import Iterable, Optional, Callable, Union, BinaryIO, IO
import json
import math
import numpy as np


load_dotenv(find_dotenv())


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

def adamw_memory(
    batch_size: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    precision_in_bits: int = 32,
) -> dict:

    gb_per_param = precision_in_bits / (8 * 1024**3)
    
    # PARAMETERS
    
    # Parameters: embeddings + transformer blocks + final RMSNorm + LM head

    # Embedding layer
    #       = vocab_size * d_model

    # Transformer blocks: num_layers * (2 RMSNorm + Attention + FFN)
    # - RMSNorm: before attention and before FFN
    #       = d_model
    # - Attention: Q, K, V, Output projections
    #       = 4 * d_model * d_model =
    #       = 4 * d_model**2
    # - FFN: W1, W3, W2
    #       = 3 * d_model * (4 * d_model) =
    #       = 12 * d_model**2
    # - Total transformer blocks parameters
    #       = num_layers * (2 * d_model + 4 * d_model**2 + 12 * d_model**2) =
    #       = 2 * num_layers * (8 * d_model**2 + d_model)

    # Final RMSNorm
    #       = d_model

    # LM head
    #       = d_model * vocab_size

    # Total parameters
    #       = vocab_size * d_model +
    #       + 2 * num_layers * (8 * d_model**2 + d_model) +
    #       + d_model +
    #       + d_model * vocab_size =
    #       = 2 * num_layers * (8 * d_model**2 + d_model) + vocab_size * d_model + d_model =
    #       = 2 * num_layers * (8 * d_model**2 + d_model) + d_model * (vocab_size + 1)
    
    params = 2 * num_layers * (8 * d_model**2 + d_model) + d_model * (vocab_size + 1)
    params_memory_gb = params * gb_per_param

    # GRADIENTS

    # We store the gradients for each parameter
    grads = params
    grads_memory_gb = grads * gb_per_param

    # OPTIMISER STATE

    # We store the first and second moments for each parameter
    opt_states = 2 * params
    opt_states_memory_gb = opt_states * gb_per_param
        
    # ACTIVATIONS

    # Activations: transformer blocks + final RMSNorm + output embeddings + cross-entropy on logits
    
    # Transformer blocks: num_layers * (2 RMSNorm + Attention + FFN)
    # - RMSNorms outputs
    #       = batch_size * context_length * d_model
    # - Attention: QKV + QK^T + softmax + AV + O projection
    #   - Q, K and V projections output
    #        = 3 * batch_size * context_length * d_model
    #   - QK^T output
    #       = batch_size * num_heads * context_length * context_length
    #   - Softmax output
    #       = batch_size * num_heads * context_length * context_length
    #   - AV output of all heads
    #       = batch_size * context_length * d_model
    #   - O projection output
    #       = batch_size * context_length * d_model
    #   - Total attention activations:
    #       = 3 * batch_size * context_length * d_model +
    #       + batch_size * num_heads * context_length * context_length +
    #       + batch_size * num_heads * context_length * context_length +
    #       + batch_size * context_length * d_model +
    #       + batch_size * context_length * d_model =
    #       =  batch_size (2 * num_heads * context_length**2 + 5 * d_model * context_length)
    # - FFN: W1 projection + SiLU + Pairwise multiplication of SiLU and W1 projection + W3 projection
    #    - W1 projection output
    #       = batch_size * context_length * (d_model * 4) =
    #       = batch_size * context_length * d_model * 4
    #    - SiLU output
    #       = batch_size * context_length * (d_model * 4)
    #    - Pairwise multiplication of SiLU and W1 projection
    #       = batch_size * context_length * (d_model * 4)
    #    - W3 projection output
    #       = batch_size * context_length * d_model
    #    - Total FFN activations:
    #       = batch_size * context_length * (d_model * 4) +
    #       + batch_size * context_length * (d_model * 4) +
    #       + batch_size * context_length * (d_model * 4) +
    #       + batch_size * context_length * d_model =
    #       = 13 * batch_size * d_model * context_length
    # - Total transformer blocks activations:
    #       = num_layers * 
    #       * (2 * batch_size * context_length * d_model +
    #       + batch_size (2 * num_heads * context_length**2 + 5 * d_model * context_length) +
    #       + 13 * batch_size * d_model * context_length) =
    #       = batch_size * num_layers * (2 * num_heads * context_length**2 + 20 * d_model * context_length)
    
    # Final RMSNorm
    #       = batch_size * context_length * d_model

    # Output embeddings
    #       = batch_size * context_length * vocab_size

    # Cross-entropy on logits
    #       = batch_size * context_length

    # Total activations
    #       = batch_size * num_layers * (2 * num_heads * context_length**2 + 20 * d_model * context_length) +
    #       + batch_size * context_length * d_model +
    #       + batch_size * context_length * vocab_size +
    #       + batch_size * context_length =
    #       = batch_size * (num_layers * (2 * num_heads * context_length**2 + 20 * d_model * context_length) + context_length * (d_model + vocab_size + 1))

    activations = batch_size * (num_layers * (2 * num_heads * context_length**2 + 20 * d_model * context_length) + context_length * (d_model + vocab_size + 1))
    activations_memory_gb = activations * gb_per_param

    # Overall
    #       = parameters + gradients + optimizer states + activations =
    #       = params + params + 2 params + activations =
    #       = 4 * params + activations =
    #       = 4 * (2 * num_layers * (8 * d_model**2 + d_model) + d_model * (vocab_size + 1)) +
    #       + batch_size * (num_layers * (2 * num_heads * context_length**2 + 20 * d_model * context_length) + context_length * (d_model + vocab_size + 1)) =

    total = 4 * params + activations
    total_memory_gb = total * gb_per_param

    # total = batch_coefficient * batch_size + constant
    # memory_gb = total * gb_per_param = (batch_coefficient * batch_size + constant) * gb_per_param
    # max_batch_size = (available_memory_gb / gb_per_param - constant) / batch_coefficient
    batch_coefficient = num_layers * (2 * num_heads * context_length**2 + 20 * d_model * context_length) + context_length * (d_model + vocab_size + 1)
    constant = 4 * (2 * num_layers * (8 * d_model**2 + d_model) + d_model * (vocab_size + 1))
    max_batch_80gb = (80 / gb_per_param - constant) / batch_coefficient

    return params_memory_gb, grads_memory_gb, opt_states_memory_gb, activations_memory_gb, total_memory_gb, max_batch_80gb


def adamw_compute(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
) -> dict:
        
    # Optimiser step FLOPs = FLOPs for forward pass + FLOPs for backward pass = FLOPs for forward pass + 2 * FLOPs for backward pass = 3 * FLOPs for forward pass
    
    # As most of the FLOPs are used during matrix multiplications, we will account only for matrix multiplications FLOPs
    # Given A ∈ R^(m×n) and B ∈ R^(n×p), the matrix-matrix product AB requires 2mnp FLOPs

    # Total FLOPs for performing one optimisation step =
    #       = transformer blocks FLOPs for one item + output embeddings FLOPs for one item
    
    # Transformer blocks FLOPs for one item: num_layers * (Attention sub-block FLOPs + FFN sub-block FLOPs)
    # - Attention
    #   - Q, K and V projections: multiply X of shape (context_length x d_model) by Q, K and V of shape (d_model x d_model)
    #       = 3 * 2 * context_length * d_model * d_model =
    #       = 6 * context_length * d_model**2
    #   - QK^T calculation: num_heads times multiply Q of shape (context_length x d_model/num_heads) by transposed K of shape (d_model/num_heads x context_length)
    #       = num_heads * 2 * context_length * d_model/num_heads * context_length =
    #       = 2 * d_model * context_length**2
    #   - AV calculation for all heads: num_heads times multiply A (QK^T) of shape (context_length x context_length) by V of shape (context_length x d_model/num_heads)
    #       = num_heads * 2 * context_length * context_length * d_model/num_heads =
    #       = 2 * d_model * context_length**2
    #   - O projection: multiply concatenated AV of shape (context_length x d_model) by O of shape (d_model x d_model)
    #       = 2 * context_length * d_model * d_model =
    #       = 2 * context_length * d_model**2
    #   - Total attention FLOPs
    #       = 6 * context_length * d_model**2 +
    #       + 2 * d_model * context_length**2 +
    #       + 2 * d_model * context_length**2 +
    #       + 2 * context_length * d_model**2 =
    #       = 4 * d_model * context_length**2 + 8 * context_length * d_model**2
    # - FFN: W1 projection + SiLU + Pairwise multiplication of SiLU and W1 projection + W3 projection
    #    - W1 and W3 projections: multiply X of shape (context_length * d_model) by W1 and W3 of shape (d_model * 4 * d_model)
    #       = 2 * 2 * context_length * d_model * 4 * d_model =
    #       = 16 * context_length * d_model**2
    #    - W2 projection: multiply GLU of shape (context_length * 4 * d_model) by W2 of shape (4 * d_model * d_model)
    #       = 2 * context_length * 4 * d_model * d_model =
    #       = 8 * context_length * d_model**2
    #    - Total FFN FLOPs:
    #       = 16 * context_length * d_model**2 +
    #       + 8 * context_length * d_model**2 =
    #       = 24 * context_length * d_model**2
    # - Total transformer blocks FLOPs:
    #       = num_layers * 
    #       * (4 * d_model * context_length**2 + 8 * context_length * d_model**2 +
    #       + 24 * context_length * d_model**2) =
    #       = 4 * num_layers * (d_model * context_length**2 + 8 * context_length * d_model**2)

    # Output embeddings FLOPs for one item: multiply output of transformer blocks of shape (context_length * d_model) by Output of shape (d_model * vocab_size)
    #       = 2 * context_length * d_model * vocab_size

    # Total FLOPs for performing one optimisation step =
    #       = 3 *
    #       * (4 * num_layers * (d_model * context_length**2 + 8 * context_length * d_model**2) +
    #       + 2 * context_length * d_model * vocab_size)) =
    #       = 12 * num_layers * (d_model * context_length**2 + 8 * context_length * d_model**2) + 6 * vocab_size * context_length * d_model

    total_compute = 12 * num_layers * (d_model * context_length**2 + 8 * context_length * d_model**2) + 6 * vocab_size * context_length * d_model

    # Seconds required to run optimiser steps time
    #       = steps * batch_size * total_compute / (mfu * theoretical_flops_per_sec)
    theoretical_flops_per_sec = 19.5e12
    mfu = 0.5
    batch_size = 1024
    steps = 4e5
    days_to_train_on_a100 = (steps * batch_size * total_compute / (mfu * theoretical_flops_per_sec)) / (60 * 60 * 24)

    return total_compute, days_to_train_on_a100


def adamw_accounting(
    model_name: str,
    batch_size: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    precision_in_bits: int = 32,
) -> dict:
    
    params_memory, grads_memory, opt_states_memory, activations_memory, total_memory, max_batch_80gb = adamw_memory(
        batch_size,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        precision_in_bits,
    )

    total_compute, days_to_train_on_a100 = adamw_compute(
        vocab_size,
        context_length,
        num_layers,
        d_model,
    )
    
    results = {
        "model": {
            "model_name": model_name,
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "context_length": context_length,
            "num_layers": num_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "precision_in_bits": precision_in_bits,
        },
        "memory": {
            "parameters": params_memory,
            "gradients": grads_memory,
            "optimiser_states": opt_states_memory,
            "activations": activations_memory,
            "total": total_memory,
            "max_batch_80gb": max_batch_80gb,
        },
        "compute": {
            "total": total_compute,
            "days_to_train_on_a100": days_to_train_on_a100,
        }
    }
    
    filename = f"results/accounting/AdamW_{model_name}_batch_{batch_size}.json"
    with open(filename, "w") as f:
        json.dump(results, f)

    print_adamw_accounting(results)


def print_adamw_accounting(
    json_data: dict
) -> None:
    model = json_data["model"]
    memory = json_data["memory"]
    compute = json_data["compute"]
    
    print(f"\nCONFIGURATION")
    print(f"    Name: {model['model_name']}")
    print(f"    Batch size: {model['batch_size']:,}")
    print(f"    Context length: {model['context_length']:,}")
    print(f"    Vocabulary size: {model['vocab_size']:,}")
    print(f"    Number of layers: {model['num_layers']}")
    print(f"    Model dimension: {model['d_model']:,}")
    print(f"    Number of heads: {model['num_heads']}")
    print(f"    Precision: {model['precision_in_bits']} bits")
    print(f"\nMEMORY")
    print(f"    Parameters: {memory['parameters']:.3f} GB")
    print(f"    Activations: {memory['activations']:.3f} GB")
    print(f"    Gradients: {memory['gradients']:.3f} GB")
    print(f"    Optimiser states: {memory['optimiser_states']:.3f} GB")
    print(f"    Total: {memory['total']:.3f} GB")
    print(f"    Max batch size for 80 GB: {memory['max_batch_80gb']:.1f}")
    print(f"\nCOMPUTE")
    print(f"    Per optimisation step: {compute['total']:,d} FLOPs")
    print(f"    Time to train on A100 with batch size 1024: {compute['days_to_train_on_a100']:,.0f} days\n")


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


def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    if t < T_w:
        # Warm-up phase: linear increase from 0 to alpha_max
        alpha = (t / T_w) * alpha_max
    elif t <= T_c:
        # Cosine annealing phase
        cosine_factor = 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))
        alpha = alpha_min + cosine_factor * (alpha_max - alpha_min)
    else:
        # Post-annealing phase: constant at alpha_min
        alpha = alpha_min
    
    return alpha


def gradient_clipping(
    parameters: list[nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> None:
    # Collect gradients as a flat vector
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if not grads:
        return  # no gradients to clip
    
    # Compute total L2 norm: sqrt(sum(grad^2))
    grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    
     # Compute scaling factor if total_norm > max_norm
    if grad_norm > max_norm:
        scale = max_norm / (grad_norm + eps)
        for g in grads:
            g.mul_(scale)  # scale in-place

    # Return gradient norm before clipping
    return grad_norm


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
    
    precision_in_bits = 32
    vocab_size = 50257
    batch_size = 4
    context_length = 1024

    # model_name = "GPT-2_small"
    # num_layers = 12
    # d_model = 786
    # num_heads = 12

    # model_name = "GPT-2_medium"
    # num_layers = 24
    # d_model = 1024
    # num_heads = 16

    # model_name = "GPT-2_large"
    # num_layers = 36
    # d_model = 1280
    # num_heads = 20

    model_name = "GPT-2_XL"
    num_layers = 48
    d_model = 1600
    num_heads = 25
    
    args = (model_name, batch_size, vocab_size, context_length, num_layers, d_model, num_heads, precision_in_bits)
    adamw_accounting(*args)
    
    