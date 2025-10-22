# 3 Training a Transformer LM

# Problem (cross_entropy): Implement Cross entropy

from dotenv import load_dotenv
import torch
from torch import Tensor
from jaxtyping import Float, Int


# Load environment variables
load_dotenv()


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

    # So, we can substruct max logit value from logits for numerical stability of exp(logits)
    logits_stable = logits - logits.max(dim=-1, keepdim=True)[0] # shape (..., vocab_size)

    # Get the logits for the target classes from the stable logits
    target_logits_stable = logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # shape (...)

    # Compute numerically stable log-sum-exp
    log_sum_exp_stable = torch.logsumexp(logits_stable, dim=-1) # shape (...)
    
    # Compute the cross-entropy loss for each prediction
    loss = -target_logits_stable + log_sum_exp_stable # shape (...)
    
    # Return average loss across all predictions
    return loss.mean() # shape (...) -> ()


if __name__ == "__main__":
    vocab_size = 50257
    d_ff = 6400
    precision_in_bits = 32
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

    args = (model_name, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, precision_in_bits)