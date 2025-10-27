# 3 Transformer Language Model Architecture

# Problem (linear): Implementing the linear module (1 point)
# Problem (embedding): Implement the embedding module (1 point)
# Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)
# Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)
# Problem (rope): Implement RoPE (2 points)
# Problem (softmax): Implement softmax (1 point)
# Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)
# Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)
# Problem (transformer_block): Implement a Transformer block (5 points)
# Problem (transformer_lm): Implementing the Transformer LM (3 points)
# Problem (transformer_accounting): Transformer LM resource accounting (5 points)


from dotenv import load_dotenv
import torch
from torch import Tensor, nn
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int, Bool
import json


# Load environment variables
load_dotenv()


def transformer_accounting(
    model_name: str,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    precision_in_bits: int,
) -> dict[str, int]:
    # EMBEDDINGS
    embeddings_params = vocab_size * d_model
    embeddings_flops = 0    # Lookup operation, not a matrix multiply

    # ALL TRANSFORMER BLOCKS

    # Single transformer block
    
    # Two pre-layer RMSNorms: one before attention and one before FFN sub-blocks
    rmsnorm_params_per_block = 2 * d_model  # 2 matrices of size d_model
    rmsnorm_flops_per_block = 0 # Does not involve matrix multiplies
    
    # Multi-head Self-Attention sub-block
    attention_params_per_block = 4 * d_model * d_model  # Q, K, V and Output matrices each of size (d_model x d_model)
    
    # Rule: Given A ∈ R^(m×n) and B ∈ R^(n×p), the matrix-matrix product AB requires 2mnp FLOPs
    qkv_proj_flops_per_block = 3 * 2 * context_length * d_model * d_model   # 3 times multiply (context_length x d_model) by (d_model x d_model)
    d_k = d_model // num_heads # Dimension of Q and K heads
    qk_flops_per_block = num_heads * 2 * context_length * context_length * d_k  # num_heads times multiply (context_length x d_k) by (d_k x context_length)
    av_flops_per_block = num_heads * 2 * context_length * context_length * d_k  # num_heads times multiply (context_length x context_length) by (context_length x d_k)
    output_proj_flops_per_block = 2 * context_length * d_model * d_model   # Multiply (context_length x d_model) by (d_model x d_model)

    attention_flops_per_block = qkv_proj_flops_per_block + qk_flops_per_block + av_flops_per_block + output_proj_flops_per_block

    # SwiGLU Feed-Forward Network sub-block
    ffn_params_per_block = 3 * d_model * d_ff  # W1 and W3 matrices of size (d_ff x d_model) and W2 matrix of size (d_model x d_ff)
    ffn_flops_per_block = 3 * 2 * context_length * d_model * d_ff # 2 times multiply (d_ff x d_model) by (d_model x context_length) and 1 time multiply (d_model x d_ff) by (d_ff x context_length)

    single_transformer_block_params = rmsnorm_params_per_block + attention_params_per_block + ffn_params_per_block
    single_transformer_block_flops = rmsnorm_flops_per_block + attention_flops_per_block + ffn_flops_per_block

    all_transformer_blocks_params = num_layers * single_transformer_block_params
    all_transformer_blocks_flops = num_layers * single_transformer_block_flops
    
    # FINAL LAYER NORM
    final_layernorm_params = d_model
    final_layernorm_flops = 0   # Does not involve matrix multiplies
    
    # LM HEAD
    lm_head_params = d_model * vocab_size
    lm_head_flops = 2 * context_length * d_model * vocab_size # Multiply (context_length x d_model) by (d_model x vocab_size)
    
    # TOTAL
    total_params = embeddings_params + all_transformer_blocks_params + final_layernorm_params + lm_head_params
    gb_per_param = precision_in_bits / (8 * 1024**3)
    total_flops = embeddings_flops + all_transformer_blocks_flops + final_layernorm_flops + lm_head_flops
    
    results = {
        "model": {
            "model_name": model_name,
            "vocab_size": vocab_size,
            "context_length": context_length,
            "num_layers": num_layers,
            "d_model": d_model,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "precision_in_bits": precision_in_bits,
        },
        "accounting": {
            "total": {
                "params": total_params,
                "memory": total_params * gb_per_param,
                "flops": total_flops,
            },
            "embeddings": {
                "params": embeddings_params,
                "memory": embeddings_params * gb_per_param,
                "flops": embeddings_flops,
            },
            "transformer_blocks": {
                "total": {
                    "params": all_transformer_blocks_params,
                    "memory": all_transformer_blocks_params * gb_per_param,
                    "flops": {
                        "Count": all_transformer_blocks_flops,
                        "Proportion": all_transformer_blocks_flops / total_flops,
                    },
                },
                "per_block": {
                    "total": {
                        "params": single_transformer_block_params,
                        "memory": single_transformer_block_params * gb_per_param,
                        "flops": single_transformer_block_flops,
                    },
                    "rmsnorm": {
                        "params": rmsnorm_params_per_block,
                        "memory": rmsnorm_params_per_block * gb_per_param,
                        "flops": rmsnorm_flops_per_block,
                    },
                    "attention": {
                        "params": attention_params_per_block,
                        "memory": attention_params_per_block * gb_per_param,
                        "flops": {
                            "Count": attention_flops_per_block,
                            "Proportion": attention_flops_per_block / single_transformer_block_flops,
                        },
                    },
                    "ffn": {
                        "params": ffn_params_per_block,
                        "memory": ffn_params_per_block * gb_per_param,
                        "flops": {
                            "Count": ffn_flops_per_block,
                            "Proportion": ffn_flops_per_block / single_transformer_block_flops,
                        },
                    },
                },
            },
            "final_layernorm": {
                "params": final_layernorm_params,
                "memory": final_layernorm_params * gb_per_param,
                "flops": final_layernorm_flops,
            },
            "lm_head": {
                "params": lm_head_params ,
                "memory": lm_head_params * gb_per_param,
                "flops": {
                    "Count": lm_head_flops,
                    "Proportion": lm_head_flops / total_flops,
                },
            },
        },
    }

    model_name = results["model"]["model_name"].replace(' ', '')
    context_length = results["model"]["context_length"]
    filename = f"results/accounting/{model_name}_context_{context_length}.json"
    with open(filename, "w") as f:
        json.dump(results, f)

    print_accounting(results)


def print_accounting(
    json_data: dict,
) -> None:
    model = json_data["model"]
    accounting = json_data["accounting"]
    
    print(f"\nCONFIGURATION\n")
    print(f"    Name: {model["model_name"]}")
    print(f"    Vocabulary size: {model["vocab_size"]:,}")
    print(f"    Context length: {model["context_length"]:,}")
    print(f"    Number of transformer layers: {model["num_layers"]}")
    print(f"    Model dimension: {model["d_model"]:,}")
    print(f"    Number of heads: {model["num_heads"]}")
    print(f"    Feed-forward dimension: {model["d_ff"]:,}")
    print(f"    Precision in bits: {model["precision_in_bits"]}\n")
    print(f"ACCOUNTING\n")
    print(f"    Total")
    print(f"        Parameters: {accounting["total"]["params"]:,}")
    print(f"        Memory: {accounting["total"]["memory"]:.3f} GB")
    print(f"        Compute: {accounting["total"]["flops"]:,} FLOPs\n")
    print(f"    Embeddings")
    print(f"        Parameters: {accounting["embeddings"]["params"]:,}")
    print(f"        Memory: {accounting["embeddings"]["memory"]:.3f} GB")
    print(f"        Compute: {accounting["embeddings"]["flops"]:,} FLOPs\n")
    print(f"    All transformer blocks")
    print(f"        Parameters: {accounting["transformer_blocks"]["total"]["params"]:,}")
    print(f"        Memory: {accounting["transformer_blocks"]["total"]["memory"]:.3f} GB")
    print(f"        Compute: {accounting["transformer_blocks"]["total"]["flops"]["Count"]:,} FLOPs\n")
    print(f"        Per block")
    print(f"            Parameters: {accounting["transformer_blocks"]["per_block"]["total"]["params"]:,}")
    print(f"            Memory: {accounting["transformer_blocks"]["per_block"]["total"]["memory"]:.3f} GB")
    print(f"            Compute: {accounting["transformer_blocks"]["per_block"]["total"]["flops"]:,} FLOPs\n")
    print(f"            RMSNorm")
    print(f"                Parameters: {accounting["transformer_blocks"]["per_block"]["rmsnorm"]["params"]:,}")
    print(f"                Memory: {accounting["transformer_blocks"]["per_block"]["rmsnorm"]["memory"]:.6f} GB")
    print(f"                Compute: {accounting["transformer_blocks"]["per_block"]["rmsnorm"]["flops"]:,} FLOPs\n")
    print(f"            Attention")
    print(f"                Parameters: {accounting["transformer_blocks"]["per_block"]["attention"]["params"]:,}")
    print(f"                Memory: {accounting["transformer_blocks"]["per_block"]["attention"]["memory"]:.3f} GB")
    print(f"                Compute: {accounting["transformer_blocks"]["per_block"]["attention"]["flops"]["Count"]:,} FLOPs\n")
    print(f"            FFN")
    print(f"                Parameters: {accounting["transformer_blocks"]["per_block"]["ffn"]["params"]:,}")
    print(f"                Memory: {accounting["transformer_blocks"]["per_block"]["ffn"]["memory"]:.3f} GB")
    print(f"                Compute: {accounting["transformer_blocks"]["per_block"]["ffn"]["flops"]["Count"]:,} FLOPs\n")
    print(f"    Final layernorm")
    print(f"        Parameters: {accounting["final_layernorm"]["params"]:,}")
    print(f"        Memory: {accounting["final_layernorm"]["memory"]:.6f} GB")
    print(f"        Compute: {accounting["final_layernorm"]["flops"]:,} FLOPs\n")
    print(f"    LM head")
    print(f"        Parameters: {accounting["lm_head"]["params"]:,}")
    print(f"        Memory: {accounting["lm_head"]["memory"]:.3f} GB")
    print(f"        Compute: {accounting["lm_head"]["flops"]["Count"]:,} FLOPs\n")
    print(f"COMPONENTS COMPUTE\n")
    print(f"    All transformer blocks: {accounting["transformer_blocks"]["total"]["flops"]["Proportion"] * 100:.1f}%")
    print(f"        Attention: {accounting["transformer_blocks"]["per_block"]["attention"]["flops"]["Proportion"] * 100:.1f}%")
    print(f"        FFN: {accounting["transformer_blocks"]["per_block"]["ffn"]["flops"]["Proportion"] * 100:.1f}%\n")
    print(f"    LM head: {accounting["lm_head"]["flops"]["Proportion"] * 100:.1f}% of total FLOPs\n")

def silu(
    x: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)

def softmax(
    x: Float[Tensor, "..."],
    dim: int,
    temperature: float = 1.0,
) -> Float[Tensor, "..."]:
    # Perform greedy decoding if temperature is 0
    if temperature == 0.0:
        output = torch.zeros_like(x)
        argmax_indices = x.argmax(dim=dim, keepdim=True)
        output.scatter_(dim=dim, index=argmax_indices, value=1.0)
        return output
    
    # Subtract the maximum value for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    x_scaled = x_shifted / temperature
    
    exp_x_scaled = torch.exp(x_scaled)
    sum_exp_x_scaled = exp_x_scaled.sum(dim=dim, keepdim=True)
    
    return exp_x_scaled / sum_exp_x_scaled


def scaled_dot_product_attention(
    Q: Float[Tensor, " batch_size ... seq_len_q d_k"],
    K: Float[Tensor, " batch_size ... seq_len_k d_k"],
    V: Float[Tensor, " batch_size ... seq_len_k d_v"],
    mask: Bool[Tensor, " seq_len_q seq_len_k"] | None = None,
) -> Float[Tensor, " batch_size ... seq_len_k d_v"]:
    # Calculate attention scores: Q.T @ K / sqrt(d_k)
    d_k = Q.size(-1)    
    scores = einsum(Q, K, " ... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / (d_k ** 0.5)
    
    # Apply mask if provided
    if mask is not None:
        # Set masked positions to large negative value so they become 0 after softmax
        scores = scores.masked_fill(mask == False, float('-inf'))
    
    # Apply softmax to get attention weights and then apply them to values
    attention_weights = softmax(scores, dim=-1)
    output = einsum(attention_weights, V, " ... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    
    return output


class Linear(nn.Module):


    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        
        # Initialise weight with truncated normal distribution
        # mean = 0, std**2 = 2/(d_in + d_out), truncated at [-3*std, 3*std]
        self.weight = nn.Parameter(torch.empty(self.d_out, self.d_in))
        std = (2 / (self.d_in + self.d_out)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(
        self,
        x: Float[Tensor, " ... d_in"]
    ) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, " ... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):


    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model))
        std = (2 / (self.vocab_size + self.d_model)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(
        self,
        token_ids: Int[Tensor, " ..."]
    ) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):


    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(self.d_model))


    def forward(
        self,
        x: Float[Tensor, " d_batch d_seq d_model"]
    ) -> Float[Tensor, " d_batch d_seq d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = (x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()) * self.weight
        
        return result.to(in_dtype)


class SwiGLU(nn.Module):


    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        
        if d_ff is None:
            # Calculate d_ff as 8/3 * d_model, rounded to nearest multiple of 64
            d_ff_raw = int((8 / 3) * self.d_model)
            self.d_ff = ((d_ff_raw + 63) // 64) * 64
        else:
            self.d_ff = d_ff
        
        # Three linear layers for SwiGLU
        self.w1 = Linear(d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)
        self.w3 = Linear(d_model, self.d_ff)


    def forward(
        self,
        x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        # Swish(W1(x))
        w1_out = self.w1(x)
        swish_out = silu(w1_out)
        
        # SwiGLU(x, W1, W2, W3) = W2(Swish(W1(x)) * W3(x))
        w3_out = self.w3(x)
        result = self.w2(swish_out * w3_out)
        
        return result


class RotaryPositionalEmbedding(nn.Module):


    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Pre-compute sin and cos values for efficiency
        pairs_indices = torch.arange(0, d_k, 2)
        freqs = 1.0 / (theta ** (pairs_indices / self.d_k))

        positions = torch.arange(max_seq_len)

        angles = einsum(positions, freqs, "pos, freq -> pos freq")
        
        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        
        # Register as non-persistent buffers (not saved in state_dict)
        self.register_buffer('cos_values', cos_values, persistent=False)
        self.register_buffer('sin_values', sin_values, persistent=False)


    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        # Extract cos and sin values for the given positions
        cos = self.cos_values[token_positions]
        sin = self.sin_values[token_positions]
        
        # Split x into even and odd parts
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # Apply rotation
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        
        # Interleave the rotated parts back together
        x_rope = torch.zeros_like(x)
        x_rope[..., ::2] = x_even_rot
        x_rope[..., 1::2] = x_odd_rot
        
        return x_rope


class MultiheadSelfAttention(nn.Module):


    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)


    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_model"]:
        # Project inputs to Q, K, V: (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape Q, K, V for multi-head: (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k/d_v)
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(Q.shape[-2])
            token_positions = repeat(token_positions, "... seq_len -> ... num_heads seq_len", num_heads=Q.shape[-3])
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Create causal mask: (seq_len, seq_len)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # Apply scaled dot-product attention with causal mask
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads: (..., num_heads, seq_len, d_v) -> (..., seq_len, d_model)
        attn_out = rearrange(attn_out, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)", num_heads=self.num_heads, d_v=self.d_v)
        
        # Apply output projection
        out = self.output_proj(attn_out)
        
        return out


class TransformerBlock(nn.Module):
    

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # First sublayer: Multi-head self-attention
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        
        # Second sublayer: Position-wise feed-forward network
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_model"],
    ) -> Float[Tensor, " ... seq_len d_model"]:
        # First sublayer: y = x + MultiHeadSelfAttention(RMSNorm(x))
        norm_x = self.ln1(x)
        attn_out = self.attn(norm_x)
        x = x + attn_out
        
        # Second sublayer: y = x + FFN(RMSNorm(x))
        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x


class TransformerLM(nn.Module):

    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        
        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(d_model)
        
        # Language model head (output projection)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(
        self,
        token_ids: Int[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len vocab_size"]:
        # Get token embeddings: (..., seq_len) -> (..., seq_len, d_model)
        x = self.token_embeddings(token_ids)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Apply final layer norm
        x = self.ln_final(x)
        
        # Apply language model head: (..., seq_len, d_model) -> (..., seq_len, vocab_size)
        logits = self.lm_head(x)
        
        return logits


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

    # model_name = "GPT-2_XL"
    # num_layers = 48
    # d_model = 1600
    # num_heads = 25
    
    # Context length experiment
    model_name = "GPT-2_XL"
    num_layers = 48
    d_model = 1600
    num_heads = 25
    context_length = 16384

    args = (model_name, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, precision_in_bits)
    transformer_accounting(*args)