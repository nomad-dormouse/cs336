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

from dotenv import load_dotenv
import torch
from torch import Tensor, nn
from einops import einsum, rearrange
from jaxtyping import Float, Int, Bool


# Load environment variables
load_dotenv()


def softmax(
    x: Float[Tensor, "..."],
    dim: int,
) -> Float[Tensor, "..."]:
    # Subtract the maximum value for numerical stability
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    
    # Compute softmax: exp(x_shifted) / sum(exp(x_shifted))
    exp_x = torch.exp(x_shifted)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    return exp_x / sum_exp


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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.device = device
        self.dtype = dtype
        
        # Initialise weight with truncated normal distribution
        # mean = 0, std**2 = 2/(d_in + d_out), truncated at [-3*std, 3*std]
        self.weight = nn.Parameter(torch.empty(self.d_out, self.d_in, device=self.device, dtype=self.dtype))
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model, device=self.device, dtype=self.dtype))
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))


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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        
        if d_ff is None:
            # Calculate d_ff as 8/3 * d_model, rounded to nearest multiple of 64
            d_ff_raw = int((8 / 3) * self.d_model)
            self.d_ff = ((d_ff_raw + 63) // 64) * 64
        else:
            self.d_ff = d_ff
        
        # Three linear layers for SwiGLU
        self.w1 = Linear(d_model, self.d_ff, device=self.device, dtype=self.dtype)
        self.w2 = Linear(self.d_ff, d_model, device=self.device, dtype=self.dtype)
        self.w3 = Linear(d_model, self.d_ff, device=self.device, dtype=self.dtype)


    def forward(
        self,
        x: Float[Tensor, " ... d_model"]
    ) -> Float[Tensor, " ... d_model"]:
        # Swish(W1(x)) = W1(x) * Sigmoid(W1(x))
        w1_out = self.w1(x)
        swish_out =w1_out * torch.sigmoid(w1_out)
        
        # SwiGLU(x, W1, W2, W3) = W2(Swish(W1(x)) * W3(x)) = W2( (W1(x) * Sigmoid(W1(x))) * W3(x))
        w3_out = self.w3(x)
        result = self.w2(swish_out * w3_out)
        
        return result


class RotaryPositionalEmbedding(nn.Module):


    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        # Pre-compute sin and cos values for efficiency
        pairs_indices = torch.arange(0, d_k, 2, device=self.device)
        freqs = 1.0 / (theta ** (pairs_indices / self.d_k))

        positions = torch.arange(max_seq_len, device=self.device)

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
        theta: float,
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
    # vocab = {i: bytes([i]) for i in range(128)}
    # vocab[128] = b"<|endoftext|>"
    # merges = [(b"h", b"e"), (b"l", b"l"), (b"o", b" ")]
    # tokenizer = Tokenizer(vocab, merges, special_tokens)
    # text = "hello <|endoftext|><|endoftext|> world"

    # # Corpus with Corpus tokeniser
    # vocab_filepath = "results/vocab_corpus_500.json"
    # merges_filepath = "results/merges_corpus_500.txt"
    # input_filepath = "tests/fixtures/corpus.en"

    # # TS valid with TS tokeniser
    # vocab_filepath = "results/vocab_TinyStoriesV2-GPT4-valid_10000.json"
    # merges_filepath = "results/merges_TinyStoriesV2-GPT4-valid_10000.txt"
    # input_filepath = "data/TinyStoriesV2-GPT4-valid.txt"
    
    # # TS train with TS tokeniser
    # vocab_filepath = "results/vocab_TinyStoriesV2-GPT4-train_10000.json"
    # merges_filepath = "results/merges_TinyStoriesV2-GPT4-train_10000.txt"
    # input_filepath = "data/TinyStoriesV2-GPT4-train.txt"

    # # OWT valid with OWT tokeniser
    # vocab_filepath = "results/vocab_owt_valid_32000.json"
    # merges_filepath = "results/merges_owt_valid_32000.txt"
    # input_filepath = "data/owt_valid.txt"

    # # OWT train with OWT tokeniser
    # vocab_filepath = "results/vocab_owt_train_32000.json"
    # merges_filepath = "results/merges_owt_train_32000.txt"
    # input_filepath = "data/owt_train.txt"

    # OWT train with TS tokeniser
    vocab_filepath = "results/vocab_TinyStoriesV2-GPT4-train_10000.json"
    merges_filepath = "results/merges_TinyStoriesV2-GPT4-train_10000.txt"
    input_filepath = "data/owt_train.txt"