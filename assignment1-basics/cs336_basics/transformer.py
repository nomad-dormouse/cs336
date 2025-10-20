# 3 Transformer Language Model Architecture

# Problem (linear): Implementing the linear module (1 point)
# Problem (embedding): Implement the embedding module (1 point)
# Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)

from dotenv import load_dotenv
import torch
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Float, Int


# Load environment variables
load_dotenv()


class Linear(torch.nn.Module):


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
        self.W = torch.nn.Parameter(torch.empty(self.d_out, self.d_in, device=self.device, dtype=self.dtype))
        std = (2 / (self.d_in + self.d_out)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(
        self,
        x: Float[Tensor, " ... d_in"]
    ) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.W, " ... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):


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
        
        self.E = torch.nn.Parameter(torch.empty(self.vocab_size, self.d_model, device=self.device, dtype=self.dtype))
        std = (2 / (self.vocab_size + self.d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.E, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(
        self,
        token_ids: Int[Tensor, " ..."]
    ) -> Float[Tensor, " ... d_model"]:
        return self.E[token_ids]


class RMSNorm(torch.nn.Module):


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
        
        self.G = torch.nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))


    def forward(
        self,
        x: Float[Tensor, " d_batch d_seq d_model"]
    ) -> Float[Tensor, " d_batch d_seq d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = (x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()) * self.G
        
        return result.to(in_dtype)


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