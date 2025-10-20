# 2.6 BPE Tokenizer: Encoding and Decoding
# 2.7 Experiments

# Problem (tokenizer): Implementing the tokenizer (15 points)
# Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

from dotenv import load_dotenv
import torch
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Float

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
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        # Initialise weight with truncated normal distribution
        # mean = 0, std**2 = 2/(d_in + d_out), truncated at [-3*std, 3*std]
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(
        self,
        x: Float[Tensor, " ... in_features"]
    ) -> torch.Tensor:
        return einsum(x, self.W, " ... in_features, out_features in_features -> ... out_features")


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