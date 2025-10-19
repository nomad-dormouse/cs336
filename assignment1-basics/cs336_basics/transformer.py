# 2.6 BPE Tokenizer: Encoding and Decoding
# 2.7 Experiments

# Problem (tokenizer): Implementing the tokenizer (15 points)
# Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

from dotenv import load_dotenv



# Load environment variables
load_dotenv()






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
    # vocab_filepath = "results/vocab_TinyStoriesV2-GPT4-train_10000.json"
    # merges_filepath = "results/merges_TinyStoriesV2-GPT4-train_10000.txt"
    # input_filepath = "data/owt_train.txt"