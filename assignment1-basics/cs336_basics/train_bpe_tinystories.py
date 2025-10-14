# 2.5 Experimenting with BPE Tokenizer Training
# Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

from train_bpe import train_bpe, save_train_bpe_results


def find_longest_token(vocab: dict[int, bytes]) -> tuple[str, int]:
    longest_token = ''
    max_length = 0
    for _, token_bytes in vocab.items():
        token_str = token_bytes.decode('utf-8', errors='replace')
        if len(token_str) > max_length:
            max_length = len(token_str)
            longest_token = token_str
    print(f"Longest token in vocabulary '{longest_token}' is {max_length} characters")


if __name__ == "__main__":    
    # file_name = "TinyStoriesV2-GPT4-valid"
    file_name = "TinyStoriesV2-GPT4-train"
    file_extention = "txt"
    vocab_size = 10000

    vocab, merges = train_bpe(
        f"data/{file_name}.{file_extention}",
        vocab_size,
        ["<|endoftext|>"]
    )
    
    save_train_bpe_results(
        vocab,
        merges,
        vocab_filename=f"results/vocab_{file_name}_{vocab_size}.json",
        merges_filename=f"results/merges_{file_name}_{vocab_size}.txt",
    )
    
    find_longest_token(vocab)