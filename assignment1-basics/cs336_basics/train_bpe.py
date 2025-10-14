# 2.5 Experimenting with BPE Tokenizer Training
# Problem (train_bpe): BPE Tokenizer Training (15 points)

from tqdm import tqdm
import os
from typing import BinaryIO
import regex as re
import multiprocessing
import json
import sys
import os

# Add the tests directory to the path to import gpt2_bytes_to_unicode
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))
from common import gpt2_bytes_to_unicode


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenise_chunk(args) -> dict[str, int]:
    start, end, input_file, pattern_special_tokens, pattern_pre_tokens = args
    words = {}
    with open(input_file, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="surrogateescape")
        documents = re.split(pattern_special_tokens, chunk)
        for doc in documents:
            for match in re.finditer(pattern_pre_tokens, doc):
                words[match.group()] = words.get(match.group(), 0) + 1
    return words


def parallel_pre_tokenisation(
    chunk_ranges,
    input_file,
    num_processes,
    pattern_special_tokens,
    pattern_pre_tokens
) -> dict[str, int]:
    chunk_args = [(start, end, input_file, pattern_special_tokens, pattern_pre_tokens) 
                  for start, end in chunk_ranges]
    with multiprocessing.Pool(processes=num_processes) as pool:
        words_in_chunks = list(tqdm(
            pool.imap_unordered(pre_tokenise_chunk, chunk_args), 
            total=len(chunk_args),
            desc="Parallel pre-tokenisation"
        ))

    words_in_file = words_in_chunks[0]
    for i in range(1, len(words_in_chunks)):
        for token, count in words_in_chunks[i].items():
            words_in_file[token] = words_in_file.get(token, 0) + count

    return words_in_file


def pre_tokenise_file(
    input_file: str,
    num_chunks: int,
    pattern_special_tokens: str,
    pattern_pre_tokens: str
) -> dict[tuple[bytes], int]:
    """
    Pre-tokenise each chunk in parallel and then merge results.
    """
    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    
    words_in_file = parallel_pre_tokenisation(
        chunk_ranges,
        input_file,
        num_chunks,
        pattern_special_tokens,
        pattern_pre_tokens
    )
    
    words_pre_tokens = {
        tuple(bytes([b]) for b in word.encode("utf-8")): count
        for word, count in words_in_file.items()
    }

    return words_pre_tokens


def calculate_pair_freq(
    words: dict[tuple[bytes], int]
) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes], int]]:
    pairs_freq = {}
    pairs_in_words = {}
    for word, freq in words.items():
        seen_pairs = set()
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs_freq[pair] = pairs_freq.get(pair, 0) + freq
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pairs_in_words.setdefault(pair, []).append(word)
    return pairs_freq, pairs_in_words


def merge_pair_in_word(
    word: tuple[bytes],
    pair: tuple[bytes]
) -> tuple[tuple[bytes], list[int]]:
    merged_indices = []
    updated_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged_indices.append(i)
            updated_word.append(word[i] + word[i + 1])
            i += 2
        else:
            updated_word.append(word[i])
            i += 1
    return merged_indices, tuple(updated_word)


def train_bpe(
    input_file: str,
    vocab_size: int,
    special_tokens: list[str]
) -> (dict[int, bytes], list[tuple[bytes, bytes]]):

    pattern_special_tokens = "|".join([re.escape(token) for token in special_tokens])
    pattern_pre_tokens = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    num_chunks = 4
    
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])

    words = pre_tokenise_file(input_file, num_chunks, pattern_special_tokens, pattern_pre_tokens)
    merges = []
    
    num_merges = vocab_size - len(vocab)
    with tqdm(total=num_merges, desc="Training BPE") as pbar:
        while len(vocab) < vocab_size:
            pairs_freq, pairs_in_words = calculate_pair_freq(words)
            max_freq = max(pairs_freq.values())
            top_pairs = [pair for pair, freq in pairs_freq.items() if freq == max_freq] 
            lex_greater_pair = max(top_pairs)
            merges.append(lex_greater_pair)
            vocab[len(vocab)] = lex_greater_pair[0] + lex_greater_pair[1]
            
            words_to_update = pairs_in_words[lex_greater_pair]
            for word in words_to_update:
                freq = words.pop(word)
                _, updated_word = merge_pair_in_word(word, lex_greater_pair)
                words[updated_word] = freq
            
            pbar.update(1)

    return vocab, merges


def save_train_bpe_results(vocab, merges, vocab_filename='vocab.json', merges_filename='merges.txt'):
    # Create GPT-2 style mappings
    gpt2_bytes_to_unicode_map = gpt2_bytes_to_unicode()
    
    # Save vocabulary in GPT-2 JSON format
    vocab_dict = {}
    for token_id in vocab.keys():
        token_bytes = vocab[token_id]
        # Convert bytes to GPT-2 unicode representation
        token_str = ''.join([gpt2_bytes_to_unicode_map[byte] for byte in token_bytes])
        vocab_dict[token_str] = token_id
    with open(vocab_filename, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(vocab)} vocabulary entries to {vocab_filename}")
    
    # Save merges in GPT-2 format
    with open(merges_filename, 'w', encoding='utf-8') as f:
        for merge in merges:
            # Convert bytes to GPT-2 unicode representation
            token1_str = ''.join([gpt2_bytes_to_unicode_map[byte] for byte in merge[0]])
            token2_str = ''.join([gpt2_bytes_to_unicode_map[byte] for byte in merge[1]])
            f.write(f"{token1_str} {token2_str}\n")
    print(f"Saved {len(merges)} merges to {merges_filename}")


if __name__ == "__main__":
    file_name = "corpus"
    vocab_size = 500
    # file_name = "tinystories_sample_5M"
    # vocab_size = 1000
    file_extention = "en"
    
    vocab, merges = train_bpe(
        f"tests/fixtures/{file_name}.{file_extention}",
        vocab_size,
        ["<|endoftext|>"]
    )

    save_train_bpe_results(
        vocab,
        merges,
        vocab_filename=f"results/vocab_{file_name}_{vocab_size}.json",
        merges_filename=f"results/merges_{file_name}_{vocab_size}.txt",
    )