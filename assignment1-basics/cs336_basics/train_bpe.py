# 2.5 Experimenting with BPE Tokenizer Training

# Problem (train_bpe): BPE Tokenizer Training (15 points)
# Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
# Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

from dotenv import load_dotenv
import os
import sys
from typing import BinaryIO
from tqdm import tqdm
import regex as re
import multiprocessing
import json


# Load environment variables
load_dotenv()


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


def serial_pre_tokenisation(
    chunk_ranges: list[tuple[int, int]],
    input_file: str,
    pattern_special_tokens: str,
    pattern_pre_tokens: str
) -> dict[str, int]:
    words = {}
    for start, end in tqdm(chunk_ranges, desc="Serial pre-tokenisation"):
        chunk_words = pre_tokenise_chunk((start, end, input_file, pattern_special_tokens, pattern_pre_tokens))
        for token, count in chunk_words.items():
            words[token] = words.get(token, 0) + count
    return words
    

def parallel_pre_tokenisation(
    chunk_ranges: list[tuple[int, int]],
    input_file: str,
    pattern_special_tokens: str,
    pattern_pre_tokens: str
) -> dict[str, int]:
    chunk_args = [(start, end, input_file, pattern_special_tokens, pattern_pre_tokens) 
                  for start, end in chunk_ranges]
    with multiprocessing.Pool(processes=len(chunk_ranges)) as pool:
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
    pattern_special_tokens: str,
    pattern_pre_tokens: str,
    split_special_token: bytes = b'<|endoftext|>',
    num_chunks: int = 4
) -> (list[tuple[bytes]], list[int]):
    """
    Pre-tokenise each chunk and merge results.
    """
    file_size = os.path.getsize(input_file)
    print(f"Input file size: {file_size / (1024 ** 3):.2f} GB")

    with open(input_file, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_special_token)
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    
    if file_size < 1024 ** 3 * 4:
        words_in_file = parallel_pre_tokenisation(
            chunk_ranges,
            input_file,
            pattern_special_tokens,
            pattern_pre_tokens,
        )
    else:
        words_in_file = serial_pre_tokenisation(
            chunk_ranges,
            input_file,
            pattern_special_tokens,
            pattern_pre_tokens,
        )
    
    words = []
    frequencies = []
    for word, count in tqdm(words_in_file.items(), desc="Splitting words to bytes"):
        tokenised_word = tuple(bytes([b]) for b in word.encode("utf-8"))
        words.append(tokenised_word)
        frequencies.append(count)

    return words, frequencies


def calculate_pair_freq(
    words: list[tuple[bytes]],
    frequencies: list[int]
) -> dict[tuple[bytes], dict[str, int | list[tuple[bytes]]]]:
    pairs = {}
    
    for word_id, (word, freq) in enumerate(zip(words, frequencies)):
        seen_pairs = set()
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])

            if pair not in pairs:
                pairs[pair] = {"freq": 0, "words": []}
            
            pairs[pair]["freq"] += freq
            
            if pair not in seen_pairs:
                pairs[pair]["words"].append(word_id)
                seen_pairs.add(pair)

    return pairs


def update_after_merge(
    merged_pair: tuple[bytes],
    tokenised_words: list[tuple[bytes]],
    words_frequencies: list[int],
    merge_candidates: dict[tuple[bytes], dict[str, int | list[tuple[bytes]]]],
) -> (list[tuple[bytes]], list[int], dict[tuple[bytes], dict[str, int | list[tuple[bytes]]]]):
    
    words_to_update = merge_candidates.pop(merged_pair)["words"]

    for word_id in words_to_update:
        word = tokenised_words[word_id]
        word_freq = words_frequencies[word_id]
        
        updated_word = []
        i = 0
        new_pairs = {}
        
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == merged_pair:
                merged_token = word[i] + word[i + 1]
                updated_word.append(merged_token)
                
                if i >= 1:
                    neighbouring_left_pair = (word[i-1], word[i])
                    if neighbouring_left_pair in merge_candidates:
                        merge_candidates[neighbouring_left_pair]["freq"] -= word_freq

                    new_left_pair = (word[i-1], merged_token)
                    if new_left_pair not in new_pairs:
                        new_pairs[new_left_pair] = {"freq": 0, "words": []}    
                    new_pairs[new_left_pair]["freq"] += word_freq
                
                if i < len(word) - 2:            
                    neighbouring_right_pair = (word[i+1], word[i+2])
                    if neighbouring_right_pair in merge_candidates:
                        merge_candidates[neighbouring_right_pair]["freq"] -= word_freq

                    new_right_pair = (merged_token, word[i+2])
                    if new_right_pair not in new_pairs:
                        new_pairs[new_right_pair] = {"freq": 0, "words": []}    
                    new_pairs[new_right_pair]["freq"] += word_freq

                i += 2
            else:
                updated_word.append(word[i])
                i += 1
        
        updated_word_tuple = tuple(updated_word)

        tokenised_words[word_id] = updated_word_tuple
        
        for pair, info in new_pairs.items():
            if pair in merge_candidates:
                merge_candidates[pair]["freq"] += info["freq"]
                if updated_word_tuple not in merge_candidates[pair]["words"]:
                    merge_candidates[pair]["words"].append(word_id)
            else:
                info["words"].append(word_id)
                merge_candidates[pair] = info
    
    return tokenised_words, words_frequencies, merge_candidates


def train_bpe(
    input_file: str,
    vocab_size: int,
    special_tokens: list[str]
) -> (dict[int, bytes], list[tuple[bytes, bytes]]):

    pattern_special_tokens = "|".join([re.escape(token) for token in special_tokens])
    pattern_pre_tokens = os.getenv('REGEX_PRE_TOKENS')
    
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])

    words, frequencies = pre_tokenise_file(
        input_file,
        pattern_special_tokens,
        pattern_pre_tokens
    )

    merges = []
    
    pairs = calculate_pair_freq(words, frequencies)
    num_merges = vocab_size - len(vocab)

    with tqdm(total=num_merges, desc="Training BPE", unit="merge", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {percentage:.2f}%]") as pbar:
        while len(vocab) < vocab_size:

            max_freq = max(info["freq"] for info in pairs.values())
            top_pairs = [pair for pair, info in pairs.items() if info["freq"] == max_freq] 
            lex_greater_pair = max(top_pairs)

            merges.append(lex_greater_pair)
            vocab[len(vocab)] = lex_greater_pair[0] + lex_greater_pair[1]
            
            words, frequencies, pairs = update_after_merge(
                lex_greater_pair,
                words,
                frequencies,
                pairs
            )
            
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


def find_longest_token(vocab: dict[int, bytes]):
    longest_token = ''
    max_length = 0
    for _, token_bytes in vocab.items():
        token_str = token_bytes.decode('utf-8', errors='replace')
        if len(token_str) > max_length:
            max_length = len(token_str)
            longest_token = token_str
    print(f"Longest token in vocabulary: '{longest_token}' ({max_length} characters)")


if __name__ == "__main__":
    
    file_dir = "tests/fixtures"
    file_name = "corpus"
    vocab_size = 500
    file_extention = "en"

    # file_dir = "data"
    # file_name = "TinyStoriesV2-GPT4-valid"
    # file_extention = "txt"
    # vocab_size = 10000

    # file_dir = "data"
    # file_name = "TinyStoriesV2-GPT4-train"
    # file_extention = "txt"
    # vocab_size = 10000

    # file_dir = "data"
    # file_name = "owt_valid"
    # file_extention = "txt"
    # vocab_size = 32000

    # file_dir = "data"
    # file_name = "owt_train"
    # file_extention = "txt"
    # vocab_size = 32000

    vocab, merges = train_bpe(
        f"{file_dir}/{file_name}.{file_extention}",
        vocab_size,
        [os.getenv('ENDOFTEXT_TOKEN')]
    )

    save_train_bpe_results(
        vocab,
        merges,
        vocab_filename=f"results/vocab_{file_name}_{vocab_size}.json",
        merges_filename=f"results/merges_{file_name}_{vocab_size}.txt",
    )

    find_longest_token(vocab)