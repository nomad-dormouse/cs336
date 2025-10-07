from tqdm import tqdm
import time
import os
from typing import BinaryIO
import regex as re
import multiprocessing

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
    start, end, file_path, pattern_special_tokens, pattern_pre_tokens, words = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        documents = re.split(pattern_special_tokens, chunk)
        for doc in documents:
            for match in re.finditer(pattern_pre_tokens, doc):
                words[match.group()] = words.get(match.group(), 0) + 1
    return words

def serial_pre_tokenisation(
    chunk_ranges,
    file_path,
    pattern_special_tokens,
    pattern_pre_tokens
) -> dict[str, int]:
    
    serial_start = time.time()

    words = {}
    for start, end in chunk_ranges:
        args = start, end, file_path, pattern_special_tokens, pattern_pre_tokens, words
        words_in_file = pre_tokenise_chunk(args)
        words = words_in_file
    
    serial_end = time.time()
    serial_time = serial_end - serial_start
    print(f"Serial processing time: {serial_time:.2f} seconds")

    return words_in_file

def parallel_pre_tokenisation(
    chunk_ranges,
    file_path,
    num_processes,
    pattern_special_tokens,
    pattern_pre_tokens
) -> dict[str, int]:
    
    parallel_start = time.time()

    words = {}
    chunk_args = [(start, end, file_path, pattern_special_tokens, pattern_pre_tokens, words) 
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

    parallel_end = time.time()
    parallel_time = parallel_end - parallel_start
    print(f"Parallel processing time: {parallel_time:.2f} seconds")

    return words_in_file

def pre_tokenise_file(
    file_path: str,
    num_chunks: int,
    pattern_special_tokens: str,
    pattern_pre_tokens: str
) -> dict:
    """
    Pre-tokenise each chunk in parallel and then merge results.
    """
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    
    words_in_file = parallel_pre_tokenisation(
        chunk_ranges,
        file_path,
        num_chunks,
        pattern_special_tokens,
        pattern_pre_tokens
    )

    words_in_file_serial = serial_pre_tokenisation(
        chunk_ranges,
        file_path,
        pattern_special_tokens,
        pattern_pre_tokens
    )
    
    pre_tokens = {
        tuple(char.encode('utf-8') for char in word): count
        for word, count in words_in_file.items()
    }

    return pre_tokens

if __name__ == "__main__":

    special_tok = ["<|endoftext|>"]
    escaped_special_tok = [re.escape(token) for token in special_tok]
    pattern_special_tok = "|".join(escaped_special_tok)

    pre_tokens = pre_tokenise_file(
        "../data/TinyStoriesV2-GPT4-train.txt",
        4,
        pattern_special_tok,
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    for i, (k, v) in enumerate(pre_tokens.items()):
        if i >= 10:
            break
        print(f"{k}: {v}")