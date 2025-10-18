# 2.6 BPE Tokenizer: Encoding and Decoding
# 2.7 Experiments

# Problem (tokenizer): Implementing the tokenizer (15 points)
# Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

from dotenv import load_dotenv
import os
import sys
from typing import Iterable, Iterator
import regex as re
import json
import random
import time
import numpy as np
import multiprocessing
from tqdm import tqdm

try:
    from train_bpe import find_chunk_boundaries
except ImportError:
    from .train_bpe import find_chunk_boundaries


# Load environment variables
load_dotenv()


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.id_to_token = vocab
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.id_to_token.items()}

        self.merges = {pair: i for i, pair in enumerate(merges)}

        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
            

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> "Tokenizer":

        # Load vocabulary from JSON file and convert string vocab to bytes
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        vocab = {
            token_id: token_str.encode("utf-8")
            for token_str, token_id in vocab_dict.items()
        }
        
        # Load merges from TXT file
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.strip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    token1_str, token2_str = cleaned_line.split(" ")
                    merge = (
                        token1_str.encode("utf-8"),
                        token2_str.encode("utf-8")
                    )
                    merges.append(merge)
        
        # Add special tokens if they don't exist in vocab
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token
        
        # Initialise a tokeniser
        tokenizer = cls(vocab, merges, special_tokens)
        print("Tokeniser:")
        print(f"- {vocab_filepath}")
        print(f"- {merges_filepath}")
        
        return tokenizer


    def apply_bpe(
        self,
        tokens: list[bytes]
    ) -> list[int]:
        while True:
            pairs = {(tokens[i], tokens[i+1]): i for i in range(len(tokens) - 1)}
            
            ranked_pairs = [(self.merges[pair], pair) for pair in pairs if pair in self.merges]
            if not ranked_pairs:
                break

            _, best_pair = min(ranked_pairs, key=lambda x: x[0])
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens


    def encode_word(
        self,
        word: str
    ) -> list[int]:
        tokens = [bytes([b]) for b in word.encode("utf-8")]
        merged_tokens = self.apply_bpe(tokens)

        # Convert tokens to token IDs
        token_ids = []
        for token in merged_tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Fallback: if token not found, split into individual bytes
                for byte_val in token:
                    byte_token = bytes([byte_val])
                    if byte_token in self.token_to_id:
                        token_ids.append(self.token_to_id[byte_token])

        return token_ids


    def encode(
        self,
        text: str
    ) -> list[int]:
        # Handle special tokens first as they should not be processed through BPE
        if self.special_tokens:
            # Sort special tokens by length (longest first) to handle overlapping tokens
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern_special_tokens = "|".join([re.escape(token) for token in sorted_special_tokens])
            
            # Split text by special tokens first
            parts = re.split(f"({pattern_special_tokens})", text)
        else:
            parts = [text]
        
        token_ids = []
        for part in parts:
            if part in self.special_tokens:
                # Handle special token directly
                byte_seq = part.encode("utf-8")
                if byte_seq in self.token_to_id:
                    token_ids.append(self.token_to_id[byte_seq])
                else:
                    # Fallback: if token not found, split into individual bytes
                    for byte_val in byte_seq:
                        byte_token = bytes([byte_val])
                        if byte_token in self.token_to_id:
                            token_ids.append(self.token_to_id[byte_token])
            else:
                # Process regular text through BPE
                words = []
                for match in re.finditer(os.getenv("REGEX_PRE_TOKENS"), part):
                    words.append(match.group())
                
                # For each word, apply BPE merges and then convert to tokens
                for word in words:
                    word_token_ids = self.encode_word(word)
                    token_ids.extend(word_token_ids)
        
        return token_ids


    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id


    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join(self.id_to_token[token_id] for token_id in ids)
        text = byte_seq.decode("utf-8", errors="replace")
        
        return text


def test_tokeniser(
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str],
    input_filepath: str
):
    tokenizer = Tokenizer.from_files(
        vocab_filepath,
        merges_filepath,
        special_tokens
    )

    with open(input_filepath, "r", encoding="utf-8") as f:
        text = f.readline().strip()
        if not text:  # If first line is empty, get next non-empty line
            for line in f:
                text = line.strip()
                if text:
                    break
    
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Vocabulary: {vocab_filepath}")
    print(f"Input: {input_filepath}")
    print(f"Original: {text}")
    print(f"Decoded:  {decoded}")
    print(f"Match: {text == decoded}")
    
    
def sample_docs(
    input_filepath: str,
    samples_count
) -> list[str]:
    # Sample random non-emply lines
    docs = []
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if len(docs) < samples_count:
                docs.append(line)
            else:
                # Draw a random index from 0 to i (inclusive)
                j = random.randint(0, i)
                if j < samples_count:
                    docs[j] = line
    return docs


def compression_ratio(
    text: str,
    token_ids: list[int]
) -> float:
    bytes_count = len(text.encode('utf-8'))
    tokens_count = len(token_ids)
    ratio = bytes_count / tokens_count if token_ids else 0    
    
    return ratio


def average_ratio(
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str],
    input_filepath: str,
    samples_count: int
):
    tokenizer = Tokenizer.from_files(
        vocab_filepath,
        merges_filepath,
        special_tokens
    )
    
    docs = sample_docs(
        input_filepath,
        samples_count
    )
    
    ratios = [compression_ratio(doc, tokenizer.encode(doc)) for doc in docs]
    average_ratio = sum(ratios)/len(ratios)
    
    print(f"\nVocabulary: {vocab_filepath}")
    print(f"Input: {input_filepath}")
    print(f"Average comparison ratio for random {samples_count} lines: {average_ratio:.2f} bytes/token")


def throughput(
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str],
    input_filepath: str,
    samples_count: int
) -> float:
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    
    docs = sample_docs(
        input_filepath,
        samples_count
    )

    total_time = 0
    for doc in docs:
        start_time = time.time()
        tokenizer.encode(doc)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    total_bytes = sum(len(doc.encode('utf-8')) for doc in docs)
    throughput = total_bytes / total_time
    
    # Time for tokenising Pile dataset
    pile_gigabytes = 825
    pile_bytes = pile_gigabytes * 1024**3
    pile_seconds = pile_bytes / throughput
    pile_days = (pile_seconds / 3600) / 24

    # Time for tokenising input file
    file_bytes = os.path.getsize(input_filepath)
    file_gigabytes = file_bytes / (1024**3)
    file_seconds = file_bytes / throughput
    file_minutes = file_seconds / 60
    
    print("Throughput:")
    print(f"- {throughput:,.0f} bytes/second")
    print(f"- tested on {len(docs)} samples of {total_bytes:,} bytes total")
    print("Time to tokenise:")
    print(f"- Pile dataset (825 GB): {pile_days:,.1f} days")
    print(f"- input file {input_filepath} ({file_gigabytes:.2f} GB): {file_minutes:.1f} minutes")


def tokenise_chunk(args) -> np.ndarray:
    tokenizer, input_filepath, start, end, chunk_id = args
    token_ids = []
    
    with open(input_filepath, "r", encoding="utf-8") as f:
        f.seek(start)
        chunk_data = f.read(end - start)
        
        lines = chunk_data.splitlines()
        for line in tqdm(lines, desc=f"Chunk {chunk_id}", unit="line", position=chunk_id, leave=False):
            line = line.strip()
            if line:  # Skip empty lines
                chunk_token_ids = tokenizer.encode(line)
                token_ids.extend(chunk_token_ids)
    
    # Convert to NumPy array with uint16 dtype
    token_array = np.array(token_ids, dtype=np.uint16)
    
    return token_array


def tokenise_chunks_parallel(
    tokenizer: "Tokenizer",
    input_filepath: str,
    num_chunks: int = 4
) -> np.ndarray:
    # Find chunk boundaries
    endoftext_token = os.getenv("ENDOFTEXT_TOKEN").encode("utf-8")
    with open(input_filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, endoftext_token)
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    
    # Process chunks in parallel
    chunk_args = [(tokenizer, input_filepath, start, end, i) 
                  for i, (start, end) in enumerate(chunk_ranges)]
    with multiprocessing.Pool(processes=len(chunk_ranges)) as pool:
        chunk_token_ids = pool.map(tokenise_chunk, chunk_args)
    
    # Combine all token IDs from all chunks
    file_token_ids = np.concatenate(chunk_token_ids)
    
    return file_token_ids


def tokenise_chunks_serial(
    tokenizer: "Tokenizer",
    input_filepath: str,
    num_serial_chunks: int = 12,
    num_parallel_chunks: int = 4
) -> np.ndarray:
    # Find sequential chunks boundaries
    endoftext_token = os.getenv("ENDOFTEXT_TOKEN").encode("utf-8")
    with open(input_filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_serial_chunks, endoftext_token)
    serial_chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    
    all_token_ids = []
    
    # Process each sequential chunk
    for i, (start, end) in enumerate(serial_chunk_ranges):
        print(f"Processing serial chunk {i+1}/{len(serial_chunk_ranges)}")
        
        # Extract chunk data to temporary file
        import tempfile
        temp_filepath = os.path.join(tempfile.gettempdir(), f"temp_chunk_{i}.txt")
        with open(input_filepath, "rb") as source_f:
            source_f.seek(start)
            chunk_data = source_f.read(end - start)
        with open(temp_filepath, "wb") as temp_f:
            temp_f.write(chunk_data)
        
        # Process temporary file in parallel
        chunk_token_ids = tokenise_chunks_parallel(
            tokenizer, 
            temp_filepath, 
            num_chunks=num_parallel_chunks
        )
        all_token_ids.append(chunk_token_ids)
            
        os.remove(temp_filepath)
    
    # Combine all token IDs from all serial chunks
    file_token_ids = np.concatenate(all_token_ids)
    
    return file_token_ids


def tokenise_file(
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str],
    input_filepath: str,
    file_size_limit_gb: int = 4
):
    tokenizer = Tokenizer.from_files(
        vocab_filepath,
        merges_filepath,
        special_tokens
    )

    # Check file size and choose processing method
    file_size_gb = os.path.getsize(input_filepath) / (1024**3)
    print("Input file:")
    print(f"- {input_filepath}")
    print(f"- {file_size_gb:.2f} GB")
    print("- split file into 4 chunks")
    if file_size_gb < file_size_limit_gb:
        print("- process chunks in parallel on 4 CPU cores")
        token_array = tokenise_chunks_parallel(tokenizer, input_filepath)
    else:
        print("- process chunks sequentially")
        print("- split each chunk into 4 sub-chunks to process in parallel on 4 CPU cores")
        token_array = tokenise_chunks_serial(tokenizer, input_filepath)
    
    # Save to file
    input_filename = os.path.basename(input_filepath)
    input_name, _ = os.path.splitext(input_filename)
    output_filename = input_name + "_tokenised" + ".npy"
    output_filepath = os.path.join("results", "tokenised", output_filename)
    np.save(output_filepath, token_array)
    
    print(f"Tokenised file:")
    print(f"- {output_filepath}")


if __name__ == "__main__":
    special_tokens = [os.getenv("ENDOFTEXT_TOKEN")]
    samples_count = 50
    
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
    
    # TS train with TS tokeniser
    vocab_filepath = "results/vocab_TinyStoriesV2-GPT4-train_10000.json"
    merges_filepath = "results/merges_TinyStoriesV2-GPT4-train_10000.txt"
    input_filepath = "data/TinyStoriesV2-GPT4-train.txt"

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

    # test_tokeniser(
    #     vocab_filepath,
    #     merges_filepath,
    #     special_tokens,
    #     input_filepath,
    # )

    # average_ratio(
    #     vocab_filepath,
    #     merges_filepath,
    #     special_tokens,
    #     input_filepath,
    #     samples_count
    # )
    # print("\n")

    # throughput(
    #     vocab_filepath,
    #     merges_filepath,
    #     special_tokens,
    #     input_filepath,
    #     samples_count
    # )
    # print("\n")

    tokenise_file(
        vocab_filepath,
        merges_filepath,
        special_tokens,
        input_filepath,
    )