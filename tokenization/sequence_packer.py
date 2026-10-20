"""
Constant-Length Sequence Packer (Multipacking)
Packs multiple variable-length code files into fixed-size context windows (e.g., 8192 tokens).

Why Multipacking is Crucial for 70B Training:
- Code files average 200–1200 tokens.
- Padding individual files to 8192 wastes ~85% of GPU tensor core compute on <pad> tokens.
- Packing achieves 100% compute efficiency with zero wasted padding tokens.
"""

import json
import os
import sys
from typing import Generator, List, Dict, Any, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("sequence_packer")


def pack_sequences_constant_length(
    tokenized_docs: Generator[List[int], None, None],
    max_seq_len: int = 8192,
    eos_token_id: int = 128001,
) -> Generator[Dict[str, Any], None, None]:
    """
    Concatenate tokenized documents into fixed-length blocks.
    
    Yields:
        Dictionary with:
          - 'input_ids': List[int] of exact length max_seq_len
          - 'labels': List[int] of exact length max_seq_len (for causal LM loss)
          - 'position_ids': List[int] position indices reset per document
    """
    buffer: List[int] = []
    doc_boundaries: List[int] = [0]  # Tracks start index of each document inside the packed buffer

    for tokens in tokenized_docs:
        # Append document tokens and ensure EOS boundary marker
        tokens_with_eos = tokens if tokens and tokens[-1] == eos_token_id else tokens + [eos_token_id]
        
        buffer.extend(tokens_with_eos)
        doc_boundaries.append(len(buffer))

        # Once buffer has enough tokens, emit packed chunks of max_seq_len
        while len(buffer) >= max_seq_len:
            packed_chunk = buffer[:max_seq_len]
            buffer = buffer[max_seq_len:]

            # Reconstruct reset position IDs so each document in the pack starts at pos 0
            position_ids = []
            curr_pos = 0
            # Track positions based on document boundaries
            for idx in range(max_seq_len):
                position_ids.append(curr_pos)
                curr_pos += 1
                if packed_chunk[idx] == eos_token_id:
                    curr_pos = 0  # Reset position for the next document in this packed sequence

            # For causal language modeling, labels are identical to input_ids
            # (loss computation in model handles the 1-position shift internally)
            yield {
                "input_ids": packed_chunk,
                "labels": list(packed_chunk),
                "position_ids": position_ids,
            }

            # Adjust document boundary markers for the remaining buffer
            doc_boundaries = [max(0, b - max_seq_len) for b in doc_boundaries if b > max_seq_len]
            if not doc_boundaries or doc_boundaries[0] != 0:
                doc_boundaries.insert(0, 0)


def export_packed_binaries(
    packed_generator: Generator[Dict[str, Any], None, None],
    output_bin_path: str,
    max_seq_len: int = 8192,
    dtype=np.uint32,
):
    """
    Saves packed sequences as flat memory-mapped binary files for high-speed dataloading.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    logger.info(f"Writing packed binary dataset to: {output_bin_path}")

    all_ids = []
    chunk_count = 0

    for sample in packed_generator:
        all_ids.extend(sample["input_ids"])
        chunk_count += 1
        if chunk_count % 1000 == 0:
            logger.info(f"Packed {chunk_count:,} sequences ({chunk_count * max_seq_len:,} total tokens)...")

    # Save to disk as memory-mappable binary
    arr = np.array(all_ids, dtype=dtype)
    arr.tofile(output_bin_path)
    logger.info(f"Finished packing! Total packed sequences: {chunk_count:,} | Binary size: {arr.nbytes / (1024**2):.1f} MB")


if __name__ == "__main__":
    # Demonstration with simulated tokenized code files
    fake_docs = [
        [10, 20, 30, 40],        # Doc 1 (4 tokens)
        [50, 60, 70, 80, 90],    # Doc 2 (5 tokens)
        [100, 101, 102],         # Doc 3 (3 tokens)
        [200, 201, 202, 203, 204, 205, 206],  # Doc 4 (7 tokens)
    ]

    print("Demonstrating sequence packing with max_seq_len=8:")
    for packed in pack_sequences_constant_length(iter(fake_docs), max_seq_len=8, eos_token_id=999):
        print("Packed Chunk:", packed["input_ids"])
        print("Position IDs:", packed["position_ids"])
