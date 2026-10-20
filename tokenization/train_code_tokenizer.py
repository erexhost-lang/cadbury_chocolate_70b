"""
Custom Byte-Level BPE Tokenizer for Code
Trains a 128k or 64k vocabulary tokenizer preserving indentation, whitespaces,
and FIM (Fill-In-The-Middle) control tokens without destructive normalization.
"""

import argparse
import os
import sys
from typing import List, Generator

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.models import BPE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("train_code_tokenizer")

# Specialized Code & FIM Tokens
CODE_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|unk|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    # Whitespace indentation tokens (preserves multi-space indentation blocks)
    "    ",      # 4 spaces
    "  ",        # 2 spaces
    "\t",        # Tab
]


def train_code_bpe_tokenizer(
    input_files_or_iterator,
    output_dir: str = "tokenization/cadbury_tokenizer",
    vocab_size: int = 128256,
    min_frequency: int = 2,
) -> Tokenizer:
    """
    Train a Byte-Level BPE tokenizer tailored for multilingual code.
    
    Key Design Choices:
    1. ByteLevel pre-tokenizer: Starts with 256 byte tokens. Every raw byte is
       representable; no Out-Of-Vocabulary (OOV) tokens ever occur.
    2. No lowercase normalization: Variable casing ('MyClass' vs 'my_class')
       is semantically critical in programming languages.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Initializing Byte-Level BPE tokenizer with vocab target: {vocab_size:,}")

    # Base BPE model with unknown token fallback
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # ByteLevel splits text at byte level and handles whitespace preservation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=CODE_SPECIAL_TOKENS,
        show_progress=True,
    )

    logger.info("Training BPE merges over corpus...")
    if isinstance(input_files_or_iterator, list) and isinstance(input_files_or_iterator[0], str):
        tokenizer.train(files=input_files_or_iterator, trainer=trainer)
    else:
        tokenizer.train_from_iterator(input_files_or_iterator, trainer=trainer)

    save_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(save_path)
    logger.info(f"Tokenizer saved successfully to: {save_path}")
    logger.info(f"Actual vocabulary size: {tokenizer.get_vocab_size():,}")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom code tokenizer")
    parser.add_argument("--input-file", type=str, default="data/processed/clean_corpus.txt")
    parser.add_argument("--output-dir", type=str, default="tokenization/cadbury_tokenizer")
    parser.add_argument("--vocab-size", type=int, default=128256)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        train_code_bpe_tokenizer([args.input_file], args.output_dir, args.vocab_size)
    else:
        # Dummy demonstration run
        logger.info(f"Input file '{args.input_file}' not found. Showing demo with code strings.")
        demo_code = [
            "def add(a, b):\n    return a + b\n",
            "class NeuralNet:\n    def __init__(self):\n        pass\n",
            "fn main() {\n    println!(\"Hello World\");\n}\n",
        ]
        train_code_bpe_tokenizer(demo_code, args.output_dir, vocab_size=1000)
