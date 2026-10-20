"""
Hugging Face Streaming Code Ingestion
Streams petabyte-scale code corpora (e.g. bigcode/the-stack-v2, starcoderdata)
without exhausting local disk storage.
"""

import argparse
import json
import os
import sys
from typing import Generator, Dict, Any, List, Optional
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("stream_hf_code")

# Supported primary languages for code pretraining
DEFAULT_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "rust",
    "go",
    "cpp",
    "c",
    "java",
    "sql",
    "markdown",
]


def stream_huggingface_code(
    dataset_name: str = "bigcode/the-stack-v2-train-smol",
    languages: Optional[List[str]] = None,
    split: str = "train",
    max_samples_per_lang: Optional[int] = 10000,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream code files from Hugging Face Hub on the fly.
    
    Args:
        dataset_name: Identifier on HF Hub (e.g., 'bigcode/starcoderdata')
        languages: List of language sub-directories/subsets to stream
        split: Dataset split ('train', 'validation')
        max_samples_per_lang: Stop after N samples per language (None for unlimited)
    """
    selected_languages = languages or DEFAULT_LANGUAGES
    logger.info(f"Initiating streaming for dataset '{dataset_name}' across {len(selected_languages)} languages.")

    for lang in selected_languages:
        logger.info(f"Connecting stream for language: '{lang}'...")
        count = 0
        try:
            # streaming=True fetches chunks on demand over HTTP without saving full dataset to disk
            dataset_stream = load_dataset(
                dataset_name,
                data_dir=lang,
                split=split,
                streaming=True,
                trust_remote_code=True,
            )

            for sample in dataset_stream:
                content = sample.get("content", sample.get("text", ""))
                if not content or len(content.strip()) < 50:
                    continue

                yield {
                    "source": dataset_name,
                    "language": lang,
                    "filepath": sample.get("max_stars_repo_path", sample.get("filepath", f"{lang}_file")),
                    "content": content,
                    "license": sample.get("license", "unknown"),
                    "stars": sample.get("max_stars_count", 0),
                }

                count += 1
                if max_samples_per_lang and count >= max_samples_per_lang:
                    logger.info(f"Reached limit of {max_samples_per_lang} samples for language: {lang}")
                    break

        except Exception as e:
            logger.warning(f"Skipping language '{lang}' due to stream error: {e}")
            continue


def export_stream_to_jsonl(
    output_file: str,
    dataset_name: str = "bigcode/the-stack-v2-train-smol",
    languages: Optional[List[str]] = None,
    max_samples: int = 5000,
):
    """Utility to consume a stream and write buffered JSONL files for downstream preprocessing."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    logger.info(f"Writing streamed data to: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        written = 0
        for record in stream_huggingface_code(dataset_name=dataset_name, languages=languages, max_samples_per_lang=max_samples):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 1000 == 0:
                logger.info(f"Streamed and wrote {written:,} code records...")

    logger.info(f"Completed! Total records saved: {written:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream code datasets from Hugging Face Hub")
    parser.add_argument("--dataset", type=str, default="bigcode/the-stack-v2-train-smol")
    parser.add_argument("--output", type=str, default="data/raw/hf_code_stream.jsonl")
    parser.add_argument("--languages", nargs="+", default=["python", "rust", "go", "typescript"])
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    export_stream_to_jsonl(
        output_file=args.output,
        dataset_name=args.dataset,
        languages=args.languages,
        max_samples=args.max_samples,
    )
